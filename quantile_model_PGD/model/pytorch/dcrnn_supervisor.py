import os
import time

import torch.autograd as autograd
import numpy as np
import torch
#from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.dcrnn_model import DCRNNModel
from model.pytorch.loss import width,mis_loss,masked_mse_loss,masked_mae_loss,quantile_loss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Random seed
random_seed = 0
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)


class DCRNNSupervisor:
    def __init__(self, adj_mx, **kwargs):
        self._kwargs = kwargs
        self._data_kwargs = kwargs.get('data')
        self._model_kwargs = kwargs.get('model')
        self._train_kwargs = kwargs.get('train')

        self.max_grad_norm = self._train_kwargs.get('max_grad_norm', 1.)

        # logging.
        self._log_dir = self._get_log_dir(kwargs)
        #self._writer = SummaryWriter('runs/' + self._log_dir)

        log_level = self._kwargs.get('log_level', 'INFO')
        self._logger = utils.get_logger(self._log_dir, __name__, 'info.log', level=log_level)

        # data set
        self._data = utils.load_dataset(**self._data_kwargs)
        self.standard_scaler = self._data['scaler']

        self.num_nodes = int(self._model_kwargs.get('num_nodes', 1))
        self.input_dim = int(self._model_kwargs.get('input_dim', 1))
        self.seq_len = int(self._model_kwargs.get('seq_len'))  # for the encoder
        self.output_dim = int(self._model_kwargs.get('output_dim', 1))
        self.use_curriculum_learning = bool(
            self._model_kwargs.get('use_curriculum_learning', False))
        self.horizon = int(self._model_kwargs.get('horizon', 1))  # for the decoder

        # setup model
        dcrnn_model = DCRNNModel(adj_mx, self._logger, **self._model_kwargs)
        self.dcrnn_model = dcrnn_model.cuda() if torch.cuda.is_available() else dcrnn_model
        self._logger.info("Model created")

        self._epoch_num = self._train_kwargs.get('epoch', 0)
        if self._epoch_num > 0:
            self.load_model()

    @staticmethod
    def _get_log_dir(kwargs):
        log_dir = kwargs['train'].get('log_dir')
        if log_dir is None:
            batch_size = kwargs['data'].get('batch_size')
            learning_rate = kwargs['train'].get('base_lr')
            max_diffusion_step = kwargs['model'].get('max_diffusion_step')
            num_rnn_layers = kwargs['model'].get('num_rnn_layers')
            rnn_units = kwargs['model'].get('rnn_units')
            structure = '-'.join(
                ['%d' % rnn_units for _ in range(num_rnn_layers)])
            horizon = kwargs['model'].get('horizon')
            filter_type = kwargs['model'].get('filter_type')
            filter_type_abbr = 'L'
            if filter_type == 'random_walk':
                filter_type_abbr = 'R'
            elif filter_type == 'dual_random_walk':
                filter_type_abbr = 'DR'
            run_id = 'dcrnn_%s_%d_h_%d_%s_lr_%g_bs_%d_%s/' % (
                filter_type_abbr, max_diffusion_step, horizon,
                structure, learning_rate, batch_size,
                time.strftime('%m%d%H%M%S'))
            base_dir = kwargs.get('base_dir')
            log_dir = os.path.join(base_dir, run_id)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        return log_dir

    def save_model(self, epoch):
        if not os.path.exists('models/'):
            os.makedirs('models/')

        config = dict(self._kwargs)
        config['model_state_dict'] = self.dcrnn_model.state_dict()
        config['epoch'] = epoch
        torch.save(config, 'models/epo%d.tar' % epoch)
        self._logger.info("Saved model at {}".format(epoch))
        return 'models/epo%d.tar' % epoch

    def load_model(self):
        self._setup_graph() # this is to make sure the model is built before loading the weights
        assert os.path.exists('models/epo%d.tar' % self._epoch_num), 'Weights at epoch %d not found' % self._epoch_num
        
        checkpoint = torch.load('models/epo%d.tar' % self._epoch_num, map_location='cpu')
        self.dcrnn_model.load_state_dict(checkpoint['model_state_dict'])
        self._logger.info("Loaded model at {}".format(self._epoch_num))

    def _setup_graph(self):
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['val_loader'].get_iterator()

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                output = self.dcrnn_model(x)
                break

    def train(self, **kwargs):
        kwargs.update(self._train_kwargs)
        return self._train(**kwargs)



    def evaluate_pgd(self, dataset='val', batches_seen=0,
                     epsilon=0.03, steps=2, alpha=None, random_start=False,
                     clip_min=None, clip_max=None):

        self.dcrnn_model.eval()

        it = self._data[f'{dataset}_loader'].get_iterator()
        losses_mis, losses_width, losses_mse, losses_mae = [], [], [], []
        y_truths, y_preds = [], []

        step_size = (epsilon / max(1, steps)) if (alpha is None) else alpha

        for _, (x, y) in enumerate(it):
            x, y = self._prepare_data(x, y)

            # ---- PGD 初始化 ----
            x0 = x.detach()
            if random_start:
                x_adv = x0 + (2 * torch.rand_like(x0) - 1.0) * epsilon
            else:
                x_adv = x0.clone()
            if (clip_min is not None) and (clip_max is not None):
                x_adv = torch.clamp(x_adv, clip_min, clip_max)

            # ---- PGD 迭代 ----
            for _ in range(steps):
                x_adv = x_adv.detach().clone().requires_grad_(True)
                self.dcrnn_model.zero_grad(set_to_none=True)

                out_gen = self.dcrnn_model(x_adv)          # 前向（带梯度）
                loss_gen = self.compute_mse(out_gen, y)    # 如需一致也可换成 self._compute_loss(out_gen, y)
                loss_gen.backward()

                with torch.no_grad():
                    # 梯度上升一步
                    x_adv = x_adv + step_size * x_adv.grad.sign()
                    # 投影回 L∞(x0, ε)
                    x_adv = torch.max(torch.min(x_adv, x0 + epsilon), x0 - epsilon)
                    # 合法范围裁剪
                    if (clip_min is not None) and (clip_max is not None):
                        x_adv.clamp_(clip_min, clip_max)

            # ---- 在对抗样本上评估（无梯度）----
            with torch.no_grad():
                output = self.dcrnn_model(x_adv)
                losses_mis.append(self.compute_mis(output, y).item())
                losses_width.append(self.compute_width(output, y).item())
                losses_mse.append(self.compute_mse(output, y).item())
                losses_mae.append(self._compute_mae(output, y).item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

        # 拼接到 batch 维
        y_preds  = np.concatenate(y_preds,  axis=1)
        y_truths = np.concatenate(y_truths, axis=1)

        # 反标准化
        y_truths_scaled, y_preds_scaled = [], []
        for t in range(len(y_preds)):
            y_truths_scaled.append(self.standard_scaler.inverse_transform(y_truths[t]))
            y_preds_scaled.append(self.standard_scaler.inverse_transform(y_preds[t]))

        # 指标
        loss_mis   = float(np.mean(losses_mis))
        loss_width = float(np.mean(losses_width))
        loss_mse   = float(np.mean(losses_mse))
        loss_rmse  = float(np.sqrt(loss_mse))
        loss_mae   = float(np.mean(losses_mae))

        return loss_mis, loss_width, loss_mse, loss_rmse, loss_mae, {
            'prediction': y_preds_scaled, 'truth': y_truths_scaled
        }


    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # ==== PGD 训练期参数（由 FGSM 改为 PGD） ====
        adv_use_pgd      = kwargs.get('adv_use_pgd', True)     # 是否启用训练期 PGD
        adv_eps          = kwargs.get('adv_eps', 0.03)         # L∞ 半径 ε
        adv_steps        = kwargs.get('adv_steps', 2)          # PGD 步数（默认 2）
        adv_alpha        = kwargs.get('adv_alpha', None)       # 每步步长；默认 ε/steps
        adv_random_start = kwargs.get('adv_random_start', True)# 随机起点
        adv_clip_min     = kwargs.get('adv_clip_min', None)    # 输入下界（如 0.0）
        adv_clip_max     = kwargs.get('adv_clip_max', None)    # 输入上界（如 1.0）
        if adv_alpha is None:
            adv_alpha = adv_eps / max(1, adv_steps)

        # 评估用 PGD 参数（默认与训练一致）
        pgd_eps_eval   = kwargs.get('pgd_eps_eval', adv_eps)
        pgd_steps_eval = kwargs.get('pgd_steps_eval', adv_steps)
        pgd_alpha_eval = kwargs.get('pgd_alpha_eval', None)

        # ==== 优化器 & 学习率 ====
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps, gamma=lr_decay_ratio)

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):
            print("epoch_num = ", epoch_num)

            self.dcrnn_model = self.dcrnn_model.train()
            train_iterator = self._data['train_loader'].get_iterator()

            for _, (x, y) in enumerate(train_iterator):
                # --------- 准备数据 ---------
                x, y = self._prepare_data(x, y)

                # --------- 1) 生成对抗样本（PGD，多步） ---------
                if adv_use_pgd:
                    x0 = x.detach()
                    if adv_random_start:
                        x_adv = x0 + (2 * torch.rand_like(x0) - 1.0) * adv_eps
                    else:
                        x_adv = x0.clone()
                    if (adv_clip_min is not None) and (adv_clip_max is not None):
                        x_adv = torch.clamp(x_adv, adv_clip_min, adv_clip_max)

                    for _ in range(adv_steps):
                        x_adv = x_adv.detach().clone().requires_grad_(True)
                        self.dcrnn_model.zero_grad(set_to_none=True)

                        out_gen = self.dcrnn_model(x_adv, y, batches_seen)
                        loss_gen = self._compute_loss(out_gen, y)  # 与训练同一损失
                        loss_gen.backward()                        # 得到 ∇_x

                        with torch.no_grad():
                            # 梯度上升一步
                            x_adv = x_adv + adv_alpha * x_adv.grad.sign()
                            # 投影回 L∞(x0, ε)
                            x_adv = torch.max(torch.min(x_adv, x0 + adv_eps), x0 - adv_eps)
                            # 合法范围裁剪
                            if (adv_clip_min is not None) and (adv_clip_max is not None):
                                x_adv.clamp_(adv_clip_min, adv_clip_max)

                    x_adv = x_adv.detach()
                else:
                    x_adv = x  # 关闭对抗训练则退化为 clean

                # --------- 2) 真正训练步：仅用 x_adv ---------
                optimizer.zero_grad()

                # accommodate dynamically registered parameters in DCGRUCell
                if batches_seen == 0:
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                                        gamma=lr_decay_ratio)

                output = self.dcrnn_model(x_adv, y, batches_seen)
                loss = self._compute_loss(output, y)
                self._logger.debug(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - in place
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)
                optimizer.step()

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")

            # ------- PGD 验证集 -------
            eval_alpha = pgd_alpha_eval if pgd_alpha_eval is not None else (pgd_eps_eval / max(1, pgd_steps_eval))
            val_mis_adv, val_width_adv, val_mse_adv, val_rmse_adv, val_mae_adv, _ = \
                self.evaluate_pgd(dataset='val',
                                  batches_seen=batches_seen,
                                  epsilon=pgd_eps_eval,
                                  steps=pgd_steps_eval,
                                  alpha=eval_alpha,
                                  clip_min=adv_clip_min, clip_max=adv_clip_max)

            message_val_adv = (
                f"Epoch [{epoch_num}/{epochs}] ({batches_seen}) "
                f"[PGD eps={pgd_eps_eval}, steps={pgd_steps_eval}] "
                f"val_mae_adv:{float(val_mae_adv):.4f}, val_mis_adv:{float(val_mis_adv):.4f}, "
                f"val_width_adv:{float(val_width_adv):.4f}, val_mse_adv:{float(val_mse_adv):.4f}, "
                f"val_rmse_adv:{float(val_rmse_adv):.4f}"
            )
            self._logger.info(message_val_adv)

            # ------- PGD 测试集 -------
            test_mis_adv, test_width_adv, test_mse_adv, test_rmse_adv, test_mae_adv, _ = \
                self.evaluate_pgd(dataset='test',
                                  batches_seen=batches_seen,
                                  epsilon=pgd_eps_eval,
                                  steps=pgd_steps_eval,
                                  alpha=eval_alpha,
                                  clip_min=adv_clip_min, clip_max=adv_clip_max)

            message_test_adv = (
                f"Epoch [{epoch_num}/{epochs}] ({batches_seen}) "
                f"[PGD eps={pgd_eps_eval}, steps={pgd_steps_eval}] "
                f"test_mae_adv:{float(test_mae_adv):.4f}, test_mis_adv:{float(test_mis_adv):.4f}, "
                f"test_width_adv:{float(test_width_adv):.4f}, test_mse_adv:{float(test_mse_adv):.4f}, "
                f"test_rmse_adv:{float(test_rmse_adv):.4f}"
            )
            self._logger.info(message_test_adv)

            # ------- Early Stopping / 保存（用 adv 指标）-------
            if val_mae_adv < min_val_loss:
                wait = 0
                if save_model:
                    model_file_name = self.save_model(epoch_num)
                    self._logger.info(
                        'Val loss_adv decrease from {:.4f} to {:.4f}, saving to {}'.format(
                            min_val_loss, val_mae_adv, model_file_name
                        )
                    )
                min_val_loss = val_mae_adv
            else:
                wait += 1
                if wait >= patience:
                    self._logger.info(
                        f'Early stopping at epoch {epoch_num} (no adv improvement for {patience} epochs).'
                    )
                    break





    def _prepare_data(self, x, y):
        x, y = self._get_x_y(x, y)
        x, y = self._get_x_y_in_correct_dims(x, y)
        return x.to(device), y.to(device)

    def _get_x_y(self, x, y):
        """
        :param x: shape (batch_size, seq_len, num_sensor, input_dim)
        :param y: shape (batch_size, horizon, num_sensor, input_dim)
        :returns x shape (seq_len, batch_size, num_sensor, input_dim)
                 y shape (horizon, batch_size, num_sensor, input_dim)
        """
        x = torch.from_numpy(x).float()
        y = torch.from_numpy(y).float()
        self._logger.debug("X: {}".format(x.size()))
        self._logger.debug("y: {}".format(y.size()))
        x = x.permute(1, 0, 2, 3)
        y = y.permute(1, 0, 2, 3)
        return x, y

    def _get_x_y_in_correct_dims(self, x, y):
        """
        :param x: shape (seq_len, batch_size, num_sensor, input_dim)
        :param y: shape (horizon, batch_size, num_sensor, input_dim)
        :return: x: shape (seq_len, batch_size, num_sensor * input_dim)
                 y: shape (horizon, batch_size, num_sensor * output_dim)
        """
        batch_size = x.size(1)
        x = x.view(self.seq_len, batch_size, self.num_nodes * self.input_dim)

        #print("x.size = ",x.size())
        #print("y.size = ",y.size())

        #y = torch.split(y,6)[0]
        y = y[..., :self.output_dim].view(self.horizon, batch_size,
                                          self.num_nodes * self.output_dim)
        return x, y

    def _compute_loss(self, y_pred,y_true):
        y_pred = self.standard_scaler.inverse_transform(y_pred)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return quantile_loss(y_pred, y_true)
    def _compute_mae(self, y_pred,y_true):
        y_pred = self.standard_scaler.inverse_transform(y_pred)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return masked_mae_loss(y_pred, y_true)
    def compute_mse(self, y_pred,y_true):
        y_pred = self.standard_scaler.inverse_transform(y_pred)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return masked_mse_loss(y_pred, y_true)
    def compute_mis(self, y_pred,y_true):
        y_pred = self.standard_scaler.inverse_transform(y_pred)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return mis_loss(y_pred, y_true)
    def compute_width(self, y_pred,y_true):
        y_pred = self.standard_scaler.inverse_transform(y_pred)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return width(y_pred, y_true)
