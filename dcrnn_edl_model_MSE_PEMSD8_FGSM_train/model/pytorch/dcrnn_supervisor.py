import os
import time

import torch.autograd as autograd
import numpy as np
import torch
#from torch.utils.tensorboard import SummaryWriter

from lib import utils
from model.pytorch.dcrnn_model import DCRNNModel
#from model.pytorch.loss import masked_mae_loss
from model.pytorch.loss import edl_loss
from model.pytorch.loss import masked_mae_loss,student_t_mis,width,masked_mse_loss,ECE_loss

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
        self._setup_graph()
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



    def evaluate_fgsm(self, dataset='test', batches_seen=0,
                      epsilon=0.03, clip_min=None, clip_max=None,
                      attack_loss='mse'):
        """
        FGSM 对抗评估（只在测试/验证阶段使用）
        返回：loss_mis, loss_width, loss_mse, loss_rmse, loss_mae, results
        results['prediction'] 为对抗输入下的 [gamma, nu, alpha, beta]（已反标准化）
        results['truth']      为真值（已反标准化）
        """
        self.dcrnn_model = self.dcrnn_model.eval()
        val_iterator = self._data[f'{dataset}_loader'].get_iterator()

        losses_mis, losses_width, losses_mse, losses_mae = [], [], [], []
        y_truths = []
        gamma_preds, nu_preds, alpha_preds, beta_preds = [], [], [], []

        print("FGSM eval | data mean:", self.standard_scaler.mean)
        print("FGSM eval | data std :", self.standard_scaler.std)

        for _, (x, y) in enumerate(val_iterator):
            x, y = self._prepare_data(x, y)

            # ------- 生成 x_adv（需要梯度） -------
            x_adv = x.clone().detach().requires_grad_(True)
            self.dcrnn_model.zero_grad()

            # 前向（带梯度）
            gamma_gen, nu_gen, alpha_gen, beta_gen = self.dcrnn_model(x_adv)

            # 选择攻击损失
            if attack_loss == 'full':
                loss_gen = self._compute_loss(gamma_gen, nu_gen, alpha_gen, beta_gen, y)
            else:  # 'mse'（默认）
                loss_gen = self.compute_mse(gamma_gen, y)

            loss_gen.backward()

            # FGSM 更新 + 合法范围裁剪
            with torch.no_grad():
                x_adv = x_adv + epsilon * x_adv.grad.sign()
                if (clip_min is not None) and (clip_max is not None):
                    x_adv.clamp_(clip_min, clip_max)

            # ------- 在 x_adv 上做推理与指标（无梯度） -------
            with torch.no_grad():
                gamma, nu, alpha, beta = self.dcrnn_model(x_adv)

                losses_mis.append(self.compute_mis(gamma, nu, alpha, beta, y).item())
                losses_width.append(self.compute_width(gamma, nu, alpha, beta, y).item())
                losses_mse.append(self.compute_mse(gamma, y).item())
                losses_mae.append(self._compute_mae(gamma, y).item())

                y_truths.append(y.cpu())
                gamma_preds.append(gamma.cpu())
                nu_preds.append(nu.cpu())
                alpha_preds.append(alpha.cpu())
                beta_preds.append(beta.cpu())

        # ------- 指标汇总 -------
        loss_mis = np.mean(losses_mis)
        loss_width = np.mean(losses_width)
        loss_mse = np.mean(losses_mse)
        loss_rmse = np.sqrt(loss_mse)
        loss_mae = np.mean(losses_mae)

        # 拼接 batch 维
        gamma_preds = np.concatenate(gamma_preds, axis=1)
        nu_preds    = np.concatenate(nu_preds,    axis=1)
        alpha_preds = np.concatenate(alpha_preds, axis=1)
        beta_preds  = np.concatenate(beta_preds,  axis=1)
        y_truths    = np.concatenate(y_truths,    axis=1)

        # 反标准化
        y_truths_scaled = []
        gamma_pred_scaled, nu_pred_scaled = [], []
        alpha_pred_scaled, beta_pred_scaled = [], []

        for t in range(len(gamma_preds)):
            y_truth   = self.standard_scaler.inverse_transform(y_truths[t])
            gamma_pred = self.standard_scaler.inverse_transform(gamma_preds[t])
            nu_pred    = self.standard_scaler.inverse_transform(nu_preds[t])
            alpha_pred = self.standard_scaler.inverse_transform(alpha_preds[t])
            beta_pred  = self.standard_scaler.inverse_transform(beta_preds[t])

            y_truths_scaled.append(y_truth)
            gamma_pred_scaled.append(gamma_pred)
            nu_pred_scaled.append(nu_pred)
            alpha_pred_scaled.append(alpha_pred)
            beta_pred_scaled.append(beta_pred)

        results = {
            'prediction': [gamma_pred_scaled, nu_pred_scaled, alpha_pred_scaled, beta_pred_scaled],
            'truth': y_truths_scaled,
            'fgsm': {
                'eps': float(epsilon),
                'attack_loss': attack_loss
            }
        }

        return loss_mis, loss_width, loss_mse, loss_rmse, loss_mae, results


    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # ---- FGSM 训练期超参（可通过 kwargs 覆盖）----
        adv_use_fgsm  = kwargs.get('adv_use_fgsm', True)     # 是否启用训练期 FGSM
        adv_eps       = kwargs.get('adv_eps', 0.03)          # 扰动强度（按数据尺度调整）
        adv_mix_clean = kwargs.get('adv_mix_clean', False)   # True: clean+adv 混合；False: 仅 adv
        adv_lambda    = kwargs.get('adv_lambda', 0.5)        # clean 权重（0~1），仅在混合模式下用
        adv_clip_min  = kwargs.get('adv_clip_min', None)     # 输入下界（如 0.0 或标准化下界）
        adv_clip_max  = kwargs.get('adv_clip_max', None)     # 输入上界（如 1.0 或标准化上界)

        # ---- 评估阶段 FGSM（保留你现有评估逻辑）----
        fgsm_eps_eval = kwargs.get('fgsm_eps_eval', adv_eps)

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

            start_train_time = time.time()

            for _, (x, y) in enumerate(train_iterator):

                # --------- 准备数据 ---------
                x, y = self._prepare_data(x, y)  # x:[12,128,510]  y:[12,128,170]

                # --------- 1) 生成 x_adv：FGSM（用与你训练一致的损失签名）---------
                if adv_use_fgsm:
                    x_adv = x.detach().clone().requires_grad_(True)
                    self.dcrnn_model.zero_grad()

                    # 前向得到梯度用的输出（与训练相同的调用与损失）
                    gamma_g, nu_g, alpha_g, beta_g = self.dcrnn_model(x_adv, y, batches_seen)
                    loss_gen = self._compute_loss(gamma_g, nu_g, alpha_g, beta_g, y)
                    loss_gen.backward()  # 仅为了拿 ∇_x

                    with torch.no_grad():
                        x_adv = x_adv + adv_eps * x_adv.grad.sign()
                        if (adv_clip_min is not None) and (adv_clip_max is not None):
                            x_adv.clamp_(adv_clip_min, adv_clip_max)
                    x_adv = x_adv.detach()  # 切断图
                else:
                    x_adv = x  # 关闭对抗训练则退化为 clean

                # --------- 2) 真正训练步：仅用 x_adv 或与 clean 混合 ---------
                optimizer.zero_grad()

                # accommodate dynamically registered parameters in DCGRUCell
                if batches_seen == 0:
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                        optimizer, milestones=steps, gamma=lr_decay_ratio
                    )

                if adv_use_fgsm and adv_mix_clean:
                    # clean 分支
                    gamma_c, nu_c, alpha_c, beta_c = self.dcrnn_model(x, y, batches_seen)
                    loss_clean = self._compute_loss(gamma_c, nu_c, alpha_c, beta_c, y)

                    # adv 分支
                    gamma_a, nu_a, alpha_a, beta_a = self.dcrnn_model(x_adv, y, batches_seen)
                    loss_adv  = self._compute_loss(gamma_a, nu_a, alpha_a, beta_a, y)

                    loss = adv_lambda * loss_clean + (1.0 - adv_lambda) * loss_adv

                else:
                    # 仅对抗样本（或 adv_use_fgsm=False 时就是 clean）
                    gamma_out, nu_out, alpha_out, beta_out = self.dcrnn_model(x_adv, y, batches_seen)
                    loss = self._compute_loss(gamma_out, nu_out, alpha_out, beta_out, y)

                self._logger.debug(loss.item())

                batches_seen += 1
                loss.backward()

                # gradient clipping - in place
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)
                optimizer.step()

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")
            end_train_time = time.time()

            # ------- FGSM 验证集 -------
            val_mis_adv, val_width_adv, val_mse_adv, val_rmse_adv, val_mae_adv, _ = \
                self.evaluate_fgsm(dataset='val',
                                   batches_seen=batches_seen,
                                   epsilon=fgsm_eps_eval,
                                   clip_min=adv_clip_min, clip_max=adv_clip_max)

            message_val_adv = (
                f"Epoch [{epoch_num}/{epochs}] ({batches_seen}) "
                f"val_mae_adv:{float(val_mae_adv):.4f}, val_mis_adv:{float(val_mis_adv):.4f}, "
                f"val_width_adv:{float(val_width_adv):.4f}, val_mse_adv:{float(val_mse_adv):.4f}, "
                f"val_rmse_adv:{float(val_rmse_adv):.4f}"
            )
            self._logger.info(message_val_adv)

            # ------- FGSM 测试集 -------
            test_mis_adv, test_width_adv, test_mse_adv, test_rmse_adv, test_mae_adv, _ = \
                self.evaluate_fgsm(dataset='test',
                                   batches_seen=batches_seen,
                                   epsilon=fgsm_eps_eval,
                                   clip_min=adv_clip_min, clip_max=adv_clip_max)

            message_test_adv = (
                f"Epoch [{epoch_num}/{epochs}] ({batches_seen}) "
                f"test_mae_adv:{float(test_mae_adv):.4f}, test_mis_adv:{float(test_mis_adv):.4f}, "
                f"test_width_adv:{float(test_width_adv):.4f}, test_mse_adv:{float(test_mse_adv):.4f}, "
                f"test_rmse_adv:{float(test_rmse_adv):.4f}"
            )
            self._logger.info(message_test_adv)

            end_test_time = time.time()

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
        #x = self.standard_scaler.inverse_transform(x)
        #y = self.standard_scaler.inverse_transform(y)
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

    def _compute_loss(self,gamma, nu, alpha, beta,y_true):
        #gamma = self.standard_scaler.inverse_transform(gamma)
        #nu = self.standard_scaler.inverse_transform(nu)
        #alpha = self.standard_scaler.inverse_transform(alpha)
        #beta = self.standard_scaler.inverse_transform(beta)
        #y_true = self.standard_scaler.inverse_transform(y_true)
        return edl_loss(gamma,nu,alpha,beta, y_true)
    def _compute_mae(self,gamma,y_true):
        gamma = self.standard_scaler.inverse_transform(gamma)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return masked_mae_loss(gamma,y_true)
    def compute_mse(self,gamma,y_true):
        gamma = self.standard_scaler.inverse_transform(gamma)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return masked_mse_loss(gamma,y_true)
    def compute_mis(self, gamma, nu, alpha, beta,y_true):
        #gamma = self.standard_scaler.inverse_transform(gamma)
        #nu = self.standard_scaler.inverse_transform(nu)
        #alpha = self.standard_scaler.inverse_transform(alpha)
        #beta = self.standard_scaler.inverse_transform(beta)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return student_t_mis(self,gamma,nu,alpha,beta,y_true)
    def compute_width(self, gamma, nu, alpha, beta,y_true):
        #gamma = self.standard_scaler.inverse_transform(gamma)
        #nu = self.standard_scaler.inverse_transform(nu)
        #alpha = self.standard_scaler.inverse_transform(alpha)
        #beta = self.standard_scaler.inverse_transform(beta)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return width(self,gamma, nu, alpha, beta, y_true)
    def compute_ece(self, gamma, nu, alpha, beta,y_true):
        gamma = self.standard_scaler.inverse_transform(gamma)
        nu = self.standard_scaler.inverse_transform(nu)
        alpha = self.standard_scaler.inverse_transform(alpha)
        beta = self.standard_scaler.inverse_transform(beta)
        y_true = self.standard_scaler.inverse_transform(y_true)
        return ECE_loss(gamma, nu, alpha, beta, y_true)

