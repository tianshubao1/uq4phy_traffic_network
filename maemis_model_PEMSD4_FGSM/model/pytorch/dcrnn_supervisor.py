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

    def evaluate(self, dataset='val', batches_seen=0):
        """
        Computes mean L1Loss
        :return: mean L1Loss
        """
        with torch.no_grad():
            self.dcrnn_model = self.dcrnn_model.eval()

            val_iterator = self._data['{}_loader'.format(dataset)].get_iterator()
            losses_mis = []
            losses_width = []
            losses_mse = []
            losses_mae = []

            y_truths = []
            y_preds = []

            for _, (x, y) in enumerate(val_iterator):
                x, y = self._prepare_data(x, y)
                #print("num = ", _ )
                output  = self.dcrnn_model(x)

                #if (dataset == 'test'):
                #    print("num = ",_)
                #    print("y_size = ",y.size())
                #    print("gamma_size = ",gamma.size())
                
                #print("Start evaluate mis !!")
                #st = time.time()
                losses_mis.append(self.compute_mis(output ,y).item())
                #ed = time.time()
                #print("mis_time = ",ed - st)

                #st = time.time()
                losses_width.append(self.compute_width(output ,y).item())
                #ed = time.time()
                #print("compute_width_time = ",ed -st)
                
                #st = time.time()
                losses_mse.append(self.compute_mse(output ,y).item())
                #ed = time.time()
                #print("mse_time = " ,ed - st)
                
                #st = time.time()
                losses_mae.append(self._compute_mae(output ,y).item())
                #ed = time.time()
                #print("mae_time = " ,ed - st)
                
                #st = time.time()
                #losses_ECE.append(self.compute_ece(gamma,nu,alpha,beta,y).item())
                #ed = time.time()
                #print("ece_time = ", ed - st)

                y_truths.append(y.cpu())
                y_preds.append(output .cpu())

            y_preds = np.concatenate(y_preds, axis=1)
            y_truths = np.concatenate(y_truths, axis=1)  # concatenate on batch dimension
            
            # scaled prediction and ground truth
            y_truths_scaled = []
            y_preds_scaled = []
            #for t in range(y_preds.shape[0]):
            for t in range(len(y_preds)):  
                y_truth = self.standard_scaler.inverse_transform(y_truths[t])
                y_pred = self.standard_scaler.inverse_transform(y_preds[t])
                y_truths_scaled.append(y_truth)
                y_preds_scaled.append(y_pred)

            loss_mis = np.mean(losses_mis)
            loss_width = np.mean(losses_width)
            loss_mse = np.mean(losses_mse)
            loss_rmse = np.sqrt(loss_mse)
            loss_mae = np.mean(losses_mae)

            return loss_mis,loss_width,loss_mse,loss_rmse,loss_mae,{'prediction': y_preds_scaled, 'truth': y_truths_scaled}

    def evaluate_fgsm(self, dataset='val', batches_seen=0,
                      epsilon=0.03, clip_min=None, clip_max=None):
        """
        FGSM 对抗评估（与 evaluate 返回保持一致）：
        返回: loss_mis, loss_width, loss_mse, loss_rmse, loss_mae, {'prediction': y_preds_scaled, 'truth': y_truths_scaled}
        """
        
        import numpy as np
        self.dcrnn_model.eval()

        val_iterator = self._data[f'{dataset}_loader'].get_iterator()

        losses_mis, losses_width, losses_mse, losses_mae = [], [], [], []
        y_truths, y_preds = [], []

        for _, (x, y) in enumerate(val_iterator):
            x, y = self._prepare_data(x, y)

            # ---- 生成 FGSM 对抗样本（基于 MSE 攻击）----
            x_adv = x.clone().detach().requires_grad_(True)
            self.dcrnn_model.zero_grad()

            out_gen = self.dcrnn_model(x_adv)          # 前向（带梯度）
            loss_gen = self.compute_mse(out_gen, y)    # 也可换成你训练用的 _compute_loss(out_gen, y)（若存在）
            loss_gen.backward()

            with torch.no_grad():
                x_adv = x_adv + epsilon * x_adv.grad.sign()
                if (clip_min is not None) and (clip_max is not None):
                    x_adv.clamp_(clip_min, clip_max)

            # ---- 在对抗样本上做推理与指标（无梯度）----
            with torch.no_grad():
                output = self.dcrnn_model(x_adv)
                losses_mis.append(self.compute_mis(output, y).item())
                losses_width.append(self.compute_width(output, y).item())
                losses_mse.append(self.compute_mse(output, y).item())
                losses_mae.append(self._compute_mae(output, y).item())

                y_truths.append(y.cpu())
                y_preds.append(output.cpu())

        # ---- 拼接 batch 维 ----
        y_preds  = np.concatenate(y_preds,  axis=1)
        y_truths = np.concatenate(y_truths, axis=1)

        # ---- 反标准化 ----
        y_truths_scaled, y_preds_scaled = [], []
        for t in range(len(y_preds)):
            y_truths_scaled.append(self.standard_scaler.inverse_transform(y_truths[t]))
            y_preds_scaled.append(self.standard_scaler.inverse_transform(y_preds[t]))

        # ---- 聚合指标 ----
        loss_mis  = float(np.mean(losses_mis))
        loss_width= float(np.mean(losses_width))
        loss_mse  = float(np.mean(losses_mse))
        loss_rmse = float(np.sqrt(loss_mse))
        loss_mae  = float(np.mean(losses_mae))

        return loss_mis, loss_width, loss_mse, loss_rmse, loss_mae, {
            'prediction': y_preds_scaled,
            'truth': y_truths_scaled
        }


    def _train(self, base_lr,
               steps, patience=50, epochs=100, lr_decay_ratio=0.1, log_every=1, save_model=1,
               test_every_n_epochs=10, epsilon=1e-8, **kwargs):
        # steps is used in learning rate - will see if need to use it?
        min_val_loss = float('inf')
        wait = 0
        optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)

        lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

        self._logger.info('Start training ...')

        # this will fail if model is loaded with a changed batch_size
        num_batches = self._data['train_loader'].num_batch
        self._logger.info("num_batches:{}".format(num_batches))

        batches_seen = num_batches * self._epoch_num

        for epoch_num in range(self._epoch_num, epochs):
            
            print("epoch_num = ",epoch_num)

            self.dcrnn_model = self.dcrnn_model.train()

            train_iterator = self._data['train_loader'].get_iterator()

            for _, (x, y) in enumerate(train_iterator):

                optimizer.zero_grad()

                x, y = self._prepare_data(x, y)

                output= self.dcrnn_model(x, y, batches_seen)

                #print("h2")

                if batches_seen == 0:
                    # this is a workaround to accommodate dynamically registered parameters in DCGRUCell
                    optimizer = torch.optim.Adam(self.dcrnn_model.parameters(), lr=base_lr, eps=epsilon)
                    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=steps,
                                                            gamma=lr_decay_ratio)

                loss = self._compute_loss(output,y)
                self._logger.debug(loss.item())

                batches_seen += 1
            #    with autograd.detect_anomaly():
                loss.backward()

                # gradient clipping - this does it in place
                torch.nn.utils.clip_grad_norm_(self.dcrnn_model.parameters(), self.max_grad_norm)
                optimizer.step()

            self._logger.info("epoch complete")
            lr_scheduler.step()
            self._logger.info("evaluating now!")


            # ------- FGSM 验证集 -------
            
            fgsm_eps = 0.03
            val_mis_adv, val_width_adv, val_mse_adv, val_rmse_adv, val_mae_adv, _ = \
                self.evaluate_fgsm(dataset='val',
                                   batches_seen=batches_seen,
                                   epsilon=fgsm_eps,
                                   clip_min=None, clip_max=None)
                                   
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
                                   epsilon=fgsm_eps,
                                   clip_min=None, clip_max=None)
            
            message_test_adv = (
                f"Epoch [{epoch_num}/{epochs}] ({batches_seen}) "
                f"test_mae_adv:{float(test_mae_adv):.4f}, test_mis_adv:{float(test_mis_adv):.4f}, "
                f"test_width_adv:{float(test_width_adv):.4f}, test_mse_adv:{float(test_mse_adv):.4f}, "
                f"test_rmse_adv:{float(test_rmse_adv):.4f}"
            )
            self._logger.info(message_test_adv)
            
            end_test_time = time.time()



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
        return mis_loss(y_pred, y_true)
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
