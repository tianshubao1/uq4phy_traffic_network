import argparse
import numpy as np
import os
import sys
import yaml
import torch

from lib.utils import load_graph_data
from model.pytorch.dcrnn_supervisor import DCRNNSupervisor


def run_dcrnn(args):
    with open(args.config_filename) as f:
        supervisor_config = yaml.load(f)

        graph_pkl_filename = supervisor_config['data'].get('graph_pkl_filename')
        sensor_ids, sensor_id_to_ind, adj_mx = load_graph_data(graph_pkl_filename)

        supervisor = DCRNNSupervisor(adj_mx=adj_mx, **supervisor_config)
        mean_mis, mean_width, mean_mse, mean_rmse, mean_mae, outputs = supervisor.evaluate('test')

        # Save evaluation metrics and model predictions as compressed npz files
        np.savez_compressed(args.output_filename, 
                    mis=mean_mis, 
                    width=mean_width, 
                    mse=mean_mse, 
                    rmse=mean_rmse, 
                    mae=mean_mae, 
                    **outputs)


        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print('Using device:', device)
        #print("MAE : {}".format(mean_score))
        #print('Predictions saved as {}.'.format(args.output_filename))
        print(f"MAE : {mean_mae}")
        print(f"RMSE: {mean_rmse}")
        print(f"MIS : {mean_mis}")
        print(f"Width: {mean_width}")
        print(f"MSE : {mean_mse}")
        print('Predictions saved as {}.'.format(args.output_filename))



if __name__ == '__main__':
    sys.path.append(os.getcwd())
    parser = argparse.ArgumentParser()
    parser.add_argument('--use_cpu_only', default=False, type=str, help='Whether to run tensorflow on cpu.')
    parser.add_argument('--config_filename', default='gpde_edl_model_MSE_PEMSD8/data/model/pretrained/METR-LA/config.yaml', type=str,
                        help='Config file for pretrained model.')
    parser.add_argument('--output_filename', default='gpde_edl_model_MSE_PEMSD8/data/gpde_edl_MSE_PEMSD8_seed0_new.npz')
    args = parser.parse_args()
    run_dcrnn(args)
