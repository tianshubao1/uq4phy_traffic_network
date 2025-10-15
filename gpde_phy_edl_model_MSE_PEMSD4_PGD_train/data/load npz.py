

import numpy as np

seed = np.load('gpde_phy_edl_MSE_PEMSD4_seed0_pgd_train_00005.npz')
seed.files   # ['mis', 'width', 'mse', 'rmse', 'mae', 'prediction', 'truth']
temp = seed['prediction']
v = temp[0, :,:, :]     #[12, 3584, 307]
# temp = seed['truth']
