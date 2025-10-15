# -*- coding: utf-8 -*-



import numpy as np

seed = np.load('edl_MSE_PEMSD4_seed0.npz')
seed.files   # ['mis', 'width', 'mse', 'rmse', 'mae', 'prediction', 'truth']
temp = seed['prediction']
v = temp[0, :,:, :]     #[12, 3584, 307]
