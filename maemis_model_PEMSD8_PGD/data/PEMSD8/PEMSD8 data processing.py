# -*- coding: utf-8 -*-
"""
Created on Mon Aug 14 17:31:20 2023

@author: tians
"""



import numpy as np
# from matplotlib import pyplot as plt



dataset = np.load('pems8ndy.npy' )  #speed, occupancy, flow
# adj_matrix = np.load('pemsd8_adj_matrix.npy' )  
# adj_matrix = np.exp(adj_matrix/1609.344)
# adj_matrix = 1/adj_matrix
# np.save('PEMSD8_matrix.npy', adj_matrix)



# train = dataset[:int(17856*0.7),:,:2]   #[12499,170,2]
train = dataset[:int(17856*0.7),:,:]   #[12499,170,3]
# train_x = train[:-12,:,:]   #[12487,170,2]      META-LA:[23974, 12, 207, 2]
# train_y = train[12:,:,:]    #[12487,170,2]
train_x = np.zeros([12476,12,170,3])        #[11:12487], remove the first 11 and last 12 values
train_y = np.zeros([12476,12,170,3])

for i in range(0,12476):    # 12499 - 23 = 12476  very slow
    train_x[i:,:,:,:] = train[i:i+12,:,:]   #[12,170,2]      
    train_y[i:,:,:,:] = train[i+12:i+24,:,:]    #[12,170,2]

# val = dataset[int(17856*0.7):int(17856*0.8),:,:2]   #[1785,170,2]
val = dataset[int(17856*0.7):int(17856*0.8),:,:]   #[1785,170,3]
# val_x = val[:-12,:,:]   #[1773,170,2]
# val_y = val[12:,:,:]    #[1773,170,2]
val_x = np.zeros([1762,12,170,3])   #[11:1773]
val_y = np.zeros([1762,12,170,3])   
for i in range(0,1762):    # 12499 - 23 = 12476
    val_x[i:,:,:,:] = val[i:i+12,:,:]   #[12,170,2]      
    val_y[i:,:,:,:] = val[i+12:i+24,:,:] 

# test = dataset[int(17856*0.8):,:,:2]    #[3572,170,2]
test = dataset[int(17856*0.8):,:,:]    #[3572,170,3]
# test_x = test[:-12,:,:] #[3560,170,2]
# test_y = test[12:,:,:]  #[3560,170,2]
test_x = np.zeros([3549,12,170,3])   #[11:3560]
test_y = np.zeros([3549,12,170,3])   
for i in range(0,3549):    # 12499 - 23 = 12476
    test_x[i:,:,:,:] = test[i:i+12,:,:]   #[12,170,2]      
    test_y[i:,:,:,:] = test[i+12:i+24,:,:] 



# train_x = np.float32(train_x)
# train_y = np.float32(train_y)
# val_x = np.float32(val_x)
# val_y = np.float32(val_y)
# test_x = np.float32(test_x)
# test_y = np.float32(test_y)


np.savez('data/train.npz', x=train_x, y=train_y)    #float64
np.savez('data/val.npz', x=val_x, y=val_y)  #float64
np.savez('data/test.npz', x=test_x, y=test_y)   #float64

# train = np.load('train.npz' )
# val = np.load('val.npz' )
# test = np.load('test.npz' )

# test.files


# x = dataset[:, :, 1].reshape([-1, 1])       #occupancy
# y = dataset[:, :, 0].reshape([-1, 1])       #speed


# from sklearn.linear_model import LinearRegression


# train_data = int(17856*0.7)

# reg = LinearRegression().fit(x[train_data:], y[train_data:])
# reg.score(x[train_data:], y[train_data:])       #0.566
# reg.coef_  # -109.12
# reg.intercept_    # 70.86, rho_max = 70.86/109.12 = 0.649
# pred = reg.predict(x)   


# plt.scatter(x, y, s = 0.05, label='ground truth')
# plt.scatter(x, pred, s = 0.5, label='prediction')
# plt.xlim(0, 1)
# plt.ylim(0, 100)
# plt.legend()
# plt.xlabel('occupancy')
# plt.ylabel('speed')

# plt.show()


