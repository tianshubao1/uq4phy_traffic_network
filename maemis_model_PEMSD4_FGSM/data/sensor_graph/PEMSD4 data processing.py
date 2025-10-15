# -*- coding: utf-8 -*-



import numpy as np
import pandas as pd



#---------------- adj matrix ----------------------------------------


adj_matrix = np.load('adjacency_matrix.npy' )  #speed, occupancy, flow
adj_matrix = np.exp(adj_matrix/1609.344)
adj_matrix = 1/adj_matrix
adj_matrix[adj_matrix == 1] = 0
np.save('PEMSD4_matrix.npy', adj_matrix)


#------------------- dataset -------------------------------------

data = np.load('pems04.npz')    #[16992, 307, 3]

# List available arrays
# print("Available keys in the dataset:")
# print(data.files)

if 'data' in data:
    dataset = data['data']
    print("\nShape of traffic data:", dataset.shape)


length = 16992
#-------------------------- trainning dataset ------------------------------------

train = dataset[:int(length*0.7),:,:]   #[11894,170,3]
train_x = np.zeros([int(length*0.7)-24,12,307,3])        
train_y = np.zeros([int(length*0.7)-24,12,307,3])

for i in range(0, int(length*0.7) - 24):    
    train_x[i,:,:,:] = train[i:i+12,:,:]   #[12,307,3]      
    train_y[i,:,:,:] = train[i+12:i+24,:,:]    #[12,307,3]

train_x[..., [0, 2]] = train_x[..., [2, 0]]
train_y[..., [0, 2]] = train_y[..., [2, 0]]

temp = train_x[1000, :, :, :]

#-------------------------- val dataset ------------------------------------

val = dataset[int(length*0.7):int(length*0.8),:,:]   #[1699,307,3]
val_x = np.zeros([1699-24,12,307,3])   #[11:1773]
val_y = np.zeros([1699-24,12,307,3])  
 
for i in range(0, 1699-24):    # 12499 - 23 = 12476
    val_x[i,:,:,:] = val[i:i+12,:,:]   #[12,307,3]      
    val_y[i,:,:,:] = val[i+12:i+24,:,:] 
    
val_x[..., [0, 2]] = val_x[..., [2, 0]]
val_y[..., [0, 2]] = val_y[..., [2, 0]]

#-------------------------- test dataset ------------------------------------

test = dataset[int(length*0.8):,:,:]    #[3399,307,3]
test_x = np.zeros([3399-24,12,307,3])   
test_y = np.zeros([3399-24,12,307,3]) 
  
for i in range(0, 3399-24):   
    test_x[i,:,:,:] = test[i:i+12,:,:]   #[12,307,3]     
    test_y[i,:,:,:] = test[i+12:i+24,:,:] 

test_x[..., [0, 2]] = test_x[..., [2, 0]]
test_y[..., [0, 2]] = test_y[..., [2, 0]]

#-----------------------------------------------------------------------------


np.savez('train.npz', x=train_x, y=train_y)    #float64
np.savez('val.npz', x=val_x, y=val_y)  #float64
np.savez('test.npz', x=test_x, y=test_y)   #float64


#-------------------------------- regression ------------------------------------

train = np.load('train.npz' )
val = np.load('val.npz' )
test = np.load('test.npz' )

# test.files
# [flow, density, speed]
X = train['x']
x = X[:, :, :, 1].reshape([-1, 1])       #occupancy
y = X[:, :, :, 0].reshape([-1, 1])       #speed


# temp = x[1000, :, :, :]

from sklearn.linear_model import LinearRegression
from matplotlib import pyplot as plt

# train_data = int(16992*0.7)

reg = LinearRegression().fit(x, y)
reg.score(x, y)       #0.673
reg.coef_  # -138.68
reg.intercept_    # 70.65, rho_max = 70.65/138.68 = 0.509
pred = reg.predict(x)   


plt.scatter(x, y, s = 0.05, label='ground truth')
plt.scatter(x, pred, s = 0.5, label='prediction')
plt.xlim(0, 1)
plt.ylim(0, 100)
plt.legend()
plt.xlabel('occupancy')
plt.ylabel('speed')

plt.show()



