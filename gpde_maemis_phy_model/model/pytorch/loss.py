from turtle import width
import torch
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def quantile_loss(y_pred, y_true):

    #print(y_pred.size())   # l(x) f(x) u(x)
    #print(y_true.size())

    mask = (y_true != 0).float()
    mask /= mask.mean()
    quantiles = [0.025, 0.5, 0.975]
    losses = []
    for i, q in enumerate(quantiles):
        errors =  y_true - torch.unbind(y_pred,3)[i]
        errors = errors * mask
        errors[errors != errors] = 0
        losses.append(torch.max((q-1) * errors, q * errors).unsqueeze(0))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    return loss
    
def underwood_loss(y_pred, x_true, y_true): #x_true[0] is normalized, x_true[1], x_true[2] not normalized, [speed, occupy, flow]

    v_f = np.load('data/sensor_graph/METR_LA_v_f.npy')   # [1,1,207]
    rho_max = np.load('data/sensor_graph/METR_LA_rho_max.npy')   # [1,1,207]
    v_f = torch.from_numpy(v_f).to(device)
    rho_max = torch.from_numpy(rho_max).to(device)
    #y_pred: torch.Size([12, 128, 207, 3])  # 3 quantile number
    y_pred_mid = torch.unbind(y_pred,3)[1]  #y_pred: torch.Size([12, 128, 207])
    #print(x_true.size())
    x_true = x_true.view(12, 128, 207, 2)   #torch.Size([12, 128, 207, 2])          x_true speed occupancy is normalized,flow is not normalized
    occupancy = torch.unbind(x_true,3)[1]   #torch.Size([12, 128, 207])
    #print(occupancy.size())
    mask = (y_true != 0).float()
    mask /= mask.mean()
    underwood = v_f*torch.exp(-occupancy/rho_max)
    loss = (y_pred_mid - underwood) ** 2
    #print(underwood)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()

    
def mis_loss(y_pred, y_true):

    #print(y_pred.size())   # l(x) f(x) u(x)
    #print(y_true.size())

    mask = (y_true != 0).float()
    mask /= mask.mean()
    quantiles = [0.025, 0.5, 0.975]
    rou = 0.05
    rou = 2.0 / rou

    l_x = torch.unbind(y_pred,3)[0]
    f_x = torch.unbind(y_pred,3)[1]
    u_x = torch.unbind(y_pred,3)[2]

    errors =  u_x - l_x + abs(y_true - f_x)
    #print("errors.size = ",errors.size())
    #print(torch.max(y_true - u_x , u_x-u_x).size())
    #print(torch.max(y_true - u_x , u_x-u_x).type())

    errors = errors + rou * torch.max(y_true - u_x , u_x-u_x) + rou * torch.max(l_x - y_true , u_x-u_x)
    errors = errors * mask
    errors[errors != errors] = 0

    losses = []
    losses.append(errors.unsqueeze(0))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    return loss
def masked_mae_loss(y_pred, y_true):
    y_pred = torch.unbind(y_pred,3)[1]
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()
def width(y_pred, y_true):

    l_x = torch.unbind(y_pred,3)[0]
    f_x = torch.unbind(y_pred,3)[1]
    u_x = torch.unbind(y_pred,3)[2]

    errors =  u_x - l_x
    #print("errors.size = ",errors.size())
    #print(torch.max(y_true - u_x , u_x-u_x).size())
    #print(torch.max(y_true - u_x , u_x-u_x).type())

    errors[errors != errors] = 0

    losses = []
    losses.append(errors.unsqueeze(0))
    width = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    return width

def masked_mse_loss(y_pred, y_true):
    y_pred = torch.unbind(y_pred,3)[1]
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = (y_pred  - y_true) ** 2
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()