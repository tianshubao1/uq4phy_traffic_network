from re import U
import torch
import numpy as np
import math
import torch.nn.functional as F
import scipy
import scipy.stats
from scipy.stats import t, norm, gamma
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def edl_loss(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
    
    pi = torch.tensor(np.pi)
        
    x1 = torch.log(pi/nu)*0.5
    x2 = -alpha*torch.log(2.*beta*(1.+ nu))
    x3 = (alpha + 0.5)*torch.log( nu*(target - gamma)**2 + 2.*beta*(1. + nu) )
    x4 = torch.lgamma(alpha) - torch.lgamma(alpha + 0.5)

    loss_value =  torch.abs(target - gamma)*(2*nu + alpha) * 0.1 #set default parameters as 0.1 
        
    objective_mse = torch.nn.MSELoss()

    loss = (x1 + x2 + x3 + x4).mean() + loss_value.mean() + objective_mse(gamma, target).mean()
    return loss


def underwood_loss(y_pred, x_true, y_true): #x_true[0] is normalized, x_true[1], x_true[2] not normalized, [speed, occupy, flow]

    v_f = np.load('data/sensor_graph/PEMSD8_v_f.npy')   # [1,1,170]
    rho_max = np.load('data/sensor_graph/PEMSD8_rho_max.npy')   # [1,1,170]
    v_f = torch.from_numpy(v_f).to(device)
    rho_max = torch.from_numpy(rho_max).to(device)
    #y_pred: torch.Size([12, 128, 170])  # 3 quantile number
    #print(y_pred.size())
    y_pred_mid = y_pred  #y_pred: torch.Size([12, 128, 170])
    x_true = x_true.view(12, 128, 170, 3)   #torch.Size([12, 128, 170, 3])          x_true speed occupancy is normalized,flow is not normalized
    occupancy = torch.unbind(x_true,3)[1]   #torch.Size([12, 128, 170])
    mask = (y_true != 0).float()
    mask /= mask.mean()
    underwood = v_f*torch.exp(-occupancy/rho_max)
    loss = (y_pred_mid - underwood) ** 2
    #print(underwood)
    loss = loss * mask
    loss[loss != loss] = 0
    return loss.mean()


def masked_mae_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = torch.abs(y_pred - y_true)
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def masked_mse_loss(y_pred, y_true):
    mask = (y_true != 0).float()
    mask /= mask.mean()
    loss = (y_pred  - y_true) ** 2
    loss = loss * mask
    # trick for nans: https://discuss.pytorch.org/t/how-to-set-nan-in-tensor-to-0/3918/3
    loss[loss != loss] = 0
    return loss.mean()

def student_t_mis(self,gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            y_true: torch.Tensor) -> torch.Tensor:

    mu = gamma
    sigma_sqr = beta / (nu*(alpha - 1))

    l = mu - sigma_sqr.sqrt()*1.96
    u = mu + sigma_sqr.sqrt()*1.96
    l = self.standard_scaler.inverse_transform(l)
    u = self.standard_scaler.inverse_transform(u)
    """
    pa1 = 2*alpha
    pa2 = gamma 
    pa3 = torch.sqrt((beta*(1+nu)/(nu*alpha)))   #Represent three parameters of student-t distribution

    h = pa3 * (torch.tensor(t.ppf(0.975, pa1.cpu())).cuda())
    l = pa2 - h
    u = pa2 + h
    mis_loss = u-l
    #second_part = -0.025*u + 0.025*l
    """
    """
    lp = torch.zeros(gamma.size()).cuda()
    up = torch.zeros(gamma.size()).cuda()

    confidence = 0.97625
    for i in range(10):
        h = pa3 * (torch.tensor(t.ppf(confidence, pa1.cpu())).cuda())
        l = pa2 - h
        u = pa2 + h
        confidence += 0.0025
        #print("lp.size = ",lp.size())
        #print("l.size = ",l.size())
        lp = lp + 0.1*l
        up = up + 0.1*u
    """
    rou=0.05
    rou=2./rou
    #second_part = second_part + 0.025*(up - lp)

    mis_loss = (u-l) + rou * torch.max(y_true - u , u-u) + rou * torch.max(l - y_true , u-u)
    return mis_loss.mean()

def width(self,gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            y_true: torch.Tensor) -> torch.Tensor:

    #mask = (y_true != 0).float()
    #mask /= mask.mean()
    """
    pa1 = 2*alpha
    pa2 = gamma 
    pa3 = torch.sqrt((beta*(1+nu)/(nu*alpha)))   #Represent three parameters of student-t distribution

    h = pa3 * (torch.tensor(t.ppf(0.975, pa1.cpu())).cuda())
    l = pa2 - h
    r = pa2 + h
    interval_width = (r-l)
    
    
    
    
    
    mu = gamma
    sigma_sqr = beta / (alpha - 1)

    """
    mu = gamma
    sigma_sqr = beta / (nu*(alpha - 1))
    l = mu - sigma_sqr.sqrt()*1.96
    u = mu + sigma_sqr.sqrt()*1.96
    l = self.standard_scaler.inverse_transform(l)
    u = self.standard_scaler.inverse_transform(u)
    
    return (u-l).mean()

def ECE_loss(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
            y_true: torch.Tensor) -> torch.Tensor:
    
    mu = gamma
    std = torch.sqrt((beta*(1+nu))/(nu*alpha))
    Y = y_true
    sample_num = 2*alpha
    confidences = np.linspace(0, 1, num=100)
    calibration_errors = []
    interval_acc_list = []
    for confidence in confidences:
        low_interval, up_interval = confidence_interval(mu, std, sample_num, confidence=confidence)
        
        hit = 0
        a = (low_interval <= Y).float()
        b = (Y <= up_interval).float()
        c = a*b

        hit = c.sum()
        
        interval_acc = hit.item()/a.numel()
        interval_acc_list.append(interval_acc)
        calibration_errors.append((confidence - interval_acc)**2)

    return np.mean(np.sqrt(calibration_errors))

def Gaussian_distribution_mis(gamma: torch.Tensor, nu: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor,
                y_true: torch.Tensor) -> torch.Tensor:

    mask = (y_true != 0).float()
    mask /= mask.mean()
    
    mu = gamma
    #mu = mu.unsqueeze(-1)

    sigam = beta / (alpha - 1)
    sigam = torch.pow(sigam , 0.5)
    #sigam = sigam.unsqueeze(-1)

    rou = 0.05


    lx = (-1.96) * sigam + mu
    ux = 1.96 * sigam + mu

    ans_mis = ux - lx
    ans_mis = ans_mis + 4 / rou * (sigam / (math.sqrt(2* math.pi)) * torch.pow(math.e,(-torch.pow(ux - mu,2) / (2 * torch.pow(sigam , 2)))))

    ans_mis = ans_mis * mask
    ans_mis[ans_mis != ans_mis] = 0

    losses = []
    losses.append(ans_mis.unsqueeze(0))
    loss = torch.mean(torch.sum(torch.cat(losses, dim=0), dim=0))
    
    return loss
