import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class WeightedMSE(nn.Module):
    '''
    A weighted MSE based on height in the atmosphere in Mm
    
    'loss_w_range': (tuple),                # range in Mm to weight loss function (not working since we switched to column mass)
    'loss_scale': (float),                  # scale weight loss function (not working since we switched to column mass)
    'height_vector': (np array),            # atmosphere height vector in Mm (not working since we switched to column mass)
    'channels': (int),                      # channels (energy levels) of input data of shape [ch, z, x, y]
    'features': (int),                      # number of depth points. z in [ch, z, x, y]
    'device': torch.device()                # either 'cuda' or 'cpu'
    '''
    def __init__(self, h_vec, channels, height, scale_rng, scale, device):
        super(WeightedMSE, self).__init__()
        self.lower = (np.abs(h_vec-scale_rng[0])).argmin() #actually higher index, z is reversed
        self.upper = (np.abs(h_vec-scale_rng[1])).argmin() #actually lower index
        self.scale = scale
        
        print(f'Losses from {h_vec[self.lower]:.3f} Mm (idx = {self.lower}) to {h_vec[self.upper]:.3f} Mm (idx = {self.upper})' 
              f' will be weighted with a factor of {self.scale}')
        
        weights = torch.ones(channels, height, device = device)
        weights[:, self.upper:self.lower+1] = scale
        self.weights =  weights/weights.sum()
    
    def forward(self, y_pred, y_true):
        loss = F.mse_loss(y_pred, y_true, reduction = 'none')
        loss = self.weights * loss
        loss = torch.mean(loss)
        return loss