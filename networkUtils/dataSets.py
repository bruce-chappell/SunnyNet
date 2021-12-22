import h5py
import torch
from torch.utils.data import Dataset

class PopulationDataset3d(Dataset):
    '''
    Dataset wrapper for our [c,z,x,y] shaped input data
    
    also handles mu/std for scaling out inputs and outputs
    '''
    def __init__(self, path, train = True):
        
        self.path = path
            
        if train:
            self.idx_0 = 'lte training windows'
            self.idx_1 = 'non lte training points'
        else:
            self.idx_0 = 'lte test windows'
            self.idx_1 = 'non lte test points'
            
        with h5py.File(self.path, 'r') as f:
            self.mu_inp = f[self.idx_0].attrs['mu']
            self.std_inp = f[self.idx_0].attrs['std']
            self.len = f[self.idx_0].attrs['len']
            self.z = f[self.idx_0].attrs['z']
            
            self.mu_out = f[self.idx_1].attrs['mu']
            self.std_out = f[self.idx_1].attrs['std']
        
    def __len__(self):
        return self.len
    
    def __getitem__(self,idx):
        with h5py.File(self.path, 'r') as f:
            lte = f[self.idx_0][idx,...]
            n_lte = f[self.idx_1][idx,...]
            
        return lte, n_lte # lte -> X, n_lte -> y_hat 