import os
import sys
import torch
import numpy as np
from torch.utils.data import DataLoader, SubsetRandomSampler
from networkUtils.modelWrapper import Model
from networkUtils.dataSets import PopulationDataset3d
from networkUtils.trainingFunctions import train

if __name__ == '__main__':
    
    ## model parameters ##
    params = {
        'model': 'ColMass_3x3',
        'optimizer': 'Adam',
        'loss_fxn': 'MSELoss',
        'learn_rate': 1e-3,
        'channels': 6,
        'features': 400,
        #'loss_w_range': (0,4),
        #'loss_scale': 3,
        'cuda': {'use_cuda': True, 'multi_gpu': False},
        'mode': 'training'
    }
    
    ## training configuration ##
    config = {
        'data_path': 'path/name.hdf5',
        'save_folder': 'path/',
        'model_save': 'cbh24_ColMass_3x3_single_50e_128b_2a_ComboData.pt',    
        'early_stopping': 5,
        'num_epochs': 50,        
        'train_size': 106250,   
        #'train size': 212500,
        'batch_size_train': 128,
        'val_size': 18750, 
        #'val_size': 37500,
        'batch_size_val': 128,
        'num_workers': 64,
        'alpha': 0.2 # to turn off make None
    }
    
    if os.path.exists(os.path.join(config['save_folder'], config['model_save'])):
        print(f'YO! Save path already exists! Exiting...')
        sys.exit(1)

    
    print('Python VERSION:', sys.version)
    print('pyTorch VERSION:', torch.__version__)
    print('CUDA VERSION: ', torch.version.cuda)
    print(f'CUDA available: {torch.cuda.is_available()}')
    if torch.cuda.is_available():
        print('GPU name:', torch.cuda.get_device_name())
        print(f'Number of GPUS: {torch.cuda.device_count()}')
    print(f"Using {params['model']} architecture...")
    
    print('Creating dataset...')
    tr_data = PopulationDataset3d(config['data_path'], train=True)
    params['height_vector'] = tr_data.z
    val_data = PopulationDataset3d(config['data_path'], train=False)
    
    print('Creating dataloaders...')
    loader_dict = {}

    train_loader = DataLoader(
        tr_data, 
        batch_size = config['batch_size_train'], 
        pin_memory = True, 
        num_workers = config['num_workers']
    )
    
    val_loader = DataLoader(
        val_data, 
        batch_size = config['batch_size_val'], 
        pin_memory = True, 
        num_workers = config['num_workers']
    )
    
    loader_dict['train'] = train_loader
    loader_dict['val'] = val_loader
    
    h_model = Model(params)
    epoch_loss = train(config, h_model, loader_dict)
    
    ## save epoch losses for plotting ##
    with open(f"{config['save_folder']}{config['model_save'][:-3]}_loss.txt", "w") as f:
        for i in range(len(epoch_loss['train'])):
            f.write(str(epoch_loss['train'][i]) + '   ' + str(epoch_loss['val'][i]) + '\n')