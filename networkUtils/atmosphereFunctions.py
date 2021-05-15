import os
import numpy
import torch
import h5py
from collections import OrderedDict
from networkUtils.dataSets import PopulationDataset3d
from networkUtils.modelWrapper import Model
from torch.utils.data import DataLoader

def predict_populations(pop_path, train_data_path, config):
    '''
    Function to predict NLTE populations of prepared data file of
    3x3x400 LTE test points.
    
    train_data_path (str)      # path to training data for model, need it for data scaling
    pop_path (str)             # path to file of 3x3x400 testing data points from 1_build_solving_set.py
    
     config={       
        'cuda': (bool),                # whether to use CUDA for forward pass 
        'model': (str),                # class of model from modelArchitectures.py
        'model_path': (str),           # path to trained model
        'channels': (int),             # channels (energy levels) of input data of shape [ch, z, x, y]
        'features': (int),             # number of depth points. z in [ch, z, x, y]
        'mode': (str),                 # either 'testing' or 'training', going to be testing for this
        'multi_gpu_train': (bool),     # whether the model was traine on multiple GPUs or just 1
        'loss_fxn': (str),             # one of the loss functions from torch.nn
        'alpha': (float) or (None),    # weight of conservation of mass term in loss function
        'output_XY': (int),            # X/Y size of output populations
    }
    '''

    train_data = PopulationDataset3d(train_data_path, train = False)

    ##### LOAD MODEL #####
    model = Model(config)
    print(f'Loading model...')
    ## fix dict keys from multi gpu training ##
    if config['multi_gpu_train']:
        state_dict = torch.load(config['model_path'])
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove module.
            new_state_dict[name] = v
        model.network.load_state_dict(new_state_dict)
    else:
        model.network.load_state_dict(torch.load(config['model_path']))

    if config['cuda']: 
        model.network.to('cuda')
    else:
        model.network.to('cpu')
    model.network.eval()

    print(f'Loading/scaling data...')
    # pull out scaling numbers
    mu_inp = train_data.mu_inp
    std_inp = train_data.std_inp

    mu_out = train_data.mu_out
    std_out = train_data.std_out
    
    with h5py.File(pop_path, 'r') as f:
        lte = f['lte test windows'][:]
        non_lte = f['non lte test points'][:]
        z = f['z'][:]
        cmass_mean = f['column mass'][:]
        cmass_scale = f['column scale'][:]
        
    lte = (lte-mu_inp)/std_inp
    non_lte = (non_lte-mu_out)/std_out
    
    data = [list(a) for a in zip(lte,non_lte)]
    

    mu_out = torch.tensor(mu_out).to(model.device, torch.float)
    std_out = torch.tensor(std_out).to(model.device, torch.float)
    
    print(f'Forward pass of data throught model...')
    loader = DataLoader(data, batch_size = 256, pin_memory = True, num_workers=8)
    pred_list = []
    for i, point in enumerate(loader):
        with torch.no_grad():
            model.network.eval()
            X = point[0].to(model.device, torch.float, non_blocking=True) #lte
            y_true = point[1].to(model.device, torch.float, non_blocking=True) #non lte
            y_pred = model.network(X)
            loss = torch.nn.MSELoss()(y_pred, y_true)
            if config['alpha']:
                X_pop = (10**X[...,1,1]).sum(1)
                y_pop = (10**y_pred[...,0,0]).sum(1)
                loss = (1-config['alpha'])*loss + config['alpha']*torch.nn.MSELoss()(y_pop,X_pop)
                    
            y_pred = y_pred * std_out + mu_out
            pred_list.append(y_pred)
        if i%10 == 0:
            print(f'Batch {i} loss: {loss.item()}')
            
    
    print(f'Fixing up dimensions...')
    pred_list = torch.cat(pred_list, dim = 0)   
    pred_final = pred_list.squeeze(3).squeeze(3).detach().cpu().numpy()
    pred_final = numpy.transpose(pred_final,(0,2,1))

    
    dim = config['output_XY']
    dimz = config['features']
    dimc = config['channels']
    pred_final = pred_final.reshape((dim,dim,dimz,dimc))
    
    return pred_final, z, cmass_mean, cmass_scale