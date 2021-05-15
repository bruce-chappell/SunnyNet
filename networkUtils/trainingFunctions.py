import os
import torch
import numpy as np

def train(params, model, dataLoaders):
    '''
    trains a given model
    
    model: Pytorch nn.Module instance
    dataLoaders = {'train': nn.DataLoader, 'val': nn.DataLoader} dataloaders for training and validation set
    
        params = {
        'data_path': (str),        # path to training data file from 1_build_training_set_multi_sim.py
        'save_folder': (str)       # path to model save folder
        'model_save': (str)        # model_name.pt   
        'early_stopping': (int),   # early stop tolerance
        'num_epochs': (int),       # epochs     
        'train_size': (int),       # size of training set
        'batch_size_train': (int), # training batch size
        'val_size': (int),         # size of validation set 
        'batch_size_val': (int)    # validation batch size
        'num_workers': (int)       # CPU threads to use
        'alpha': (float) / None    # how to weight conservation of mass with normal loss
    }
    '''
    full_path = os.path.join(params['save_folder'], params['model_save'])
    loss_dict = {'train':[], 'val':[]}
    no_improv = 0
    min_loss = np.Inf
    stop = False
    
    for epoch in range(params['num_epochs']):
        ## train forwards ##
        model.network.train()
        tr_loss = run_epoch('train', model, epoch, dataLoaders, params['alpha'])
        loss_dict['train'].append(tr_loss)
        ## eval forward ##
        with torch.no_grad():
            model.network.eval()
            val_loss = run_epoch('val', model, epoch, dataLoaders, params['alpha'])
            loss_dict['val'].append(val_loss)
        
        ## check los ##
        if val_loss < min_loss:
            no_improv = 0
            min_loss = val_loss
            print(f'New min loss of {min_loss:.4f}, saving checkpoint...')
            torch.save(model.network.state_dict(), full_path)
        else:
            no_improv += 1
            if (epoch + 1 > params['early_stopping']) and no_improv == params['early_stopping']:
                stop = True
        if stop == True:
            print(f'Early stopping condition met, stopping at epoch {epoch}...')
            break
            
    return loss_dict
    
def run_epoch(mode, model, cur_epoch, dataLoaders, alpha, verbose = True):
    '''
    Runs epoch given the params in train()
    '''

    epoch_loss = 0
    if verbose:
        print('-'*10, f'Epoch {cur_epoch}: {mode}', '-'*10)

    for i, instance in enumerate(dataLoaders[mode]):
        X = instance[0].to(model.device, non_blocking=True) #lte
        
        y_true = instance[1].to(model.device, non_blocking=True) #non lte

        #------------ FORWARD --------------#
        y_pred = model.network(X)
        batch_loss = model.loss_fxn(y_pred, y_true)

        if alpha:                                    #conservation of mass
            X_pop = (10**X[...,1,1]).sum(1)          #sums across all levels (-1, 400)
            y_pop = (10**y_pred[...,0,0]).sum(1)
            batchloss = alpha * torch.nn.MSELoss()(y_pop, X_pop) + (1 - alpha) * batch_loss #add conservation of mass loss to batchLoss
            
        #------------ BACKWARD ------------#
        if mode == 'train':
            model.optimizer.zero_grad()
            batch_loss.backward(retain_graph=True)
            model.optimizer.step()
        epoch_loss += batch_loss.item()
        if verbose:
            if i%10 ==0:
                print(f'Epoch {cur_epoch}- Batch {i} loss: {batch_loss.item()}')
    epoch_loss = epoch_loss / len(dataLoaders[mode])
    print(f"TOTAL {mode.upper()} LOSS = {epoch_loss:.8f}")
    return epoch_loss

