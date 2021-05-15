import torch
import torch.nn as nn
from networkUtils.modelArchitectures import *
from networkUtils.lossFunctions import *

class Model():
    def __init__(self, params):
        '''
        Model wrapper class to setup a model depending on parameters.
        
        params = {
        
        'model': (str),                                    # class of model from modelArchitectures.py
        'optimizer': (str),                                # optimizer for model from torch.optim, currently only using 'Adam'
        'loss_fxn': (str),                                 # one of the loss functions from lossFunctions.py or torch.nn
        'learn_rate': (float),                             # starting learning rate
        'channels': (int),                                 # channels (energy levels) of input data of shape [ch, z, x, y]
        'features': (int),                                 # number of depth points. z in [ch, z, x, y]
        #'loss_w_range': (tuple),                          # range in Mm to weight loss function (not working since we switched to column mass)
        #'loss_scale': (float),                            # scale weight loss function (not working since we switched to column mass)
        #'height_vector': (np array),                      # atmosphere height vector in Mm (not working since we switched to column mass)
        'cuda': {'use_cuda': (bool), 'multi_gpu': (bool)}, # whether to use cuda and multi GPU when training
        'mode': (string)                                   # either 'training' or 'testing'
        
        }        
        '''


        ## plotting only needs 1 forward pass so this gets skipped ##
        if params['mode'] == 'training':
        
            ## Pick model architecture ##
            if params['model'] == 'Regressor':
                self.network = Regressor(params['channels'], params['features'])
            elif params['model'] == 'DeeperRegressor':
                self.network = DeeperRegressor(params['channels'], params['features'])
            elif params['model'] == 'RegressorBN':
                self.network = RegressorBN(params['channels'], params['features'])
                
            elif params['model'] == 'BasicRegressor3D_3x3':
                self.network = BasicRegressor3D_3x3(params['channels'], params['features'],3,3)
            elif params['model'] == 'BasicRegressor3D_5x5':
                self.network = BasicRegressor3D_5x5(params['channels'], params['features'],5,5)
                
            elif params['model'] == 'Trans_1x1':
                self.network = Trans_1x1(params['channels'], params['features'],1,1)
                
            elif params['model'] == 'Trans_3x3':
                self.network = Trans_3x3(params['channels'], params['features'],3,3)
            elif params['model'] == 'ColMass_3x3':
                self.network = ColMass_3x3(params['channels'], params['features'],3,3)
                
            elif params['model'] == 'Trans_3x3_Deep':
                self.network = Trans_3x3_Deep(params['channels'], params['features'],[3,3,3])
                
            elif params['model'] == 'Trans_3x3_ResNet':
                self.network = Trans_3x3_ResNet(params['channels'], params['features'],[4,4,4])
                
            elif params['model'] == 'Trans_3x3_ResNet_NoBN':
                self.network = Trans_3x3_ResNet_NoBN(params['channels'], params['features'],[4,4,4])
                
            elif params['model'] == 'Trans_5x5':
                self.network = Trans_5x5(params['channels'], params['features'],5,5)
            elif params['model'] == 'Trans_7x7':
                self.network = Trans_7x7(params['channels'], params['features'],7,7)
            elif params['model'] == 'Trans_Regressor':
                self.network = Trans_Regressor(params['channels'], params['features'])
            
            else:
                raise Exception("!!Invalid model architecture!!")

            ## set CPU/GPU ##
            if params['cuda']['use_cuda']:
                self.device = torch.device("cuda:0")
                if params['cuda']['multi_gpu']:    
                    if torch.cuda.device_count() > 1:
                        print(f" Using {torch.cuda.device_count()} GPUs")
                        self.network = nn.DataParallel(self.network)
                    else:
                        print(f"Using 1 GPU")
                else:
                    print(f"Using 1 GPU")
            else:
                self.device = torch.device("cpu")
                print(f"Using CPU")

            ## send to CPU/GPU ##
            self.network.to(self.device)

            ## set loss function ##
            if params['loss_fxn'] == 'WeightedMSE':
                self.loss_fxn = WeightedMSE(params['height_vector'], params['channels'], 
                                            params['features'], params['loss_w_range'], 
                                            params['loss_scale'], self.device)
            elif params['loss_fxn'] == 'MSELoss':
                self.loss_fxn = nn.MSELoss()
            else:
                raise Exception("!!Invalid loss function!!")

            ## set optimizer ##
            if params['optimizer'] == 'Adam':
                self.optimizer = torch.optim.Adam(self.network.parameters(), lr=params['learn_rate'])
            else:
                raise Exception("!!Invalid optimizer!!")
        
        ## plotting mode, just a forward pass
        elif params['mode'] == 'testing':
          
            ## Pick model architecture ##
            if params['model'] == 'Regressor':
                self.network = Regressor(params['channels'], params['features'])
            elif params['model'] == 'DeeperRegressor':
                self.network = DeeperRegressor(params['channels'], params['features'])
            elif params['model'] == 'RegressorBN':
                self.network = RegressorBN(params['channels'], params['features'])
                
            elif params['model'] == 'BasicRegressor3D_3x3':
                self.network = BasicRegressor3D_3x3(params['channels'], params['features'],3,3)
            elif params['model'] == 'BasicRegressor3D_5x5':
                self.network = BasicRegressor3D_5x5(params['channels'], params['features'],5,5)
                
            elif params['model'] == 'Trans_1x1':
                self.network = Trans_1x1(params['channels'], params['features'],1,1)
                
            elif params['model'] == 'Trans_3x3':
                self.network = Trans_3x3(params['channels'], params['features'],3,3)
            elif params['model'] == 'ColMass_3x3':
                self.network = ColMass_3x3(params['channels'], params['features'],3,3)
                
            elif params['model'] == 'Trans_3x3_Deep':
                self.network = Trans_3x3_Deep(params['channels'], params['features'],[3,3,3])
                
            elif params['model'] == 'Trans_3x3_ResNet':
                self.network = Trans_3x3_ResNet(params['channels'], params['features'],[4,4,4])
                
            elif params['model'] == 'Trans_3x3_ResNet_NoBN':
                self.network = Trans_3x3_ResNet_NoBN(params['channels'], params['features'],[4,4,4])    
                
            elif params['model'] == 'Trans_5x5':
                self.network = Trans_5x5(params['channels'], params['features'],5,5)
            elif params['model'] == 'Trans_7x7':
                self.network = Trans_7x7(params['channels'], params['features'],7,7)
            elif params['model'] == 'Trans_Regressor':
                self.network = Trans_Regressor(params['channels'], params['features'])
            else:
                raise Exception("!!Invalid model architecture!!")
                           
            ## set CPU/GPU ##
            if params['cuda']:
                self.device = "cuda"
            else:
                self.device = "cpu"
            
            ## set loss function ##
            if params['loss_fxn'] == 'WeightedMSE':
                self.loss_fxn = WeightedMSE(params['height_vector'], params['channels'], 
                                            params['features'], params['loss_w_range'], 
                                            params['loss_scale'], self.device)
            elif params['loss_fxn'] == 'MSELoss':
                self.loss_fxn = nn.MSELoss()
            else:
                raise Exception("!!Invalid loss function!!")       
        else:
            raise Exception("!! Invalid model mode")
        
        return