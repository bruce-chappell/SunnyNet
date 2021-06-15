import torch
import torch.nn as nn
import torch.nn.functional as F


class SunnyNet_1x1(nn.Module):
    '''
    this is built to work with [6,400,1,1] input data and to output [6,400,1,1] outputs
    '''
    def __init__(self, channels, depth, height, width):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.height = height
        self.width = width
        
        self.conv1 = nn.Conv1d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = 3,
            padding = 1
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32, 
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 32, 
            out_channels =64,
            kernel_size = 3,
            padding = 0
        )
        self.conv4 = nn.Conv1d(
            in_channels = 64, 
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )

        self.max1 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max2 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max3 = nn.MaxPool1d(kernel_size=2, stride = 2)

        self.fc1 = nn.Linear(6144, 4700)
        self.fc2 = nn.Linear(4700, channels*depth)

        #self.dropout = nn.Dropout2d(p=0.5)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = x.squeeze(3).squeeze(3)
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max2(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max3(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.depth, 1, 1)      
        return x



class SunnyNet_3x3(nn.Module):
    '''
    this is built to work with [6,400,3,3] input data and to output [6,400,1,1] outputs
    '''
    def __init__(self, channels, depth, height, width):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.height = height
        self.width = width
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (3,3,3),
            padding = (1,0,0)
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32, 
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 32, 
            out_channels =64,
            kernel_size = 3,
            padding = 0
        )
        self.conv4 = nn.Conv1d(
            in_channels = 64, 
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )

        self.max1 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max2 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max3 = nn.MaxPool1d(kernel_size=2, stride = 2)

        self.fc1 = nn.Linear(6144, 4700)
        self.fc2 = nn.Linear(4700, channels*depth)

        #self.dropout = nn.Dropout2d(p=0.5)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        
        x = self.conv1(x).squeeze(3).squeeze(3)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max2(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max3(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.depth, 1, 1)      
        return x

class SunnyNet_5x5(nn.Module):
    '''
    this is built to work with [6,400,5,5] input data and to output [6,400,1,1] outputs
    '''
    def __init__(self, channels, depth, height, width):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.height = height
        self.width = width
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (5,5,5),
            padding = (2,0,0)
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32, 
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 32, 
            out_channels =64,
            kernel_size = 3,
            padding = 0
        )
        self.conv4 = nn.Conv1d(
            in_channels = 64, 
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )

        self.max1 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max2 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max3 = nn.MaxPool1d(kernel_size=2, stride = 2)

        self.fc1 = nn.Linear(6144, 4700)
        self.fc2 = nn.Linear(4700, channels*depth)

        #self.dropout = nn.Dropout2d(p=0.5)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        
        x = self.conv1(x).squeeze(3).squeeze(3)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max2(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max3(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.depth, 1, 1)      
        return x
    

class SunnyNet_7x7(nn.Module):
    '''
    this is built to work with [6,400,7,7] input data and to output [6,400,1,1] outputs
    '''
    def __init__(self, channels, depth, height, width):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.height = height
        self.width = width
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (7,7,7),
            padding = (3,0,0)
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32, 
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 32, 
            out_channels =64,
            kernel_size = 3,
            padding = 0
        )
        self.conv4 = nn.Conv1d(
            in_channels = 64, 
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )

        self.max1 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max2 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max3 = nn.MaxPool1d(kernel_size=2, stride = 2)

        self.fc1 = nn.Linear(6144, 4700)
        self.fc2 = nn.Linear(4700, channels*depth)

        #self.dropout = nn.Dropout2d(p=0.5)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        
        x = self.conv1(x).squeeze(3).squeeze(3)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max2(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max3(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.depth, 1, 1)      
        return x

####################################################################################
#
#
# EVERYTHING UNDER HERE IS OLD / USED FOR EXPERIMENTING
#

class Regressor(nn.Module):
    def __init__(self, n_channels, features):
        super().__init__()
        self.inp_channels = n_channels
        self.inp_features = features
        self.conv1 = nn.Conv1d(
            in_channels = n_channels,
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )
        self.fc1 = nn.Linear(7168, 4096)
        self.fc2 = nn.Linear(4096, 2802)
        
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self, x):
        #print(f"\tIn model input size {x.shape} on device {x.get_device()}")
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        #print(x.shape)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        #print(x.shape)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        #print(x.shape)
        
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.dropout(x)
        
        x = self.fc2(x)
        #print(x.shape)
        x = x.view(-1, self.inp_channels, self.inp_features)
        #print('size', x.shape)       
        return x
    
class RegressorBN(nn.Module):
    def __init__(self, n_channels, features):
        super().__init__()
        self.inp_channels = n_channels
        self.inp_features = features
        self.conv1 = nn.Conv1d(
            in_channels = n_channels,
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )
        self.fc1 = nn.Linear(7168, 4096)
        self.fc2 = nn.Linear(4096, 2802)
        
        self.bn1 = nn.BatchNorm1d(32)
        self.bn2 = nn.BatchNorm1d(64)
        
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self, x):
        #print(f"\tIn model input size {x.shape} on device {x.get_device()}")
        #print(x.shape)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        #print(x.shape)
        
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        #print(x.shape)
        
        x = self.bn2(x)
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        #print(x.shape)
        
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.dropout(x)
        
        x = self.fc2(x)
        #print(x.shape)
        x = x.view(-1, self.inp_channels, self.inp_features)
        #print('size', x.shape)       
        return x
    

class DeeperRegressor(nn.Module):
    def __init__(self, n_channels, features):
        super().__init__()
        self.inp_channels = n_channels
        self.inp_features = features
        self.conv1 = nn.Conv1d(
            in_channels = n_channels,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.conv2 = nn.Conv1d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.conv3 = nn.Conv1d(
            in_channels = 64,
            out_channels = 64,
            kernel_size = 3,
            padding = 1
        )
        self.conv4 = nn.Conv1d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.conv5 = nn.Conv1d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        self.conv6 = nn.Conv1d(
            in_channels = 128,
            out_channels = 128,
            kernel_size = 3,
            padding = 1
        )
        
        self.bn1 = nn.BatchNorm1d(64)
        self.bn2 = nn.BatchNorm1d(64)
        self.bn3 = nn.BatchNorm1d(64)
        self.bn4 = nn.BatchNorm1d(128)
        self.bn5 = nn.BatchNorm1d(128)
        
        self.fc1 = nn.Linear(7424, 4096)
        self.fc2 = nn.Linear(4096, 2802)
        
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self, x):

        ##### block 64 #####
        x = self.conv1(x)
        x = F.relu(x)
        
        x = self.bn1(x)
        x = self.conv2(x)
        x = F.relu(x)
        
        x = self.bn2(x)
        x = self.conv3(x)
        x = F.relu(x)        
        
        ##### block 128 #####
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.bn3(x)
        x = self.conv4(x)
        x = F.relu(x)
        
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.bn4(x)
        x = self.conv5(x)
        x = F.relu(x)
        
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        x = self.bn5(x)
        x = self.conv6(x)
        x = F.relu(x)
        
        ##### block linear #####
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)      
        x = self.fc2(x)

        x = x.view(-1, self.inp_channels, self.inp_features)    
        return x

class BasicRegressor3D_3x3(nn.Module):
    def __init__(self, channels, depth, height, width):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.height = height
        self.width = width
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (3,3,3),
            padding = (1,1,1)
        )
        self.conv2 = nn.Conv3d(
            in_channels = 32, 
            out_channels = 64,
            kernel_size = (3,3,3),
            padding = (1,1,1)
        )
        self.conv3 = nn.Conv3d(
            in_channels = 64, 
            out_channels = 128,
            kernel_size = (3,3,3),
            padding = (1,1,1)
        )
        self.conv4 = nn.Conv3d(
            in_channels = 128, 
            out_channels = 128,
            kernel_size = (1,3,3),
            padding = (0,0,0)
        )

        self.max1 = nn.MaxPool3d(kernel_size=(2,1,1), stride = (2,1,1))
        self.max2 = nn.MaxPool3d(kernel_size=(2,1,1), stride = (2,1,1))
        self.max3 = nn.MaxPool3d(kernel_size=(2,1,1), stride = (2,1,1))

        self.fc1 = nn.Linear(7296, 4096)
        self.fc2 = nn.Linear(4096, 2760)

        self.dropout = nn.Dropout2d(p=0.5)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max1(x)
        #print(x.shape)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max2(x)
        #print(x.shape)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max3(x)
        #print(x.shape)
        
        x = self.conv4(x)
        x = F.relu(x)
        
        x = torch.flatten(x, 1)
        #print(x.shape)
        x = self.fc1(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.dropout(x)
        
        x = self.fc2(x)
        #print(x.shape)
        x = x.view(-1, self.channels, self.depth, 1, 1)
        #print('size', x.shape)       
        return x



class BasicRegressor3D_5x5(nn.Module):
    def __init__(self, channels, depth, height, width):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.height = height
        self.width = width
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (3,5,5),
            padding = (1,2,2)
        )
        self.conv2 = nn.Conv3d(
            in_channels = 32, 
            out_channels = 64,
            kernel_size = (3,5,5),
            padding = (1,2,2)
        )
        self.conv3 = nn.Conv3d(
            in_channels = 64, 
            out_channels = 128,
            kernel_size = (3,5,5),
            padding = (1,2,2)
        )
        self.conv4 = nn.Conv3d(
            in_channels = 128, 
            out_channels = 128,
            kernel_size = (1,5,5),
            padding = (0,0,0)
        )

        self.max1 = nn.MaxPool3d(kernel_size=(2,1,1), stride = (2,1,1))
        self.max2 = nn.MaxPool3d(kernel_size=(2,1,1), stride = (2,1,1))
        self.max3 = nn.MaxPool3d(kernel_size=(2,1,1), stride = (2,1,1))

        self.fc1 = nn.Linear(7296, 4096)
        self.fc2 = nn.Linear(4096, 2760)

        self.dropout = nn.Dropout2d(p=0.5)
    
    def forward(self, x):
        
        x = self.conv1(x)
        x = F.relu(x)
        x = self.max1(x)
        #print(x.shape)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max2(x)
        #print(x.shape)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max3(x)
        #print(f"Shape after last max pool {x.shape}")
        
        x = self.conv4(x)
        x = F.relu(x)
        
        #print(f"Shape before flatten {x.shape}")
        x = torch.flatten(x, 1)
        #print(f"Shape after flatten{x.shape}")
        x = self.fc1(x)
        x = F.relu(x)
        #print(x.shape)
        x = self.dropout(x)
        
        x = self.fc2(x)
        #print(x.shape)
        x = x.view(-1, self.channels, self.depth, 1, 1)
        #print('size', x.shape)       
        return x

####################################################################################################

class Trans_1x1(nn.Module):
    def __init__(self, n_channels, features, height, width):
        super().__init__()
        self.inp_channels = n_channels
        self.inp_features = features
        self.conv1 = nn.Conv1d(
            in_channels = n_channels,
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )
        self.fc1 = nn.Linear(7040, 4096)
        self.fc2 = nn.Linear(4096, n_channels*features)
        
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self, x):
        
        x = x.squeeze(3).squeeze(3)
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.inp_channels, self.inp_features, 1, 1)      
        return x
    
class Trans_3x3(nn.Module):
    def __init__(self, channels, depth, height, width):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.height = height
        self.width = width
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (3,3,3),
            padding = (1,0,0)
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32, 
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 32, 
            out_channels =64,
            kernel_size = 3,
            padding = 0
        )
        self.conv4 = nn.Conv1d(
            in_channels = 64, 
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )

        self.max1 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max2 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max3 = nn.MaxPool1d(kernel_size=2, stride = 2)

        self.fc1 = nn.Linear(7040, 4096)
        self.fc2 = nn.Linear(4096, channels*depth)

        self.dropout = nn.Dropout2d(p=0.5)
    
    def forward(self, x):
        
        x = self.conv1(x).squeeze(3).squeeze(3)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max2(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max3(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.depth, 1, 1)      
        return x
    

    
class Trans_5x5(nn.Module):
    def __init__(self, channels, depth, height, width):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.height = height
        self.width = width
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (3,5,5),
            padding = (1,0,0)
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32, 
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 32, 
            out_channels =64,
            kernel_size = 3,
            padding = 0
        )
        self.conv4 = nn.Conv1d(
            in_channels = 64, 
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )

        self.max1 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max2 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max3 = nn.MaxPool1d(kernel_size=2, stride = 2)

        self.fc1 = nn.Linear(7040, 4096)
        self.fc2 = nn.Linear(4096, channels*depth)

        self.dropout = nn.Dropout2d(p=0.5)
    
    def forward(self, x):
        
        x = self.conv1(x).squeeze(3).squeeze(3)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max2(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max3(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.depth, 1, 1)      
        return x

class Trans_7x7(nn.Module):
    def __init__(self, channels, depth, height, width):
        super().__init__()
        self.channels = channels
        self.depth = depth
        self.height = height
        self.width = width
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (3,7,7),
            padding = (1,0,0)
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32, 
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 32, 
            out_channels =64,
            kernel_size = 3,
            padding = 0
        )
        self.conv4 = nn.Conv1d(
            in_channels = 64, 
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )

        self.max1 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max2 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max3 = nn.MaxPool1d(kernel_size=2, stride = 2)

        self.fc1 = nn.Linear(7040, 4096)
        self.fc2 = nn.Linear(4096, channels*depth)

        self.dropout = nn.Dropout2d(p=0.5)
    
    def forward(self, x):
        
        x = self.conv1(x).squeeze(3).squeeze(3)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = self.max1(x)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = self.max2(x)
        
        x = self.conv4(x)
        x = F.relu(x)
        x = self.max3(x)
        x = torch.flatten(x, 1)
        
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.depth, 1, 1)      
        return x
    
#################################### Deep dimension change ##################################################################################
class BasicBlock(nn.Module):
    # 2 conv operations
    def __init__(self, in_channels, out_channels, stride, padding):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = stride,
            padding = padding
        )
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1
        )
        
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.bn2 = nn.BatchNorm1d(out_channels)
    
    def forward(self, x):
        #first section
        x = F.relu(self.bn1(self.conv1(x)))
        #seconf section
        x = F.relu(self.bn2(self.conv2(x)))
        return x
    
class Trans_3x3_Deep(nn.Module):
    def __init__(self, channels, depth, layers):
        super().__init__()
        
        self.channels = channels
        self.depth = depth
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (3,3,3),
            padding = (1,0,0)
        )
        self.bn1 = nn.BatchNorm1d(32)

        self.layer1 = self._make_conv_layer(32, 32, layers[0], padding = 0, stride=1)
        self.layer2 = self._make_conv_layer(32, 64, layers[1], padding = 0, stride=1)
        self.layer3 = self._make_conv_layer(64, 128, layers[2], padding = 0, stride=1)

        self.max1 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max2 = nn.MaxPool1d(kernel_size=2, stride = 2)
        self.max3 = nn.MaxPool1d(kernel_size=2, stride = 2)

        self.fc1 = nn.Linear(7040, 4096)
        self.fc2 = nn.Linear(4096, channels*depth)
        
        self.dropout = nn.Dropout2d(p=0.5)
    
    def _make_conv_layer(self, in_channels, out_channels, layers, stride, padding):
        layer = nn.Sequential()
        for i in range(layers):
            layer_name = f'layer_{i+1}'
            if i == 0:
                layer.add_module(layer_name, BasicBlock(in_channels, out_channels, stride=stride, padding=padding))
            else:
                layer.add_module(layer_name, BasicBlock(out_channels, out_channels, stride = 1, padding = 1))
        return layer

    def forward(self, x):
        
        x = self.conv1(x).squeeze(3).squeeze(3)
        x = F.relu(self.bn1(x))
        
        x = self.layer1(x)
        x = self.max1(x)
        
        x = self.layer2(x)
        x = self.max2(x)
        
        x = self.layer3(x)
        x = self.max3(x)
        
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.depth,1,1) 

        return x

################################################## resnet ######################################################################
    
class BasicBlock_Res(nn.Module):
    # 2 conv operations
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = stride, # use stride = 2 to downsize by half ie what maxpooling did
            padding = 1,
            bias = False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        if in_channels != out_channels: #need to downsample to match new size
            self.skip = nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size = 1, 
                    stride = stride, 
                    bias=False
                ),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Sequential()
    
    def forward(self, x):
        #first section
        out = F.relu(self.bn1(self.conv1(x)))
        #second section
        out = F.relu(self.bn2(self.conv2(out)))
        #add skip
        out += self.skip(x)
        out = F.relu(out)
        return out

class BasicBlock_NoBN(nn.Module):
    # 2 conv operations
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size = 3,
            stride = stride, # use stride = 2 to downsize by half ie what maxpooling did
            padding = 1,
            bias = False
        )
        
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size = 3,
            stride = 1,
            padding = 1,
            bias = False
        )
        
        if in_channels != out_channels: #need to downsample to match new size
            self.skip = nn.Sequential(
                nn.Conv1d(
                    in_channels, 
                    out_channels, 
                    kernel_size = 1, 
                    stride = stride, 
                    bias=False
                )
            )
        else:
            self.skip = nn.Sequential()
    
    def forward(self, x):
        #first section
        out = F.relu(self.conv1(x))
        #second section
        out = F.relu(self.conv2(out))
        #add skip
        out += self.skip(x)
        out = F.relu(out)
        return out
    
class Trans_3x3_ResNet(nn.Module):
    def __init__(self, channels, depth, layers):
        super().__init__()
        print('This is ResNet')
        self.channels = channels
        self.depth = depth
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (3,3,3),
            padding = (1,0,0)
        )
        self.bn1 = nn.BatchNorm1d(32)
        
        self.layer1 = self._make_conv_layer(32, 32, layers[0], stride = 1)
        self.layer2 = self._make_conv_layer(32, 64, layers[1], stride = 2)
        self.layer3 = self._make_conv_layer(64, 128, layers[2], stride = 2)
        
        self.downsample = nn.AdaptiveAvgPool1d(55)

        self.fc1 = nn.Linear(7040, 4096)
        self.fc2 = nn.Linear(4096, channels*depth)
        
        self.dropout = nn.Dropout2d(p=0.5)
    
    def _make_conv_layer(self, in_channels, out_channels, layers, stride):
        layer = nn.Sequential()
        for i in range(layers):
            layer_name = f'layer_{i+1}'
            if i == 0:
                layer.add_module(layer_name, BasicBlock_Res(in_channels, out_channels, stride=stride))
            else:
                layer.add_module(layer_name, BasicBlock_Res(out_channels, out_channels, stride = 1))
        return layer

    def forward(self, x):
        
        x = self.conv1(x).squeeze(3).squeeze(3)
        x = F.relu(self.bn1(x)) #460

        x = self.layer1(x) #460


        x = self.layer2(x) #230

        
        x = self.layer3(x) #115

        x = self.downsample(x)

        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.depth,1,1) 

        return x

class Trans_3x3_ResNet_NoBN(nn.Module):
    def __init__(self, channels, depth, layers):
        super().__init__()
        print('This is ResNet no norm')
        self.channels = channels
        self.depth = depth
        
        self.conv1 = nn.Conv3d(
            in_channels = 6, 
            out_channels = 32,
            kernel_size = (3,3,3),
            padding = (1,0,0)
        )
        
        self.layer1 = self._make_conv_layer(32, 32, layers[0], stride = 1)
        self.layer2 = self._make_conv_layer(32, 64, layers[1], stride = 2)
        self.layer3 = self._make_conv_layer(64, 128, layers[2], stride = 2)
        
        self.downsample = nn.AdaptiveAvgPool1d(55)

        self.fc1 = nn.Linear(7040, 4096)
        self.fc2 = nn.Linear(4096, channels*depth)
        
        self.dropout = nn.Dropout2d(p=0.5)
    
    def _make_conv_layer(self, in_channels, out_channels, layers, stride):
        layer = nn.Sequential()
        for i in range(layers):
            layer_name = f'layer_{i+1}'
            if i == 0:
                layer.add_module(layer_name, BasicBlock_NoBN(in_channels, out_channels, stride=stride))
            else:
                layer.add_module(layer_name, BasicBlock_NoBN(out_channels, out_channels, stride = 1))
        return layer

    def forward(self, x):
        
        x = self.conv1(x).squeeze(3).squeeze(3) #460

        x = self.layer1(x) #460


        x = self.layer2(x) #230

        
        x = self.layer3(x) #115

        x = self.downsample(x)

        x = torch.flatten(x, 1)
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.channels, self.depth,1,1) 

        return x
    
######################################################## STACK MODEL 2nd part ##########################################################
    
class Trans_Regressor(nn.Module):
    def __init__(self, n_channels, features):
        super().__init__()
        self.inp_channels = n_channels
        self.inp_features = features
        self.conv1 = nn.Conv1d(
            in_channels = n_channels,
            out_channels = 32,
            kernel_size = 3,
            padding = 0
        )
        self.conv2 = nn.Conv1d(
            in_channels = 32,
            out_channels = 64,
            kernel_size = 3,
            padding = 0
        )
        self.conv3 = nn.Conv1d(
            in_channels = 64,
            out_channels = 128,
            kernel_size = 3,
            padding = 0
        )
        self.fc1 = nn.Linear(7040, 4096)
        self.fc2 = nn.Linear(4096, n_channels*features)
        
        self.dropout = nn.Dropout2d(p=0.5)
        
    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = self.conv3(x)
        x = F.relu(x)
        x = F.max_pool1d(x, kernel_size=2, stride=2)
        
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        
        x = self.fc2(x)
        x = x.view(-1, self.inp_channels, self.inp_features)
        return x