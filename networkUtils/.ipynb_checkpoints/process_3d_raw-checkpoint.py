from helita.sim import multi3d
import numpy as np
import argparse
import h5py
import os
'''
create datasets with scaling factors from the 7x7 windows made in extract_3d_7x7.py

OLD
'''

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--f_path', type = str, required = True, help = "File path to segmented data")
parser.add_argument('-s', '--save_path', type = str, required = True, help = "Save folder path")
parser.add_argument('-n', '--name', type = str, required = True, help = "Name of hdf5 file to save to, include extension")
parser.add_argument('-w', '--window', type = int, required = True, help = "Window size")

args = parser.parse_args()

labels = ['lte training windows', 'non lte training points', 'lte test windows', 'non lte test points']

if args.window == 1:
    print(f'Reading 7x7 and changing to 1x1...')
    with h5py.File(args.f_path, 'r') as f:
        
        lte_train = f[labels[0]][..., 3, 3][...,np.newaxis,np.newaxis]
        train_len = f[labels[0]].attrs["len"]
        x = f[labels[0]].attrs["x"]
        y = f[labels[0]].attrs["y"]
        z = f[labels[0]].attrs["z"]
        
        non_lte_train = f[labels[1]][:]
        test_len = f[labels[1]].attrs["len"]
        
        lte_test = f[labels[2]][..., 3, 3][...,np.newaxis,np.newaxis]
        non_lte_test = f[labels[3]][:]
    
    print("Scaling data...")
    #scale sets
    mu_inp = lte_train.mean(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]
    std_inp = lte_train.std(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]

    mu_out = non_lte_train.mean(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]
    std_out = non_lte_train.std(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]

    # scale training and test data
    lte_train = (lte_train - mu_inp)/std_inp
    non_lte_train = (non_lte_train - mu_out)/std_out

    lte_test = (lte_test - mu_inp)/std_inp
    non_lte_test = (non_lte_test - mu_out)/std_out

elif args.window == 3:
    print(f'Reading 7x7 and changing to 3x3...')
    with h5py.File(args.f_path, 'r') as f:
        
        lte_train = f[labels[0]][..., 2:5, 2:5]
        train_len = f[labels[0]].attrs["len"]
        x = f[labels[0]].attrs["x"]
        y = f[labels[0]].attrs["y"]
        z = f[labels[0]].attrs["z"]
        
        non_lte_train = f[labels[1]][:]
        test_len = f[labels[1]].attrs["len"]
        
        lte_test = f[labels[2]][..., 2:5, 2:5]
        non_lte_test = f[labels[3]][:]
    
    print("Scaling data...")
    #scale sets
    mu_inp = lte_train.mean(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]
    std_inp = lte_train.std(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]

    mu_out = non_lte_train.mean(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]
    std_out = non_lte_train.std(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]

    # scale training and test data
    lte_train = (lte_train - mu_inp)/std_inp
    non_lte_train = (non_lte_train - mu_out)/std_out

    lte_test = (lte_test - mu_inp)/std_inp
    non_lte_test = (non_lte_test - mu_out)/std_out  

elif args.window == 5:
    print(f'Reading 7x7 and changing to 5x5...')
    with h5py.File(args.f_path, 'r') as f:
        
        lte_train = f[labels[0]][..., 1:6, 1:6]
        train_len = f[labels[0]].attrs["len"]
        x = f[labels[0]].attrs["x"]
        y = f[labels[0]].attrs["y"]
        z = f[labels[0]].attrs["z"]
        
        non_lte_train = f[labels[1]][:]
        test_len = f[labels[1]].attrs["len"]
        
        lte_test = f[labels[2]][..., 1:6, 1:6]
        non_lte_test = f[labels[3]][:]
    
    print("Scaling data...")
    #scale sets
    mu_inp = lte_train.mean(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]
    std_inp = lte_train.std(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]

    mu_out = non_lte_train.mean(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]
    std_out = non_lte_train.std(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]

    # scale training and test data
    lte_train = (lte_train - mu_inp)/std_inp
    non_lte_train = (non_lte_train - mu_out)/std_out

    lte_test = (lte_test - mu_inp)/std_inp
    non_lte_test = (non_lte_test - mu_out)/std_out
    
elif args.window == 7:
    print(f'Reading 7x7 and remaining at 7x7...')
    with h5py.File(args.f_path, 'r') as f:
        lte_train = f[labels[0]][:]
        train_len = f[labels[0]].attrs["len"]
        x = f[labels[0]].attrs["x"]
        y = f[labels[0]].attrs["y"]
        z = f[labels[0]].attrs["z"]
        
        non_lte_train = f[labels[1]][:]
        test_len = f[labels[1]].attrs["len"]
        
        lte_test = f[labels[2]][:]
        non_lte_test = f[labels[3]][:]
    
    print("Scaling data...")
    #scale sets
    mu_inp = lte_train.mean(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]
    std_inp = lte_train.std(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]

    mu_out = non_lte_train.mean(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]
    std_out = non_lte_train.std(axis=(0,2,3,4))[:, np.newaxis, np.newaxis, np.newaxis]

    # scale training and test data
    lte_train = (lte_train - mu_inp)/std_inp
    non_lte_train = (non_lte_train - mu_out)/std_out

    lte_test = (lte_test - mu_inp)/std_inp
    non_lte_test = (non_lte_test - mu_out)/std_out

print('Saving....')
with h5py.File(os.path.join(args.save_path, args.name), 'w') as f:
    dset1 = f.create_dataset("lte training windows", data=lte_train, dtype='f')
    dset1.attrs["mu"] = mu_inp
    dset1.attrs["std"] = std_inp
    dset1.attrs["len"] = train_len
    dset1.attrs["x"] = x
    dset1.attrs["y"] = y
    dset1.attrs["z"] = z

    dset2 = f.create_dataset("non lte training points", data=non_lte_train, dtype='f')
    dset2.attrs["mu"] = mu_out
    dset2.attrs["std"] = std_out

    dset3 = f.create_dataset("lte test windows", data=lte_test, dtype='f')
    dset3.attrs["mu"] = mu_inp
    dset3.attrs["std"] = std_inp
    dset3.attrs["len"] = test_len
    dset3.attrs["x"] = x
    dset3.attrs["y"] = y
    dset3.attrs["z"] = z

    dset4 = f.create_dataset("non lte test points", data=non_lte_test, dtype='f')
    dset4.attrs["mu"] = mu_out
    dset4.attrs["std"] = std_out

