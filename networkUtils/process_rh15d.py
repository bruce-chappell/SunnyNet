from helita.sim import rh15d
import numpy as np
import argparse
import h5py
import os

'''
Create 1.5D datasets

OLD
'''

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--f_path', type = str, required = True, help = "File path to simulation")
parser.add_argument('-s', '--save_path', type = str, required = True, help = "Save folder path")
parser.add_argument('-t', '--train', type = int, required = True, help = "Percentage of data to be training data")
parser.add_argument('-x', '--test', type = int, required = True, help = "Percentage of data to be training data")
parser.add_argument('-n', '--name', type = str, required = True, help = "name of hdf5 file to save to, include extension")

args = parser.parse_args()

assert (args.train + args.test) == 100, "Percentages for train/test sets need to add up to 100"

metadata = rh15d.Rh15dout(fdir=args.f_path)

print(f'Performing log10 calculation....' )
v1 = np.log10(metadata.atom_H.populations_LTE.values)
v2 = np.log10(metadata.atom_H.populations.values)

dims = v1.shape

height_scale = metadata.atmos.height_scale[0,0].values/1e6

# reshape to data_pts X channels X inp_shape
np_lte = np.transpose(v1,(1,2,0,3)).reshape(dims[1]*dims[2], dims[0], dims[3]) 
np_non_lte = np.transpose(v2,(1,2,0,3)).reshape(dims[1]*dims[2], dims[0], dims[3])


print('splitting data....')
# get training / testing indexes
full_idx = np.arange(len(np_lte))
t_v = int(len(np_lte) * args.train/100)
t_v_idx = np.random.choice(full_idx, size = t_v, replace=False)
test_idx = np.setxor1d(t_v_idx, full_idx)

# split sets
lte_train = np_lte[t_v_idx]
non_lte_train = np_non_lte[t_v_idx]

lte_test = np_lte[test_idx]
non_lte_test = np_non_lte[test_idx]

print('scaling data....')
# calculate scales from train data
mu_inp = lte_train.mean(axis=(0,2))[:, np.newaxis]
std_inp = lte_train.std(axis=(0,2))[:, np.newaxis]

mu_out = non_lte_train.mean(axis=(0,2))[:, np.newaxis]
std_out = non_lte_train.std(axis=(0,2))[:, np.newaxis]

# scale training and test data
lte_train = (lte_train - mu_inp)/std_inp
non_lte_train = (non_lte_train - mu_out)/std_out

lte_test = (lte_test - mu_inp)/std_inp
non_lte_test = (non_lte_test - mu_out)/std_out

print('Saving....')

with h5py.File(os.path.join(args.save_path, args.name), 'w') as f:
    dset1 = f.create_dataset("lte training points", data=lte_train, dtype='f')
    dset1.attrs["mu"] = mu_inp
    dset1.attrs["std"] = std_inp
    dset1.attrs["len"] = len(t_v_idx)
    dset1.attrs["height"] = height_scale
    
    dset2 = f.create_dataset("non lte training points", data=non_lte_train, dtype='f')
    dset2.attrs["mu"] = mu_out
    dset2.attrs["std"] = std_out
    
    dset3 = f.create_dataset("lte test points", data=lte_test, dtype='f')
    dset3.attrs["mu"] = mu_inp
    dset3.attrs["std"] = std_inp
    dset3.attrs["len"] = len(test_idx)
    dset3.attrs["height"] = height_scale 
    
    dset4 = f.create_dataset("non lte test points", data=non_lte_test, dtype='f')
    dset4.attrs["mu"] = mu_out
    dset4.attrs["std"] = std_out