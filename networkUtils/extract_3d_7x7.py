from helita.sim import multi3d
import numpy as np
import argparse
import h5py
import os
'''
Use to pull out 7x7 windows from multi3d simulations. These windows will be used as training/test data

OLD
'''

parser = argparse.ArgumentParser()

parser.add_argument('-f', '--f_path', type = str, required = True, help = "File path to simulation")
parser.add_argument('-s', '--save_path', type = str, required = True, help = "Save folder path")
parser.add_argument('-t', '--train', type = int, required = True, help = "Percentage of data to be training data")
parser.add_argument('-x', '--test', type = int, required = True, help = "Percentage of data to be test data")
parser.add_argument('-n', '--name', type = str, required = True, help = "Name of hdf5 file to save to, include extension")

args = parser.parse_args()

assert (args.train + args.test) == 100, "Percentages for train/test sets need to add up to 100"

os.chdir(args.f_path)
data = multi3d.Multi3dOut()
data.readall()

print(f'Performing log10 calculation....' )
non_lte = np.log10(data.atom.n*1e6)
lte = np.log10(data.atom.nstar*1e6)

non_lte = np.transpose(non_lte, (3,2,0,1))
lte = np.transpose(lte, (3,2,0,1))

print(f"Reading geometries...")
# cm to Mm
x = data.geometry.x*1e-8
y = data.geometry.y*1e-8
z = data.geometry.z*1e-8

dims = non_lte.shape

print(f"Splitting simulation into training examples...")
pad = 7 // 2
grid = dims[2]
lte_list = []
non_lte_list = []
for i in range(pad, grid-pad):
    for j in range(pad, grid-pad):
        #lte window
        sample = lte[:, :, i-pad:i+(pad+1), j-pad:j+(pad+1)]
        lte_list.append(sample)
        #non lte in middle of window
        true = non_lte[:,:,i,j][:,:,np.newaxis,np.newaxis]
        non_lte_list.append(true)
        
print(f"Train Test Split...")
#get train/test indicies 
full_idx = np.arange(len(lte_list))
t = int(len(lte_list) * args.train/100)
t_idx = np.random.choice(full_idx, size = t, replace=False)
test_idx = np.setxor1d(t_idx, full_idx)

# split sets
lte_train = np.array([lte_list[i] for i in t_idx])
non_lte_train = np.array([non_lte_list[i] for i in t_idx])

lte_test = np.array([lte_list[i] for i in test_idx])
non_lte_test = np.array([non_lte_list[i] for i in test_idx])

print('Saving....')

with h5py.File(os.path.join(args.save_path, args.name), 'w') as f:
    dset1 = f.create_dataset("lte training windows", data=lte_train, dtype='f')
    dset1.attrs["len"] = len(t_idx)
    dset1.attrs["x"] = x
    dset1.attrs["y"] = y
    dset1.attrs["z"] = z
    
    dset2 = f.create_dataset("non lte training points", data=non_lte_train, dtype='f')
    dset2.attrs["len"] = len(test_idx)
    
    dset3 = f.create_dataset("lte test windows", data=lte_test, dtype='f')
    dset3.attrs["len"] = len(t_idx)
    dset3.attrs["x"] = x
    dset3.attrs["y"] = y
    dset3.attrs["z"] = z
    
    dset4 = f.create_dataset("non lte test points", data=non_lte_test, dtype='f')
    dset4.attrs["len"] = len(test_idx)