import os
import numpy as np
import h5py
from helita.sim import multi3d
from helita.sim.multi3d import Multi3dAtmos
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.ndimage import zoom

'''
This method lets us build training set with different window sizes, controlled by the pad variable.

For example 1x1, 3x3, 5x5, 7x7
'''

#################### USER INPUTS ################

save_path = ''

# corresponding multi3d atmos paths
atmos1 = ['', , , ] # [path, atms_x, atms_y, atms_z]
#atmos2 =
#atmos3 =
#atmos4 =

# name sims we want to sample from
path1 = ''
#path2 =
#path3 =
#path4 =

#path_holder = [path1, path2, path3, path4]
#path_holder = [path3, path4]
path_holder = [path1]

#atmos_holder = [atmos1, atmos2, atmos3, atmos4]
#atmos_holder = [atmos3, atmos4]
atmos_holder = [atmos1]

pad = 3;            # how big you want lte window to be, 0 = 1x1, 1 = 3x3, 2 = 5x5 ....
dim = 252;          # original X/Y dim of simulation
tr_percent = 85     # percent of snapshot to be used a training set, rest is validation set

# accounts for expanding to include periodic BC's
grid = dim + 2*pad;

#################### USER INPUTS ################




# check save path validity
folder = '/'.join(save_path.split('/')[:-1])
assert os.path.exists(folder) == True, "Make sure save folder exists"
#print(folder)
#print(save_path)
assert os.path.exists(save_path) == False ,"dont overwrite old results"


lte_list = []
non_lte_list = []

for s, a in zip(path_holder, atmos_holder):

    print(f'Working on sim {a[0]}...')
    # sim to be interpolated
    os.chdir(s)
    sim = multi3d.Multi3dOut(directory='.')
    sim.readall()

    # read atmos vars
    atmos = Multi3dAtmos(a[0], a[1], a[2], a[3])

    # switch to column mass and interpolate
    rho = atmos.rho * 1e3                    #  g cm-3 to kg m-3
    rho_mean = np.mean(rho, axis=(0,1))

    z = sim.geometry.z * 1e-2                # cm to m
    print(f'rho shape {rho_mean.shape}')
    print(f'z shape {z.shape}')
    cmass_mean = cumtrapz(rho_mean, -z, initial=0)
    cmass_scale = np.logspace(-6,2,400)

    print('Interpolating functions...')
    f_lte = interp1d(cmass_mean, (sim.atom.nstar * 1e6), kind='linear', axis = 2, fill_value = 'extrapolate')
    f_non_lte = interp1d(cmass_mean, (sim.atom.n * 1e6), kind='linear', axis = 2, fill_value = 'extrapolate')

    print('Applying new scale...')
    lte = f_lte(cmass_scale)
    non_lte = f_non_lte(cmass_scale)

    print('Padding data...')
    npad = ((pad,pad),(pad,pad),(0,0),(0,0))
    lte = np.pad(lte, pad_width=npad, mode='wrap')
    non_lte = np.pad(non_lte, pad_width=npad, mode='wrap')

    print('Log and transpose...')
    lte = np.transpose(np.log10(lte), (3,2,0,1))
    non_lte = np.transpose(np.log10(non_lte), (3,2,0,1))

    print('Scaling data...')
    mu_inp = lte.mean(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis]
    std_inp = lte.std(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis]

    mu_out = non_lte.mean(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis]
    std_out = non_lte.std(axis=(1,2,3))[:, np.newaxis, np.newaxis, np.newaxis]

    lte = (lte - mu_inp)/std_inp
    non_lte = (non_lte - mu_out)/std_out

    print(f"Splitting simulation into corresponding window / columns...")
    for i in range(pad, grid-pad):
        for j in range(pad, grid-pad):
            #lte window
            sample = lte[:, :, i-pad:i+(pad+1), j-pad:j+(pad+1)]
            lte_list.append(sample)
            #non lte in middle of window
            true = non_lte[:,:,i,j][:,:,np.newaxis,np.newaxis]
            non_lte_list.append(true)

print(f"Train / Test Split...")
#get train/test indicies
full_idx = np.arange(len(lte_list))
tr = int(len(lte_list) * tr_percent/100)
tr_idx = np.random.choice(full_idx, size = tr, replace=False)
val_idx = np.setxor1d(tr_idx, full_idx)

lte = np.array(lte_list)
non_lte = np.array(non_lte_list)

# split sets
lte_train = lte[tr_idx]
non_lte_train = non_lte[tr_idx]
train_len = len(lte_train)

lte_test = lte[val_idx]
non_lte_test = non_lte[val_idx]
test_len = len(lte_test)

print(f'Input shape {lte_train.shape}')
print(f'Output shape {non_lte_train.shape}')

print('Saving...')
with h5py.File(save_path, 'w') as f:
    dset1 = f.create_dataset("lte training windows", data=lte_train, dtype='f')
    dset1.attrs["mu"] = mu_inp
    dset1.attrs["std"] = std_inp
    dset1.attrs["len"] = train_len
    dset1.attrs["z"] = z

    dset2 = f.create_dataset("non lte training points", data=non_lte_train, dtype='f')
    dset2.attrs["mu"] = mu_out
    dset2.attrs["std"] = std_out

    dset3 = f.create_dataset("lte test windows", data=lte_test, dtype='f')
    dset3.attrs["mu"] = mu_inp
    dset3.attrs["std"] = std_inp
    dset3.attrs["len"] = test_len
    dset3.attrs["z"] = z

    dset4 = f.create_dataset("non lte test points", data=non_lte_test, dtype='f')
    dset4.attrs["mu"] = mu_out
    dset4.attrs["std"] = std_out
