from helita.sim import multi3d
import numpy as np
import h5py
import os
from helita.sim.multi3d import Multi3dAtmos
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.ndimage import zoom

'''
takes original atmosphere cube and interpolates to (c, 400, x, y) then
creates the (c, 400, 1, 1) / (c, 400, 3, 3) / (c, 400, 5, 5) / (c, 400, 7, 7) lte windows and (c, 400, 1, 1) non lte columns
to be fed into a trained network to make population predictions
'''

##################### USER INPUTS #######################################
# ---------- simulation -----------
save_path = f'example.hdf5'
sim_path = f'' # Multi3dOut sim path
atmos_path = f'' # Multi3dAtmos path
dim_z = #int
dim = #int

pad = 3 # how big you want lte window to be, 0 = 1x1, 1 = 3x3, 2 = 5x5 ....



# need this if sim is huge and we need to just analyize a section of it ex 252 x252 window of something 1000 x1000
# ----> turn on with lim_grid = True
lim_grid = False
st_x = 0 # where to start when pulling out training/test windows.
st_y = 0 # where to start when pulling out training/test windows.


##################### END USER INPUTS #######################################

atmos_idx = 0    # atmos.hydrogen_populations has size (sim_num, levels, x, y, z)

grid = dim + 2*pad # account for padding periodic BC's


print(f'Handling {sim_name} {sim_num_atms}')

# sim to be interpolated
os.chdir(sim_path)
sim = multi3d.Multi3dOut(directory='.')
sim.readall()

# read atmos vars
atmos = Multi3dAtmos(atmos_path, dim, dim, dim_z)

# check save path validity
folder = '/'.join(save_path.split('/')[:-1])
assert os.path.exists(folder) == True, "Make sure save folder exists"
assert os.path.exists(save_path) == False ,"dont overwrite old results"

# check sim dims and see whats gonna happen with interpolation
x_sh = sim.geometry.x.shape[0]
y_sh = sim.geometry.y.shape[0]
z_sh = sim.geometry.z.shape[0]
print(f'Sim shape: ({x_sh}, {y_sh}, {z_sh})')

if not (y_sh == x_sh):
    raise ValueError("Resizing function needs X / Y to be equal in length")

print('Padding for periodic BC...')
lte = sim.atom.nstar
non_lte = sim.atom.n
npad = ((pad,pad),(pad,pad),(0,0),(0,0))
lte = np.pad(lte, pad_width=npad, mode='wrap')
non_lte = np.pad(non_lte, pad_width=npad, mode='wrap')
print(f'LTE shape after padding: {lte.shape}')

print('Starting Z interpolation...')
############################################################
if lim_grid:
    rho = atmos.rho[st_x:st_x + grid-2, st_y:st_y + grid-2, ...] * 1e3                    #  g cm-3 to kg m-3
else:
    rho = atmos.rho * 1e3
rho_mean = np.mean(rho, axis=(0,1))
z = sim.geometry.z * 1e-2
cmass_mean = cumtrapz(rho_mean, -z, initial=0)
cmass_scale = np.logspace(-6,2,400)

print('Interpolating functions...')
############################################################
if lim_grid:
    tmp = lte[st_x:st_x + grid, st_y:st_y + grid,...] * 1e6
    tmp2 = non_lte[st_x:st_x + grid, st_y:st_y + grid,...] * 1e6
    f_lte = interp1d(cmass_mean, tmp, kind='linear', axis = 2, fill_value = 'extrapolate')
    f_non_lte = interp1d(cmass_mean, tmp2, kind='linear', axis = 2, fill_value = 'extrapolate')
else:
    f_lte = interp1d(cmass_mean, (lte * 1e6), kind='linear', axis = 2, fill_value = 'extrapolate')
    f_non_lte = interp1d(cmass_mean, (non_lte * 1e6), kind='linear', axis = 2, fill_value = 'extrapolate')
f_z = interp1d(cmass_mean, z, kind='linear', axis = -1, fill_value = 'extrapolate')

############################################################
print('Applying new scale...')
lte = f_lte(cmass_scale)
non_lte = f_non_lte(cmass_scale)
new_z = f_z(cmass_scale)

print('Rearranging and taking Log10...')
lte = np.transpose(np.log10(lte), (3,2,0,1))
non_lte = np.transpose(np.log10(non_lte), (3,2,0,1))

print(f'Splitting into windows and columns...')
lte_list = []
non_lte_list = []
for i in range(pad, grid - pad):
    for j in range(pad, grid - pad):
        #lte window
        sample = lte[:, :, i-pad:i+(pad+1), j-pad:j+(pad+1)]
        lte_list.append(sample)
        #non lte point
        true = non_lte[:,:,i,j][:,:,np.newaxis,np.newaxis]
        non_lte_list.append(true)

lte = np.array(lte_list)
non_lte = np.array(non_lte_list)

print(f'Input shape {lte.shape}')
print(f'Output shape {non_lte.shape}')

print('Saving....')
with h5py.File(save_path, 'w') as f:
    dset1 = f.create_dataset("lte test windows", data=lte, dtype='f')
    dset5 = f.create_dataset("non lte test points", data = non_lte, dtype='f')
    dset2 = f.create_dataset("column mass", data = cmass_mean, dtype='f')
    dset3 = f.create_dataset("column scale", data = cmass_scale, dtype='f')
    dset4 = f.create_dataset("z", data = new_z, dtype='f')
