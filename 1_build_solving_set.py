from helita.sim import multi3d
import numpy as np
import h5py
import os
from helita.sim.multi3d import Multi3dAtmos
from scipy.interpolate import interp1d
from scipy.integrate import cumtrapz
from scipy.ndimage import zoom

'''
takes original atmosphere cube and interpolates it to either (504,504,400) or leaves it at (252,252,400) then
creates the 3x3x400 lte windows and 1x1x400 non lte columns
'''

########################################################################################################################################################

sim_num_atms = 's525'
sim_num_sim = 's525' 
sim_name = 'qs006023'

# ---------- CBH24 -----------
#save_path = f'.../{sim_name}_{sim_num_atms}.hdf5'
#sim_path = f'...'
#atmos_path = f'...'
#dim_z = 460 
#dim = 252

# ---------- NW072100 ----------
#sim_path = f'...'
#atmos_path = f'...'
#save_path = f'.../{sim_name}_{sim_num_atms}.hdf5'
#dim_z = 475
#dim = 720

# ---------- CB24BIH -----------
#sim_path = f'...'
#atmos_path = f'...'
#save_path = f'.../{sim_name}_{sim_num_atms}.hdf5'
#dim_z = 467
#dim = 252

# ---------- QS006023 -----------
sim_path = f'...'
atmos_path = f'...'
save_path = f'.../{sim_name}_{sim_num_atms}.hdf5'
dim_z = 430
dim = 256

atmos_idx = 0    # atmos.hydrogen_populations has size (sim_num, levels, x, y, z)

interp_XY = False # interpolate X and Y to full resolution i.e. 252x252 to 504x504

#need this if sim is huge and we need to just analyize a section of it
st_x = 0 # where to start when pulling out training/test windows.
st_y = 0 # where to start when pulling out training/test windows.
lim_grid = False #  only take portion of sim, do this when grid > 252

grid = dim + 2 # account for padding periodic BC's

pad = 1 # window around point of interest

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
if interp_XY:    
    print('Will be interpolating to (504,504,400)')
else:
    print(f'Will be interpolating to ({grid},{grid},400)')

print('Padding for periodic BC...')
lte = sim.atom.nstar
non_lte = sim.atom.n
npad = ((1,1),(1,1),(0,0),(0,0))
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

print('Applying new scale...')
############################################################
lte = f_lte(cmass_scale)
non_lte = f_non_lte(cmass_scale)
new_z = f_z(cmass_scale)
print(f'new_z shape {new_z.shape}')

if interp_XY:
    print('Resizing X / Y to 504 / 504...')
    scale = 504/x_sh
    lte = zoom(lte, (scale,scale,1,1), order = 1)
    non_lte = zoom(non_lte, (scale,scale,1,1), order = 1)

print('Rearranging and taking Log10...')
lte = np.transpose(np.log10(lte), (3,2,0,1))
non_lte = np.transpose(np.log10(non_lte), (3,2,0,1))

print(f'Final atmos shape: {lte.shape}')

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

print(f'Final LTE shape: {lte.shape}')
print('Saving....')

with h5py.File(save_path, 'w') as f:
    dset1 = f.create_dataset("lte test windows", data=lte, dtype='f')
    dset5 = f.create_dataset("non lte test points", data = non_lte, dtype='f')
    dset2 = f.create_dataset("column mass", data = cmass_mean, dtype='f')
    dset3 = f.create_dataset("column scale", data = cmass_scale, dtype='f')
    dset4 = f.create_dataset("z", data = new_z, dtype='f')
        