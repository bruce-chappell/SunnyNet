import os
import sys
import h5py
import numpy as np
from astropy import units
from astropy import constants as const
from scipy.interpolate import interp1d
from helita.sim import multi3d
from helita.sim.multi3d import Multi3dAtmos
from networkUtils.intensityFunctions import calculate_halpha

# read important vars
sim_num = 's425_half'
sim_num_atm = 's0425_half'
sim_name = 'cb24bih'
dim = 252
dim_z = 467
net_name = 'cb24bih_ColMass_3x3_single_50e_128b_2a_ComboData'

st_x = 0
st_y = 0
grid = 252

#atmos_path = f'path/name'
#multi3d_path = f'path/name'

atmos_path = f'path/name'
multi3d_path = f'path/name'

pop_path = f'path/{net_name}/{sim_name}_{sim_num}.hdf5'


os.chdir(multi3d_path)
data3d = multi3d.Multi3dOut(directory='.')
data3d.readall()
data3d.set_transition(3,2)
multi3d_intensity_orig = data3d.readvar('ie')
wave_vec = data3d.d.l.to('nm')

atmos = Multi3dAtmos(atmos_path, dim, dim, dim_z)

########################################################

print('Loading matrices to memory...')
# load from file and test

with h5py.File(pop_path, 'r') as f:
    pop = f["populations"][:]
    z_interp = f["populations"].attrs["z"]
    cmass_mean = f["populations"].attrs["cmass_mean"]
    cmass_scale = f["populations"].attrs["cmass_scale"]

nn_save = f'path/{net_name}/{sim_name}_{sim_num}.npy'
multi3d_save = f'path/multi3d/{sim_name}_{sim_num}.npy'

#nn_load = f'path/{net_name}/{sim_name}_{sim_num}_topleft252.npy'
#multi3d_load = f'path/multi3d/{sim_name}_{sim_num}_topleft252.npy'

folder1 = '/'.join(nn_save.split('/')[:-1])
folder2 = '/'.join(multi3d_save.split('/')[:-1])
try:
    assert os.path.exists(folder1) == True
except AssertionError:
    os.mkdir(folder1)
try:
    assert os.path.exists(folder2) == True
except AssertionError:
    os.mkdir(folder2)

if os.path.exists(nn_save):
    print('NN intensity already calculated, checking Multi3d...')
    
else:
    print('Calculating NN intensity...')
    hpops = pop * (units.m**-3)
    hpops = np.transpose(hpops, (3,0,1,2))

    z_interp = z_interp * units.m
    
    print(hpops.shape)
    print(z_interp.shape)

    # need to shrink size / interpolate to match h populations
    f_t = interp1d(cmass_mean, atmos.temp[st_x:st_x+grid,st_y:st_y+grid,...], kind='linear', axis = -1, fill_value = 'extrapolate')
    temp = f_t(cmass_scale) * units.K

    f_e = interp1d(cmass_mean, atmos.ne[st_x:st_x+grid,st_y:st_y+grid,...] * 1e6, kind='linear', axis = -1, fill_value = 'extrapolate')
    e_density = f_e(cmass_scale) * (units.m**-3)


    f_v = interp1d(cmass_mean, atmos.vz[st_x:st_x+grid,st_y:st_y+grid,...] * 1e3, kind='linear', axis = -1, fill_value = 'extrapolate')
    v_los = f_v(cmass_scale) * (units.m / units.s)

    pressure = (e_density + hpops.sum(0)*1.1) * const.k_B * temp

    intensity = calculate_halpha(hpops, temp, e_density, pressure, v_los, z_interp, wave_vec)

    np.save(nn_save, intensity.value)

if os.path.exists(multi3d_save):
    print('Multi3d intensity already calculated, exiting...')
    sys.exit(0)

else:
    print('Calculating multi3d intensity...')
    hpops_multi3d = data3d.atom.n[st_x:st_x+grid,st_y:st_y+grid,...] * 1e6 * (units.m**-3)
    hpops_multi3d = np.transpose(hpops_multi3d, (3,0,1,2))

    z = data3d.geometry.z * 1e-2 * units.m

    temp = atmos.temp[st_x:st_x+grid,st_y:st_y+grid,...] * units.K
    e_density = atmos.ne[st_x:st_x+grid,st_y:st_y+grid,...] * 1e6 * (units.m**-3)
    v_los = atmos.vz[st_x:st_x+grid,st_y:st_y+grid,...] * 1e3 * (units.m / units.s)
    pressure = (e_density + hpops_multi3d.sum(0)*1.1) * const.k_B * temp

    intensity_multi3d = calculate_halpha(hpops_multi3d, temp, e_density, pressure, v_los, z, wave_vec)
    
    np.save(multi3d_save, intensity_multi3d.value)
