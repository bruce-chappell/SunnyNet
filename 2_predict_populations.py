import os
import h5py
from networkUtils.atmosphereFunctions import predict_populations

if __name__ == "__main__":

    SIM_NUM  = 's525'
    SIM_NAME = 'qs006023'
    MODEL = 'SunnyNet'
    MODEL_NAME = 'model_name'
    TRAIN_FILE = 'train_file.hdf5'

    # file to dataset NN was trained on
    TRAIN_DATA = f'path/{TRAIN_FILE}'

    # file of prepped data from 1_build_solving_set.py
    TEST_DATA = f'path/{SIM_NAME}_{SIM_NUM}.hdf5'

    # save path
    SAVE = f'path/{MODEL_NAME}/{SIM_NAME}_{SIM_NUM}.hdf5'

    folder1 = '/'.join(SAVE.split('/')[:-1])
    try:
        assert os.path.exists(folder1) == True
    except AssertionError:
        os.mkdir(folder1)

    assert os.path.exists(SAVE) == False, "dont want to overwrite just incase. check filenames"

    ## predict atmos configuration ##
    pred_config = {
        'cuda': True,
        'model': MODEL,
        'model_path': f'path/{MODEL_NAME}.pt',
        'channels': 6,
        'features': 400,
        'mode': 'testing',
        'multi_gpu_train': False,
        'loss_fxn': 'MSELoss',
        'alpha': 0.2,
        'output_XY': 256,
    }

    final, z, cmass_mean, cmass_scale = predict_populations(TEST_DATA, TRAIN_DATA, pred_config)

    print('Exponentiate')
    final = 10**final
    print(f'Atmos shape: {final.shape}')
    with h5py.File(SAVE, 'w') as f:
        dset1 = f.create_dataset("populations", data = final, dtype='f')
        dset1.attrs['z'] = z
        dset1.attrs['cmass_mean'] = cmass_mean
        dset1.attrs['cmass_scale'] = cmass_scale
