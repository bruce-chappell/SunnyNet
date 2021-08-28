import os
import h5py
from networkUtils.atmosphereFunctions import predict_populations

if __name__ == "__main__":



    MODEL = f'SunnyNet_3x3'    # pick one from networkUtilities/atmosphereFunctions
    MODEL_NAME = f'my_model.pt'
    TRAIN_FILE = f'my_data.hdf5'


    # file to dataset NN was trained on
    TRAIN_DATA = f'/path/to/{TRAIN_FILE}'

    # file of prepped data from 1_build_solving_set.py
    TEST_DATA = f'/path/to/my_test_data.hdf5'

    # save path
    SAVE = f'/path/to/my_prediction.hdf5'

    folder1 = '/'.join(SAVE.split('/')[:-1])
    try:
        assert os.path.exists(folder1) == True
    except AssertionError:
        os.mkdir(folder1)

    assert os.path.exists(SAVE) == False, "dont want to overwrite just incase. check filenames"

    ## predict atmos configuration ##
    pred_config = {
        'cuda': True,          # False if not running on cuda enabled machine
        'model': MODEL,
        'model_path': f'/path/to/trained/{MODEL_NAME}',
        'channels': 6,         # number of channels
        'features': 400,       # z dimension
        'mode': 'testing',
        'multi_gpu_train': False,
        'loss_fxn': 'MSELoss', # pick one from networkUtils/lossFunctions.py or a pyotch loss function class name ex MSELoss
        'alpha': 0.2,          # to turn off make None, weight in loss calculation between mass conservation and cell by cell error
        'output_XY': 252,      # x,y size of predicted atmosphere. Needs to be the same as the original atmosphere size
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
