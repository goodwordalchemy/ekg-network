import os
import pickle
import time
from random import shuffle

import keras
import keras.backend as K
from keras.optimizers import Adam
from keras.preprocessing.sequence import pad_sequences
import numpy as np

# Agg is used so plots can be saved on a remote server
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Activation, Input, LSTM
from keras.utils import Sequence

from param_utils import params_dict_to_str

DATA_SUBSET_FRACTION = 1
DATA_DIRECTORY = '/mnt/disks/ptbdb/data'
RESULTS_DIRECTORY = '/mnt/disks/ptbdb/results/'

### DEBUG PARAMS ###
DATA_SUBSET_FRACTION = .01 / 2
DATA_DIRECTORY = 'data/truncated_samples' # Comment out on remote server.
RESULTS_DIRECTORY = 'results'
#####################


MI_DATA_FILENAME = 'mi_filenames.txt'
TEST_DATA_FILENAME  = 'test_data_filenames.txt'
MAX_LENGTH = 32000
NUM_CHANNELS = 15



def _get_results_path():
    return RESULTS_DIRECTORY


def downsample_mis(all_filenames, target_num=1000):
    with open(MI_DATA_FILENAME, 'r') as f:
        mi_filenames = f.read().split('\n')

    num_to_select = len(mi_filenames) - target_num
    all_filenames = set(all_filenames) - set(mi_filenames[:num_to_select])

    return list(all_filenames)


def remove_test_data_filenames(all_filenames):
    with open(TEST_DATA_FILENAME, 'r') as f:
        test_filenames = f.read().split('\n')

    return list(set(all_filenames) - set(test_filenames))


def get_train_dev_filenames(fraction=0.15):
    ptbdb_filenames = os.listdir(DATA_DIRECTORY)

    ptbdb_filenames = downsample_mis(ptbdb_filenames)
    ptbdb_filenames = remove_test_data_filenames(ptbdb_filenames)

    shuffle(ptbdb_filenames)

    if DATA_SUBSET_FRACTION < 1:
        ptbdb_filenames = ptbdb_filenames[:int(DATA_SUBSET_FRACTION * len(ptbdb_filenames))]
        print('Only using {}% of data: {} samples'.format(DATA_SUBSET_FRACTION * 100, len(ptbdb_filenames)))
    n_holdouts = int(fraction * len(ptbdb_filenames))

    train = ptbdb_filenames[:-n_holdouts]
    dev = ptbdb_filenames[-n_holdouts:]

    return train, dev


def load_data_files_to_array(filenames):
    batch = []

    for i, filename in enumerate(filenames):
        with open(DATA_DIRECTORY + '/' + filename, 'rb') as f:
            data = pickle.load(f)

        batch.append(data)

    return batch


class CacheBatchGenerator(Sequence):

    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = min(batch_size, len(filenames))

    def __len__(self):
            return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        print('CacheBatchGenerator is getting idx {} of {}'.format(idx, len(self)))

        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch = load_data_files_to_array(batch_filenames)

        batch_x, batch_y = zip(*batch)

        batch_x = pad_sequences(
            batch_x, dtype=batch_x[0].dtype, maxlen=MAX_LENGTH
        )

        batch_y = [1 if r == 'Myocardial infarction' else 0 for r in batch_y]
        batch_y = np.array(batch_y).reshape(-1, 1)

        return batch_x, batch_y


def f1_score(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon()
    c3 = K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon()

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
            return 0

    # How many selected items are relevant?
    precision = c1 / c2

    # How many relevant items are selected?
    recall = c1 / c3

    # Calculate f1_score
    f1_score = 2 * (precision * recall) / (precision + recall + K.epsilon())

    return f1_score


def get_random_params_list(n):
    perms = get_grid_search_params(PARAMS)

    return np.random.choice(perms, size=n, replace=False)


def fit_model(params):
    results = {}

    results['paramaeters'] = params

    optimizer = Adam(lr=params['learning_rate'])

    model = Sequential([
        LSTM(params['num_hidden_units'], input_shape=(MAX_LENGTH, NUM_CHANNELS)),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=optimizer, loss='binary_crossentropy',
        metrics=['accuracy', f1_score]
    )

    results['model'] = model

    # Train model
    t_before = time.time()

    train_files, dev_files = get_train_dev_filenames()
    training_batch_generator = CacheBatchGenerator(
        train_files, batch_size=params['batch_size']
    )
    dev_batch_generator = CacheBatchGenerator(
        dev_files, batch_size=params['batch_size']
    )

    history = model.fit_generator(
        generator=training_batch_generator, validation_data=dev_batch_generator,
        epochs=params['epochs'],
        use_multiprocessing=True, workers=8, max_queue_size=8,
        verbose=2)

    results['history'] = history

    t_after = time.time()
    total_time = t_after - t_before
    results['training_time'] = total_time

    return results


def get_time_uuid():
    return str(time.time()).split('.')[0]


def run_models_with_params(params_list):
    if not os.path.exists(_get_results_path()):
        os.mkdir(_get_results_path())

    print('Searching through {} parameter sets'.format(len(params_list)))

    for i, params in enumerate(params_list):
        print('\ntesting model {} of {} with params: {}'.format(i + 1, len(params_list), params))

        model_result = fit_model(params)

        model_result['history'] = model_result['history'].history

        _run_name = params_dict_to_str(params)
        if DATA_SUBSET_FRACTION < 1:
            _run_name = 'debug__' + _run_name

        _model_path = os.path.join(_get_results_path(), _run_name + '_model.hd5')
        model_result['model'].save(_model_path)
        del model_result['model']

        _results_path = os.path.join(_get_results_path(), _run_name + '_results.pkl')
        with open(_results_path, 'wb') as f:
            pickle.dump(model_result, f)

if __name__ == '__main__':
    import grid_search_params

    ### DEBUG PARAMS ###
    PARAMS = {
        'num_hidden_units': [5],
        'batch_size': [8],
        'learning_rate': [0.005],
        'epochs': [1],
    }
    ###################

    grid_search_params.main(params=PARAMS)
