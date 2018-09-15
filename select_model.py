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

from pick_params import get_param_permutuations

DOWNSAMPLE_PARAMS = 1
PARAMS = {
    'num_hidden_units': (30, 51, 10),
    'batch_size': (50, 101, 25),
    'learning_rate': [0.001],
    'epochs': [10],
}

# DEBUG PARAMS
DATA_SUBSET_FRACTION = 0.1
PARAMS = {
    'num_hidden_units': [3],
    'batch_size': [50],
    'learning_rate': [0.001],
    'epochs': [1],
}

# CONSTANTS
DATA_DIRECTORY = '/mnt/disks/ptbdb/data'
DATA_DIRECTORY = 'data/truncated_samples' # Comment out on remote server.

RESULTS_DIRECTORY = 'results'

MI_DATA_FILENAME = 'mi_filenames.txt'
TEST_DATA_FILENAME  = 'test_data_filenames.txt'
MAX_LENGTH = 32000
NUM_CHANNELS = 15


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

    shuffle(ptbdb_filenames)

    if DATA_SUBSET_FRACTION < 1:
        ptbdb_filenames = ptbdb_filenames[:int(DATA_SUBSET_FRACTION * len(ptbdb_filenames))]
        print('Only using {}% of data: {} samples'.format(DATA_SUBSET_FRACTION * 100, len(ptbdb_filenames)))

    n_holdouts = int(fraction * len(ptbdb_filenames))

    train = ptbdb_filenames[:n_holdouts]
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
        print('CacheBatchGenerator is getting idx {} of {}'.format(idx, self.__len__()))

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
        c2 = K.sum(K.round(K.clip(y_pred, 0, 1)))
        c3 = K.sum(K.round(K.clip(y_true, 0, 1)))

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
    perms = get_param_permutuations(PARAMS)

    return np.random.choice(perms, size=n, replace=False)


def fit_model(params):
    results = {}

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


def run_model_with_random_params(epochs=5):
    results = {}

    params = get_random_params_list()
    perms = get_param_permutuations(PARAMS)

    results = fit_model(params)

    print('trying parameters: {}'.format(params))
    results['parameters'] = params

    # Test model
    train_files, dev_files = get_train_dev_filenames()
    training_batch_generator = CacheBatchGenerator(train_files, batch_size=params['batch_size'])
    dev_batch_generator = CacheBatchGenerator(dev_files, batch_size=params['batch_size'])

    results['train_metrics'] = results['model'].evaluate_generator(training_batch_generator)
    results['dev_metrics'] = results['model'].evaluate_generator(dev_batch_generator)

    # Print results
    print('metrics names: ', results['model'].metrics_names)
    print('train_scores: ', results['train_metrics'])
    print('dev_scores: ', results['dev_metrics'])

    return results


def get_time_uuid():
    return str(time.time()).split('.')[0]


def find_models():
    if not RESULTS_DIRECTORY in os.listdir('.'):
        os.mkdir(RESULTS_DIRECTORY)

    run_name = get_time_uuid()
    run_dir = os.path.join(RESULTS_DIRECTORY, run_name)
    os.mkdir(run_dir)

    params_list = get_param_permutuations(PARAMS)

    if DOWNSAMPLE_PARAMS < 1:
        shuffle(params_list)
        params_list = params_list[:int(DOWNSAMPLE_PARAMS * len(params_list))]

    print('Searching through {} parameter sets'.format(len(params_list)))

    for i, params in enumerate(params_list):
        print('\ntesting model {} of {} with params: {}'.format(i + 1, len(params_list), params))

        model_result = fit_model(params)

        model_result['history'] = model_result['history'].history
        del model_result['model']

        with open(os.path.join(run_dir, str(i)), 'wb') as f:
            pickle.dump(model_result, f)


if __name__ == '__main__':
       find_models()
