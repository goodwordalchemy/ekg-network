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


DATA_SUBSET_FRACTION = 0.1

N_MODELS = 1
N_EPOCHS = 1

DATA_DIRECTORY = '/mnt/disks/ptbdb/data'
DATA_DIRECTORY = 'data/truncated_samples'
TEST_DATA_FILENAME  = 'test_data_filenames.txt'
RESULTS_DIRECTORY = 'results'
MAX_LENGTH = 32000
NUM_CHANNELS = 15

NUM_HIDDEN_UNITS_MIN = 25
NUM_HIDDEN_UNITS_MAX = 50
BATCH_SIZE_MIN = 50
BATCH_SIZE_MAX = 100


def downsample_mis(all_filenames, target_num=1000):
    with open('mi_filenames.txt', 'r') as f:
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
        print('Only using {}% of data'.format(DATA_SUBSET_FRACTION * 100))
        ptbdb_filenames = ptbdb_filenames[:int(DATA_SUBSET_FRACTION * len(ptbdb_filenames))]

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


def get_random_params():
    train_batch, _, = get_train_dev_filenames()

    return {
        'num_hidden_units': np.random.randint(NUM_HIDDEN_UNITS_MIN, NUM_HIDDEN_UNITS_MAX),
        'batch_size': np.random.randint(BATCH_SIZE_MIN, BATCH_SIZE_MAX),
        'learning_rate': 10**(-4 * np.random.uniform(low=0.5, high=1.0))
    }


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

    params = get_random_params()
    params.update({'epochs': epochs})

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


def find_models(n_models, n_epochs):
    if not RESULTS_DIRECTORY in os.listdir('.'):
        os.mkdir(RESULTS_DIRECTORY)

    run_name = get_time_uuid()
    run_dir = os.path.join(RESULTS_DIRECTORY, run_name)
    os.mkdir(run_dir)

    for i in range(N_MODELS):
        print('\ntesting model {} of {}'.format(i + 1, n_models))

        model_result = run_model_with_random_params(epochs=n_epochs)
        model_result['history'] = model_result['history'].history
        del model_result['model']

        with open(os.path.join(run_dir, str(i)), 'wb') as f:
            pickle.dump(model_result, f)


if __name__ == '__main__':
       find_models(N_MODELS, N_EPOCHS)
