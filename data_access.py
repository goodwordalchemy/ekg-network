from random import shuffle
import numpy as np
import os
import pickle

from keras.preprocessing.sequence import pad_sequences
from keras.utils import Sequence

from config import get_config

MI_DATA_FILENAME = 'data/mi_filenames.txt'
TEST_DATA_FILENAME  = 'data/test_data_filenames.txt'

NUM_CHANNELS = 15
MAX_LENGTH = 32000

def _get_data_directory():
    return get_config().get('DataDirectory')

def _downsample_mis(all_filenames, target_num=1000):
    with open(MI_DATA_FILENAME, 'r') as f:
        mi_filenames = f.read().split('\n')

    num_to_select = len(mi_filenames) - target_num
    all_filenames = set(all_filenames) - set(mi_filenames[:num_to_select])

    return list(all_filenames)


def _remove_test_data_filenames(all_filenames):
    with open(TEST_DATA_FILENAME, 'r') as f:
        test_filenames = f.read().split('\n')

    return list(set(all_filenames) - set(test_filenames))


def get_train_dev_filenames(fraction=0.15):
    ptbdb_filenames = os.listdir(_get_data_directory())

    ptbdb_filenames = _downsample_mis(ptbdb_filenames)
    ptbdb_filenames = _remove_test_data_filenames(ptbdb_filenames)

    shuffle(ptbdb_filenames)

    config = get_config()
    data_subset_fraction = config.get('DataSubsetFraction')

    if data_subset_fraction < 1:
        ptbdb_filenames = ptbdb_filenames[:int(data_subset_fraction * len(ptbdb_filenames))]
        print('Only using {}% of data: {} samples'.format(data_subset_fraction * 100, len(ptbdb_filenames)))
    n_holdouts = int(fraction * len(ptbdb_filenames))

    train = ptbdb_filenames[:-n_holdouts]
    dev = ptbdb_filenames[-n_holdouts:]

    return train, dev


def load_data_files_to_array(filenames, name='', verbose=False):
    batch = []

    for i, filename in enumerate(filenames):
        if verbose:
            print('{}: '.format(name), i, filename)

        with open(_get_data_directory() + '/' + filename, 'rb') as f:
            data = pickle.load(f)

        batch.append(data)

    return batch


class CacheBatchGenerator(Sequence):

    def __init__(self, filenames, batch_size, name):
        self.batch_size = min(batch_size, len(filenames))

        mod = len(filenames) % self.batch_size

        if mod:
            self.filenames = filenames[:-mod]

        else:
            self.filenames = filenames

        self.name = name


    def __len__(self):
            return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        print('CacheBatchGenerator:{} is getting idx {} of {}.  Batch size: {}'.format(
            self.name, idx + 1, len(self), self.batch_size
        ))

        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch = load_data_files_to_array(batch_filenames, name=self.name + str(idx))

        batch_x, batch_y = zip(*batch)

        batch_x = pad_sequences(
            batch_x, dtype=batch_x[0].dtype, maxlen=MAX_LENGTH
        )

        batch_y = [1 if r == 'Myocardial infarction' else 0 for r in batch_y]
        batch_y = np.array(batch_y).reshape(-1, 1)

        return batch_x, batch_y
