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
DATA_DIRECTORY = 'data/untracked/truncated_samples' # Comment out on remote server.
RESULTS_DIRECTORY = 'data/untracked/results'
#####################


MI_DATA_FILENAME = 'data/mi_filenames.txt'
TEST_DATA_FILENAME  = 'data/test_data_filenames.txt'
MAX_LENGTH = 32000
NUM_CHANNELS = 15



def fit_model(model, params):
    results = {}
    results['paramaeters'] = params
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
        epochs=params['epochs'], initial_epoch=params.get('initial_epoch'),
        use_multiprocessing=True, workers=8, max_queue_size=8,
        verbose=2)

    results['history'] = history.history

    t_after = time.time()
    total_time = t_after - t_before
    results['training_time'] = total_time

    return results


def _get_run_name(params):
    _run_name = params_dict_to_str(params)
    if DATA_SUBSET_FRACTION < 1:
        _run_name = 'debug__' + _run_name

    return _run_name


def _get_model_path(params):
    _run_name = _get_run_name(params)

    _model_path = os.path.join(RESULTS_DIRECTORY, _run_name + '_model.hd5')

    return _model_path

def _get_results_path(params):
    _run_name = _get_run_name(params)

    _results_path = os.path.join(RESULTS_DIRECTORY, _run_name + '_results.pkl')

    return _results_path


def fit_model_and_cache_results(model, params):
    model_result = fit_model(model, params)

    _model_path = _get_model_path(params)
    model_result['model'].save(_model_path)
    del model_result['model']

    _results_path = _get_results_path(params)
    with open(_results_path, 'wb') as f:
        pickle.dump(model_result, f)
