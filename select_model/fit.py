import os
import pickle
import time
from random import shuffle

import keras
import keras.backend as K
import numpy as np
from keras.callbacks import EarlyStopping, ReduceLROnPlateau

from config import get_config
from data_access import CacheBatchGenerator, get_train_dev_filenames
from utils.param_utils import params_dict_to_str

import os


def _get_results_directory():
    return get_config().get('ResultsDirectory')


def _get_model_kwargs(params):
    kwargs = {}

    if 'epochs' in params:
        kwargs['epochs'] = params['epochs']

    if 'initial_epoch' in params:
        kwargs['initial_epoch'] = params['initial_epoch']

    return kwargs


def fit_model(model, params):
    results = {}
    results['parameters'] = params
    results['model'] = model

    # Train model
    t_before = time.time()

    train_files, dev_files = get_train_dev_filenames()
    training_batch_generator = CacheBatchGenerator(
        train_files, batch_size=params['batch_size'], name='training_gen'
    )
    dev_batch_generator = CacheBatchGenerator(
        dev_files, batch_size=params['batch_size'], name='dev_gen'
    )


    params_to_pass = _get_model_kwargs(params)

    reduce_lr = ReduceLROnPlateau(
        monitor='val_loss', factor=0.2, patience=5, min_lr=0.0001, verbose=1
    )
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=10, verbose=1
    )
    history = model.fit_generator(
        generator=training_batch_generator,
        validation_data=dev_batch_generator,
        # use_multiprocessing=True, workers=4, max_queue_size=4,
        verbose=2,
        callbacks=[reduce_lr, early_stopping],
        **params_to_pass)

    results['history'] = history.history

    t_after = time.time()
    total_time = t_after - t_before
    results['training_time'] = total_time

    return results


def _get_run_name(params):
    _run_name = params_dict_to_str(params)
    if get_config().get('DataSubsetFraction') < 1:
        _run_name = 'debug__' + _run_name

    return _run_name


def _get_model_path(params):
    _run_name = _get_run_name(params)

    _model_path = os.path.join(_get_results_directory(), _run_name + '_model.hd5')

    return _model_path

def _get_results_path(params):
    _run_name = _get_run_name(params)

    _results_path = os.path.join(_get_results_directory(), _run_name + '_results.pkl')

    return _results_path


def fit_model_and_cache_results(model, params):
    model_result = fit_model(model, params)

    os.makedirs(_get_results_directory(), exist_ok=True)

    _model_path = _get_model_path(params)
    model_result['model'].save(_model_path)
    del model_result['model']

    _results_path = _get_results_path(params)
    with open(_results_path, 'wb') as f:
        pickle.dump(model_result, f)
