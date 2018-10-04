import os

from config import get_config
from select_model.fit import fit_model_and_cache_results

def run_models_with_params(params_list, create_model_function):
    results_directory = get_config().get('ResultsDirectory')
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)

    print('Searching through {} parameter sets'.format(len(params_list)))

    for i, params in enumerate(params_list):
        print('\ntesting model {} of {} with params: {}'.format(i + 1, len(params_list), params))

        model = create_model_function(params)

        fit_model_and_cache_results(model, params)
