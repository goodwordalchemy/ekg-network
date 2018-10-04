import itertools
import os

from config import get_config
from select_model.fit import fit_model_and_cache_results


def _convert_param_input(params_spec):
    out = {}

    for key in params_spec.keys():
        cur = params_spec[key]
        interpret_as = cur['InterpretAs']
        values = cur['Values']

        if interpret_as == 'value_range':
            out[key] = list(range(*values))
        elif interpret_as == 'value_list':
            out[key] = values

    return out


def get_grid_search_params(params_spec):
    _params = _convert_param_input(params_spec)

    keys, values = zip(*_params.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutations


def run_models_with_params(params_list, create_model_function):
    results_directory = get_config().get('ResultsDirectory')
    if not os.path.exists(results_directory):
        os.mkdir(results_directory)

    print('Searching through {} parameter sets'.format(len(params_list)))

    for i, params in enumerate(params_list):
        print('\ntesting model {} of {} with params: {}'.format(i + 1, len(params_list), params))

        model = create_model_function(params)

        fit_model_and_cache_results(model, params)


def main(create_model_function, params_spec):
    params_list = get_grid_search_params(params_spec)

    run_models_with_params(params_list, create_model_function)


if __name__ == '__main__':
    main()
