import os

import numpy as np

from select_model.search_methods.common import run_models_with_params

NUM_PARAMS = 25


def _generate_params(params_spec):
    params = {}

    for field, spec in params_spec.items():
        r_type = spec['RandomizationType']
        low = spec['Low']
        high = spec['High']

        if r_type == 'integer':
            params[field] = np.random.randint(low, high + 1)
        elif r_type == 'log_transformed':
            params[field] = 10 ** np.random.uniform(low, high)

    return params


def _get_random_search_params(params_spec, num_params=NUM_PARAMS):
    params = [_generate_params(params_spec) for _ in range(num_params)]

    return params




def main(create_model_function, params_spec):
    params_list = _get_random_search_params(params_spec)

    run_models_with_params(params_list, create_model_function)



if __name__ == '__main__':
    main()
