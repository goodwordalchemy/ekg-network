import numpy as np

from config import get_config
from select_model.fit import fit_model_and_cache_results

NUM_PARAMS = 25
EPOCHS = 10


def _generate_params():
    params = []

    param_spec = get_config().get('ParameterSpecification')

    for field, spec in param_spec.items():
        r_type = spec['RandomizationType']
        low = spec['Low']
        high = spec['High']

        if r_type == 'integer':
            params[field] = np.random.randint(low, high + 1)
        elif r_type == 'log_transformed':
            params[field] = 10 ** np.random.uniform(low, high)

        return params


def get_random_search_params(num_params=NUM_PARAMS):
    params = [_generate_params() for _ in range(num_params)]

    return params


def main():
    params_list = get_random_search_params()

    run_models_with_params(params_list)


if __name__ == '__main__':
    main()
