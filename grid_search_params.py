import itertools

from select_model import run_models_with_params

PARAMS = {
    'num_hidden_units': [32, 16 ,4],
    'batch_size': (125, 74, -25),
    'learning_rate': [0.001],
    'epochs': [10],
}



def _convert_param_input(params):
    out = {}

    for key in params.keys():
        if isinstance(params[key], tuple):
            out[key] = list(range(*params[key]))
        else:
            out[key] = params[key]

    return out


def get_grid_search_params(params):
    _params = _convert_param_input(params)

    keys, values = zip(*_params.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutations


def main(params=PARAMS):
    params_list = get_grid_search_params(params)

    run_models_with_params(params_list)


if __name__ == '__main__':
    main()
