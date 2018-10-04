import itertools
import os

from select_model.search_methods.common import run_models_with_params


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


def main(create_model_function, params_spec):
    params_list = get_grid_search_params(params_spec)

    run_models_with_params(params_list, create_model_function)


if __name__ == '__main__':
    main()
