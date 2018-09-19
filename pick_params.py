import itertools

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


def params_dict_to_str(params_dict):
    pstring = ''

    for key, param in sorted(params_dict.items()):
        pstring += '{}_{}__'.format(key, str(param))

    pstring = pstring[:-2]

    return pstring


if __name__ == '__main__':
    example_params = {
        'a': (1, 10, 2),
        'b': [1, 2, 6, 9]
    }

    params_product = get_param_permutuations(example_params)

    print(params_product)

    print(params_dict_to_str(params_product[0]))

