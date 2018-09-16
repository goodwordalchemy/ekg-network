import itertools

def _convert_param_input(params):
    out = {}

    for key in params.keys():
        if isinstance(params[key], tuple):
            out[key] = list(range(*params[key]))
        else:
            out[key] = params[key]

    return out


def get_param_permutuations(params):
    keys, values = zip(*params.items())
    permutations = [dict(zip(keys, v)) for v in itertools.product(*values)]

    return permutations

if __name__ == '__main__':
    example_params = {
        'a': (1, 10, 2),
        'b': [1, 2, 6, 9]
    }

    converted_params = _convert_param_input(example_params)
    params_product = get_param_permutuations(converted_params)

    print(params_product)
