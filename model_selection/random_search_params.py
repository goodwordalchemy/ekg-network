import numpy as np

from select_model import run_models_with_params

NUM_PARAMS = 25
EPOCHS = 10

def _generate_params():
    return {
        'num_hidden_units': np.random.random_integers(4, high=32),
        'batch_size': np.random.random_integers(50, 125),
        'learning_rate':  10**np.random.uniform(-4, -2.5),
        'epochs': EPOCHS,
    }



def get_random_search_params(num_params=NUM_PARAMS):
    params = [_generate_params() for _ in range(num_params)]

    return params


def main():
    params_list = get_random_search_params()

    run_models_with_params(params_list)


if __name__ == '__main__':
    main()
