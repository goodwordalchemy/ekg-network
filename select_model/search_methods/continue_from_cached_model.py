import os

from keras.models import load_model

from param_utils import params_dict_to_str
from select_model import f1_score, fit_model_and_cache_results, RESULTS_DIRECTORY


DEBUG_CACHED_MODEL_PATH = 'results/debug__batch_size_8__epochs_1__learning_rate_0.005__num_hidden_units_5_model.hd5'

EPOCHS_TO_TRAIN = 1
PARAMS = {
   'batch_size': 82,
   'epochs': 10,
   'learning_rate': 0.002968950056415773,
   'num_hidden_units': 17
}

def load_cached_model(params):
    cached_model_path = params_dict_to_str(params)
    cached_model_path = cached_model_path + '.hd5'
    cached_model_path = os.path.join(RESULTS_DIRECTORY, cached_model_path)

    # ###
    # print('DEBUG: using cached model path')
    # cached_model_path = DEBUG_CACHED_MODEL_PATH
    # ###

    model = load_model(cached_model_path, custom_objects={'f1_score': f1_score})

    return model


def main(params=PARAMS):
    model = load_cached_model(params)

    params.update({
        'initial_epoch': params['epochs'],
        'epochs': EPOCHS_TO_TRAIN + params['epochs'],
    })

    print(params)

    fit_model_and_cache_results(model, params)


if __name__ == '__main__':
    main()
