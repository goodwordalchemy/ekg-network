import keras
from keras.layers import Conv1D, Dense, MaxPooling1D, Input
from keras.models import Model
from keras.optimizers import Adam

from config import get_config
from data_access import MAX_LENGTH, NUM_CHANNELS
from metrics import f1_score

def _get_number_of_filters(params):
    num_filters = params.get('num_filters')

    if not num_filters:
        raise Exception('You must specify num_filters in the parameter specification config')

    return num_filters

def _get_number_of_layers(params):
    num_layers = params.get('num_layers')

    if not num_layers:
        raise Exception('You must specify num_layers in the parameter specification config')

    return num_layers



def _inception_module(input_tensor, num_filters):
    tower_1 = Conv1D(num_filters, 1, padding='same', activation='relu')(input_tensor)
    tower_1 = Conv1D(num_filters, 3, padding='same', activation='relu')(tower_1)

    tower_2 = Conv1D(num_filters, 1, padding='same', activation='relu')(input_tensor)
    tower_2 = Conv1D(num_filters, 5, padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling1D(3, strides=1, padding='same')(input_tensor)
    tower_3 = Conv1D(num_filters, 1, padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=2)

    return output


def create_model(params):
    input_ecg = output = Input(shape=(MAX_LENGTH, NUM_CHANNELS))

    for _ in range(_get_number_of_layers(params)):
        output = _inception_module(output, _get_number_of_filters(params))

    output = keras.layers.Flatten()(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[input_ecg], outputs=[output])

    optimizer = Adam(lr=params['learning_rate'])
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy',
        metrics=['accuracy', f1_score]
    )

    return model