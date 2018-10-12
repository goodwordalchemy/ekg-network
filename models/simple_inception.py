import keras
from keras.layers import Activation, BatchNormalization, Conv1D, Dense, MaxPooling1D, Input
from keras.models import Model
from keras.optimizers import Adam

from config import get_config
from data_access import MAX_LENGTH, NUM_CHANNELS
from metrics import all_metrics
from utils.standardize_input import get_standize_input_layer


BATCH_NORM_AXIS = 1

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


def _get_conv_tower(input_tensor, num_filters, size):
    tower = Conv1D(num_filters, 1, padding='same')(input_tensor)
    tower = BatchNormalization(axis=BATCH_NORM_AXIS)(tower)
    tower = Activation('relu')(tower)

    tower = Conv1D(num_filters, size, padding='same')(input_tensor)
    tower = BatchNormalization(axis=BATCH_NORM_AXIS)(tower)
    tower = Activation('relu')(tower)

    return tower


def _get_pool_tower(input_tensor, num_filters, size):
    tower = MaxPooling1D(size, strides=1, padding='same')(input_tensor)

    tower = Conv1D(num_filters, 1, padding='same')(input_tensor)
    tower = BatchNormalization(axis=BATCH_NORM_AXIS)(tower)
    tower = Activation('relu')(tower)

    return tower


def _inception_module(input_tensor, num_filters):
    tower_1 = _get_conv_tower(input_tensor, num_filters, 3)
    tower_2 = _get_conv_tower(input_tensor, num_filters, 5)
    tower_3 = _get_pool_tower(input_tensor, num_filters, 3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=2)

    return output


def create_model(params):
    input_ecg = output = Input(shape=(MAX_LENGTH, NUM_CHANNELS))

    output = get_standize_input_layer(axis=1)(output)

    for _ in range(_get_number_of_layers(params)):
        output = _inception_module(output, _get_number_of_filters(params))
        output = MaxPooling1D(3, strides=2)(output)

    output = keras.layers.Flatten()(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[input_ecg], outputs=[output])

    optimizer = Adam(lr=params['learning_rate'])
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy',
        metrics=['accuracy', *all_metrics]
    )

    return model
