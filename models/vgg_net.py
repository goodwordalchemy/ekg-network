from keras.layers import (
    Activation, AveragePooling1D, BatchNormalization, Conv1D, Dense,
    Flatten, Input, MaxPooling1D
)
from keras.models import Model
from keras.optimizers import Adam

from data_access import MAX_LENGTH, NUM_CHANNELS
from metrics import all_metrics

BATCH_NORM_AXIS = 1

NUM_OF_VGG_BLOCKS = 5

BASE_EXP = 4
BLOCK_NUM_TO_NUM_FILTERS_MAPPER = {
    0: 2**BASE_EXP, 1: (2**BASE_EXP + 1), 2: 2**(BASE_EXP + 2),
    3: 2**(BASE_EXP + 4), 4:2**(BASE_EXP + 4),
    'GAP': 2**BASE_EXP + 4
}


def _get_num_convolutions(params, block_num):
    return params['num_convolutions_block_{block_num}'.format(block_num=block_num)]

def _get_num_filters(block_num):
    return BLOCK_NUM_TO_NUM_FILTERS_MAPPER[block_num]

def _get_run_batch_norm(params):
    return params.get('run_batch_norm', False)

def _convolution_layer(input_tensor, num_filters, run_batch_norm):
    output = Conv1D(num_filters, 3, padding='same')(input_tensor)

    if run_batch_norm:
        output = BatchNormalization(axis=BATCH_NORM_AXIS)(output)

    output = Activation('relu')(output)

    return output


def _max_pooling_layer(input_tensor):
    output = MaxPooling1D(2, strides=2, padding='valid')(input_tensor)

    return output


def _vgg_block(tensor, num_convolutions, num_filters, run_batch_norm=False):
    for _ in range(num_convolutions):
        tensor = _convolution_layer(tensor, num_filters, run_batch_norm)

    tensor = _max_pooling_layer(tensor)

    return tensor


def create_model(params):
    input_ecg = output = Input(shape=(MAX_LENGTH, NUM_CHANNELS))

    for block_num in range(NUM_OF_VGG_BLOCKS):
       num_convolutions = _get_num_convolutions(params, block_num)
       num_filters = _get_num_filters(block_num)

       output = _vgg_block(
            output, num_convolutions, num_filters, run_batch_norm=_get_run_batch_norm(params)
        )

    output = AveragePooling1D(
        pool_size=BLOCK_NUM_TO_NUM_FILTERS_MAPPER['GAP']
    )(output)

    output = Flatten()(output)

    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[input_ecg], outputs=[output])

    optimizer = Adam(lr=params['learning_rate'])
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy',
        metrics=['accuracy', *all_metrics]
    )

    model.summary()

    return model
