import keras
from keras.layers import Conv1D, Dense, MaxPooling1D, Input
from keras.models import Model
from keras.optimizers import Adam

from data_access import MAX_LENGTH, NUM_CHANNELS
from metrics import f1_score

NUM_FILTERS = 16

def create_model(params):
    input_ecg = Input(shape=(MAX_LENGTH, NUM_CHANNELS))

    tower_1 = Conv1D(NUM_FILTERS, 1, padding='same', activation='relu')(input_ecg)
    tower_1 = Conv1D(NUM_FILTERS, 3, padding='same', activation='relu')(tower_1)

    tower_2 = Conv1D(NUM_FILTERS, 1, padding='same', activation='relu')(input_ecg)
    tower_2 = Conv1D(NUM_FILTERS, 5, padding='same', activation='relu')(tower_2)

    tower_3 = MaxPooling1D(3, strides=1, padding='same')(input_ecg)
    tower_3 = Conv1D(NUM_FILTERS, 1, padding='same', activation='relu')(tower_3)

    output = keras.layers.concatenate([tower_1, tower_2, tower_3], axis=2)
    output = keras.layers.Flatten()(output)
    output = Dense(1, activation='sigmoid')(output)

    model = Model(inputs=[input_ecg], outputs=[output])

    optimizer = Adam(lr=params['learning_rate'])
    model.compile(
        optimizer=optimizer, loss='binary_crossentropy',
        metrics=['accuracy', f1_score]
    )

    return model
