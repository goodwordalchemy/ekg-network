from keras.layers import Dense, LSTM
from keras.models import Sequential
from keras.optimizers import Adam

from metrics import all_metrics
from data_access import MAX_LENGTH, NUM_CHANNELS

def create_model(params):
    optimizer = Adam(lr=params['learning_rate'])

    model = Sequential([
        LSTM(params['num_hidden_units'], input_shape=(MAX_LENGTH, NUM_CHANNELS)),
        Dense(1, activation='sigmoid'),
    ])

    model.compile(
        optimizer=optimizer, loss='binary_crossentropy',
        metrics=['accuracy', *all_metrics]
    )

    return model


if __name__ == '__main__':
    import grid_search_params

    ### DEBUG PARAMS ###
    PARAMS = {
        'num_hidden_units': [5],
        'batch_size': [8],
        'learning_rate': [0.005],
        'epochs': [1],
    }
    ###################

    grid_search_params.main(params=PARAMS)
