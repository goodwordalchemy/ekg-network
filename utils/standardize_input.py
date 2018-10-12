from keras.layers import Lambda
import keras.backend as K

def standardize_input_tensor(x, axis=1):
    mean = K.mean(x, axis=axis, keepdims=True)
    std = K.std(x, axis=axis, keepdims=True)

    return (x - mean) / std


def get_standize_input_layer(axis=1):
    return Lambda(lambda x: standardize_input_tensor(x, axis=axis))
