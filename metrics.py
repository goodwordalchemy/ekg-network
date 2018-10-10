import keras.backend as K

def f1_score(y_true, y_pred):
    _precision = precision(y_true, y_pred)
    _recall = recall(y_true, y_pred)

    # Calculate f1_score
    f1_score = 2 * (_precision * _recall) / (_precision + _recall + K.epsilon())

    return f1_score


def precision(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c2 = K.sum(K.round(K.clip(y_pred, 0, 1))) + K.epsilon()

    # If there are no true predictions, fix the F1 score at 0.
    if c2 == 0:
            return K.epsilon()

    # How many selected items are relevant?
    _precision = c1 / c2

    return _precision


def recall(y_true, y_pred):
    # Count positive samples.
    c1 = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    c3 = K.sum(K.round(K.clip(y_true, 0, 1))) + K.epsilon()

    # If there are no true samples, fix the F1 score at 0.
    if c3 == 0:
            return K.epsilon()

    # How many relevant items are selected?
    _recall = c1 / c3

    return _recall

all_metrics = [f1_score, precision, recall]
