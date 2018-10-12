import numpy as np

from data_access import MAX_LENGTH, NUM_CHANNELS


def get_training_mean(training_sequence):
    _sum = np.zeros(NUM_CHANNELS)

    for idx in range(len(training_sequence)):
        batch_sum  = training_sequence[idx][0].sum(axis=1)
        _sum += batch_sum.sum(axis=0)

    _total = MAX_LENGTH * len(training_sequence)

    return _sum / _total


def get_training_std(training_sequence, training_mean):
    _sum = np.zeros(NUM_CHANNELS)

    for idx in range(len(training_sequence)):
        batch_sum = np.sum((training_sequence[idx][0] - training_mean) ** 2, axis=1)
        _sum += batch_sum.sum(axis=0)


    _total = MAX_LENGTH * len(training_sequence)

    return np.sqrt(_sum / (_total - 1))


def get_training_mean_and_std(training_sequence):
    mean = get_training_mean(training_sequence)

    std = get_training_std(training_sequence, mean)

    return mean, std


if __name__ == '__main__':
    from data_access import get_train_dev_filenames, CacheBatchGenerator
    train_files, dev_files = get_train_dev_filenames()
    training_batch_generator = CacheBatchGenerator(
        train_files, batch_size=6, name='training_gen'
    )

    print(get_training_mean_and_std(training_batch_generator))
