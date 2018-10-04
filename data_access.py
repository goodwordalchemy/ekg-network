def downsample_mis(all_filenames, target_num=1000):
    with open(MI_DATA_FILENAME, 'r') as f:
        mi_filenames = f.read().split('\n')

    num_to_select = len(mi_filenames) - target_num
    all_filenames = set(all_filenames) - set(mi_filenames[:num_to_select])

    return list(all_filenames)


def remove_test_data_filenames(all_filenames):
    with open(TEST_DATA_FILENAME, 'r') as f:
        test_filenames = f.read().split('\n')

    return list(set(all_filenames) - set(test_filenames))


def get_train_dev_filenames(fraction=0.15):
    ptbdb_filenames = os.listdir(DATA_DIRECTORY)

    ptbdb_filenames = downsample_mis(ptbdb_filenames)
    ptbdb_filenames = remove_test_data_filenames(ptbdb_filenames)

    shuffle(ptbdb_filenames)

    if DATA_SUBSET_FRACTION < 1:
        ptbdb_filenames = ptbdb_filenames[:int(DATA_SUBSET_FRACTION * len(ptbdb_filenames))]
        print('Only using {}% of data: {} samples'.format(DATA_SUBSET_FRACTION * 100, len(ptbdb_filenames)))
    n_holdouts = int(fraction * len(ptbdb_filenames))

    train = ptbdb_filenames[:-n_holdouts]
    dev = ptbdb_filenames[-n_holdouts:]

    return train, dev


def load_data_files_to_array(filenames):
    batch = []

    for i, filename in enumerate(filenames):
        with open(DATA_DIRECTORY + '/' + filename, 'rb') as f:
            data = pickle.load(f)

        batch.append(data)

    return batch


class CacheBatchGenerator(Sequence):

    def __init__(self, filenames, batch_size):
        self.filenames = filenames
        self.batch_size = min(batch_size, len(filenames))

    def __len__(self):
            return int(np.ceil(len(self.filenames) / float(self.batch_size)))

    def __getitem__(self, idx):
        print('CacheBatchGenerator is getting idx {} of {}'.format(idx, len(self)))

        batch_filenames = self.filenames[idx * self.batch_size:(idx + 1) * self.batch_size]

        batch = load_data_files_to_array(batch_filenames)

        batch_x, batch_y = zip(*batch)

        batch_x = pad_sequences(
            batch_x, dtype=batch_x[0].dtype, maxlen=MAX_LENGTH
        )

        batch_y = [1 if r == 'Myocardial infarction' else 0 for r in batch_y]
        batch_y = np.array(batch_y).reshape(-1, 1)

        return batch_x, batch_y

