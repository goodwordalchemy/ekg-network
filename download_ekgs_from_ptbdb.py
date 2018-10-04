from io import BytesIO
import os
import pickle
import requests

from gs_utils import get_bucket

try:
    import wfdb
except RuntimeError:
    print('WARNING: Caching functionality only works within an '
          'ipython notebook because "python is not installed as an app.  '
          'To see error message and learn more about the problem, delete '
          'this try/except block.  Also note that currently the wfdb directory from the wfdb-python '
          'repository needs to be downloaded in the same directory as the notebook '
          'in order for the caching function to work')

except ModuleNotFoundError:
    print('wfdb model is not installed.  You will not be able to build a local cache from the remote ptbdb cache without it')

CACHE_PATH = 'data/cached_records'
PTBDB_PATH = 'ptbdb'

def get_record_path_list():
    r = requests.get('https://www.physionet.org/physiobank/database/ptbdb/RECORDS')

    records_str = r.text

    records_list = records_str.split()

    return records_list


def get_raw_record(record_path):
    patient, sample_name = record_path.split('/')

    pb_dir = os.path.join(PTBDB_PATH, patient)

    record = wfdb.rdrecord(sample_name, pb_dir=pb_dir)

    return record


def extract_signal_and_diagnosis(raw_record):
    signal = raw_record.p_signal

    diagnosis = raw_record.comments[4].split(': ')[1]

    return signal, diagnosis

def get_record(record_path):
    raw_record = get_raw_record(record_path)

    signal, diagnosis = extract_signal_and_diagnosis(raw_record)

    return signal, diagnosis


def _get_record_cache_path(record_name):
    record_name = record_name.replace('/', '--')

    return os.path.join(CACHE_PATH, record_name)


def _load_records_local():
    cached_records_list = os.listdir(CACHE_PATH)
    print('loading {} cached ptbdb records'.format(len(cached_records_list)))

    records = {}
    for record_name in cached_records_list:
        record_path = _get_record_cache_path(record_name)

        with open(record_path, 'rb') as f:
            records[record_name] = pickle.load(f)

    return records


def _load_records_from_google_storage():
    bucket = get_bucket()

    blobs = list(bucket.list_blobs())

    records = {}

    for i, blob in enumerate(blobs):
        print('{} of {} blobs downloading'.format(i + 1, len(blobs)))

        with BytesIO() as b:
            blob.download_to_file(b)
            b.seek(0)

            records[blob.name] = pickle.load(b)

    return records


def load_records(local=False):
    if local:
        return _load_records_local()

    return _load_records_from_google_storage()


def cache_records(reset_cache=False):
    """
    caches records from ptbdb database of labeled ECG data.

    Parameters:
        - reset_cache - deletes the cache before loading records so that all records are loaded from database.
    """
    if reset_cache and os.path.exists(CACHE_PATH):
        print('resetting cache')
        for record_name in get_record_path_list():
            record_cache_path = _get_record_cache_path(record_name)

            if os.path.exists(record_cache_path):
                os.remove(record_cache_path)

    print('loading records from online database')

    record_path_list = get_record_path_list()

    records = load_records()

    record_keys = list(records.keys())
    record_keys = [r.replace('--','/') for r in record_keys]

    record_path_list = list(set(record_path_list) - set(record_keys))

    for i, record_path in enumerate(record_path_list):
        record = get_record(record_path)
        signal, diagnosis = record

        print('reading record {} of {}: {} {}'.format(
            i, len(record_path_list), record_path, diagnosis)
        )


        with open(_get_record_cache_path(record_path), 'wb') as f:
            pickle.dump(record, f)


if __name__ == '__main__':
    cache_records(reset_cache=False)
