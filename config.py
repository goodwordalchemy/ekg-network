import yaml

import argparse

def _get_args():
    parser = argparse.ArgumentParser(description='Run a hyperparameter search')
    parser.add_argument('config')

    args = parser.parse_args()

    return args


def get_config():
    args = _get_args()

    config_filename = args.config

    with open(config_filename, 'r') as f:
        config = yaml.load(f)

    return config
