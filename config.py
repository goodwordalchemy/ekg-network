import sys
import argparse
import yaml

DEFUALT_CONFIG_FILENAME = 'config_files/example_inception_config.yaml'

def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--config', default=DEFUALT_CONFIG_FILENAME)

    if not len(sys.argv) > 1:
        args = parser.parse_args([])
    else:
        args = parser.parse_args()

    return args


def get_config():
    args = get_args()

    config_filename = args.config

    with open(config_filename, 'r') as f:
        config = yaml.load(f)

    return config
