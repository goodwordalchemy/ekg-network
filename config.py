import yaml

CONFIG_FILENAME = 'config.yaml'

def get_config():
    with open(CONFIG_FILENAME, 'r') as f:
        config = yaml.load(f)

    return config
