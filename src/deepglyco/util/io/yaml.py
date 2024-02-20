__all__ = ['save_yaml', 'load_yaml']


import yaml


def save_yaml(data, file, **kwargs):
    with open(file, "w") as f:
        yaml.dump(data, f, **kwargs)


def load_yaml(file, **kwargs):
    with open(file, "r") as f:
        return yaml.safe_load(f, **kwargs)
