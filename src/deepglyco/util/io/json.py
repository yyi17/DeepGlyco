__all__ = ['save_json', 'load_json']


import json


def save_json(data, file, **kwargs):
    with open(file, "w") as f:
        json.dump(data, f, **kwargs)


def load_json(file, **kwargs):
    with open(file, "r") as f:
        return json.load(f, **kwargs)
