__all__ = ['save_pickle', 'load_pickle']


import pickle


def save_pickle(data, file, **kwargs):   
    with open(file, "wb") as f:
        pickle.dump(data, f, **kwargs)


def load_pickle(file, **kwargs):
    with open(file, "rb") as f:
        return pickle.load(f, **kwargs)
