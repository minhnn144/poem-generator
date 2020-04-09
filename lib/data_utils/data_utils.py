import pickle
import os

root = os.getenv("PYTHONPATH")


def pickle_data(path, data):
    with open(root + path, 'wb') as f:
        pickle.dump(data, f)


def unpickle_file(path, full=False):
    if full:
        with open(path, 'rb') as f:
            return pickle.load(f)
    else:
        with open(root + path, 'rb') as f:
            return pickle.load(f)
