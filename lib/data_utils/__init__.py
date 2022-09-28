import pickle
import os
import configparser

root = os.getenv("PYTHONPATH")

SPECIAL_CONTROLS = ('<SOS>', '<EOS>', '<PAD>', '<BRK>')

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

def save_config(path, cfg):
    config = configparser.ConfigParser()
    config['DEFAULT'] = cfg
    with open(root + path, 'w') as f:
        config.write(f)


def load_config(path):
    config = configparser.ConfigParser()
    config.read(root + path)
    return config