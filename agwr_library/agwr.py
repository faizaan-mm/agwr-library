# importing the libraries
import numpy as np
import pickle
import os
import os
import pickle
from .utils import R2
import numpy as np
import time
import math
import multiprocessing
from .SpatialModules import gwr_module, mgwr_module, smgwr_module
from .MLModules import random_forrest, neural_network, xgb
from .ModularFramework import ModularFramework
import pickle


def read_dataset(path):
    path = path + "/" if path[-1] != "/" else path
    with open(path + 'training_idx.data', 'rb') as filehandle:
        training_idx = pickle.load(filehandle)
    with open(path + 'test_idx.data', 'rb') as filehandle:
        test_idx = pickle.load(filehandle)
    with open(path + 'x.data', 'rb') as filehandle:
        x = pickle.load(filehandle)
    with open(path + 'y.data', 'rb') as filehandle:
        y = pickle.load(filehandle)
    with open(path + 'coords.data', 'rb') as filehandle:
        coords = pickle.load(filehandle)

    x = x.astype(float)
    y = y.astype(float)
    coords = coords.astype(float)

    X_training, X_test = x[training_idx, :], x[test_idx, :]
    y_training, y_test = y[training_idx, :], y[test_idx, :]
    coords_training, coords_test = coords[training_idx], coords[test_idx]

    return X_training, coords_training, y_training, X_test, coords_test, y_test


# create modules and their setting
def module_selection(spatial, ml):
    
    if spatial == "GWR":
        spatial_module = gwr_module
    elif spatial == "MGWR":
        spatial_module = mgwr_module
    elif spatial == "SMGWR":
        spatial_module = smgwr_module
    else:
        spatial_module = None

    if ml == "RF":
        ml_module = random_forrest
    elif ml == "NN":
        ml_module = neural_network
    elif ml == "XGB":
        ml_module = xgb
    else:
        ml_module = None

    return spatial_module, ml_module


def train(dataset:str, spatial_module:str, ml_module:str, config={}):

    model_file_name = f'trained_model_{dataset.split("/")[-1]}-{spatial_module}-{ml_module}.pickle'

    X_training, coords_training, y_training, _x_test, _coords_test, _y_test = read_dataset(dataset)

    spatial_module, ml_module = module_selection(spatial_module, ml_module)

    agwr_model = ModularFramework(spatial_module, ml_module, config)

    # train model
    agwr_model.train(X_training, coords_training, y_training)

    #save model
    pickle_file = open(model_file_name, 'wb')
    pickle.dump(agwr_model, pickle_file)
    pickle_file.close()


def predict(model_pickle: str, x_test: list, coords_test: list):
    pickle_file = open(model_pickle, 'rb')
    loaded_model = pickle.load(pickle_file)
    x_test = np.array([np.array(x_test)])
    coords_test = np.array([np.array(coords_test)])
    pred = loaded_model.predict(x_test, coords_test)
    print(pred)

{"process_count": 8, "divide_method": "equalCount", "divide_sections": [1, 2, 3], "pipelined": True}