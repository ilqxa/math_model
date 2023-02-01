import numpy as np


def check_pairs(obs: np.ndarray):
    if obs.ndim != 2: raise Exception('Data must have 2 dimensions')
    if obs.shape[1] != 2: raise Exception('Point must have 2 coordinates')
    return obs

def check_unique_x(obs: np.ndarray):
    if np.unique(obs.T[0], return_counts=True)[-1].max() > 1:
        raise Exception('Each X value must be unique')
    return obs

def check_float_dtype(obs: np.ndarray):
    if obs.dtype != np.float64: raise TypeError('Points must be defined by float')
    return obs