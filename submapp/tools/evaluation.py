
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from sklearn.model_selection import train_test_split
import numpy as np
from typing import List, Tuple, TypeVar

def compute_error(
        values: np.ndarray, true_values: np.ndarray,
        type_error="rmse"):

    if len(values.shape) < 2:
        values = np.array([values])
        true_values = np.array([true_values]) 

    missing_data = np.logical_or(np.isnan(true_values),
                                    np.isnan(values))
    missing_vectors = np.any(missing_data, axis=1)   
    values = values[np.logical_not(missing_vectors)]
    true_values = true_values[np.logical_not(missing_vectors)]

    if type_error == "mse":
        err = mean_squared_error(true_values, values, multioutput='raw_values')
    elif type_error == "mae":
        err = mean_absolute_error(true_values, values, multioutput='raw_values')
    elif type_error == "max":
        p = len(values[0])
        err = np.zeros(p)
        for k in range(p):
            err[k] = max_error(true_values[:,k], values[:,k])
    elif type_error == "mse":
        err = np.sqrt(mean_squared_error(true_values, values, multioutput='raw_values'))
    else:
        err = np.sqrt(mean_squared_error(true_values, values, multioutput='raw_values'))
    return err


def split_indices(
        indices: np.ndarray,
        train_size=12,
        repeat: int = 1,
        shuffle: bool = False,
        ) -> List[Tuple[int]]:
    l = repeat*[()]
    if repeat>1 and not shuffle:
        shuffle = True
    for k in range(repeat):
        ind_train, ind_test = train_test_split(indices, train_size=train_size, shuffle=shuffle)
        l[k] = (ind_train, ind_test)
    return l
