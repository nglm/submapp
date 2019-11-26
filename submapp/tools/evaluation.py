
from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, TypeVar
import os

def compute_error(
        values: np.ndarray, true_values: np.ndarray,
        type_error="rmse",
        show=False,
        ) -> np.ndarray:
    """Compute error with different metrics
    
    :param values: Values estimated
    :type values: np.ndarray[float], shape=(_, p)
    :param true_values: True values
    :type true_values: np.ndarray[float], shape=(_, p)
    :param type_error: 
        Metric. Metrics available:

            - Root Mean Squared Error: "rmse"
            - Mean Squared Error: "mse"
            - Mean Absolute Error: "mae"
            - Max error: "max"
            
        , defaults to "rmse"
    :type type_error: str, optional
    :return: 
        Error computed with the chosen metric
    :rtype: np.ndarray[float], shape=(p)
    """

    if len(values.shape) < 2:
        values = np.array([values])
        true_values = np.array([true_values]) 

    # Finds vectors in which at least one value is missing
    missing_data = np.logical_or(np.isnan(true_values),
                                    np.isnan(values))
    missing_vectors = np.any(missing_data, axis=1)
    # Don't take into account these vectors
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
    elif type_error == "rmse":
        err = np.sqrt(mean_squared_error(true_values, values, multioutput='raw_values'))
    else:
        # Else RMSE
        err = np.sqrt(mean_squared_error(true_values, values, multioutput='raw_values'))

    if show:
        plot_error(err, x_label="Depth",y_label=type_error)
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

def plot_error(
        err,
        legend=None, 
        x_label=None,
        y_label=None,
        zmin=None,
        zmax=None,
        save=False,
        filename=None,
        path=None,
        ) :   
    T_year = len(err)
    if x_label is None:
        x_label= 'Time [arbitrary indices]'
    if y_label is None:
        y_label= 'Error'
    if zmin is None:
        zmin = np.nanmin(err)
    if zmax is None:
        zmax = np.nanmax(err)
    
    plt.figure()
    x = np.linspace(0,T_year,T_year)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(legend)
    plt.axis([0, T_year-1, zmin, zmax])
    plt.plot(x, err)
    if save:
        if path is None:
            path = "../figs/Err/"
        os.makedirs(path, exist_ok=True)
        if filename is None:
            filename = "err"
        plt.savefig(path + filename +'.png',format='png', dpi=1000)
    plt.show()
    plt.close()