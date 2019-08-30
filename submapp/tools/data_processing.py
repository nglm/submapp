from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
import numpy as np
from sklearn.decomposition import PCA

def remove_random_data(data, prob, ignore_nan=False):
    (T, p) = data.shape
    keep_data = np.reshape(np.random.binomial(n=1, p=prob, size=T), (T,1))
    if ignore_nan:
        empty = True
        for t in range(T):
            if empty and keep_data[t]:
                new_data = np.copy([data[t]])
                empty = False
            else:
                if keep_data[t]:
                    new_data = np.append(new_data, [data[t]] , axis=0)
    np_nan = np.reshape(np.array(p*[np.nan]), (1,p))
    new_data = np.where(keep_data, data, np_nan)
    return new_data

def remove_nan(data, entire_vector=True):
    (T, _) = data.shape
    empty = True
    for t in range(T):
        # entire_vector boolean: do we remove the entire vector if 
        # at least one element is missing?
        if ((entire_vector and not np.all(np.isnan(data[t])))
            or (not entire_vector and not np.any(np.isnan(data[t]))) ):
            if empty:
                new_data = np.copy([data[t]])
                empty=False
            else: 
                new_data = np.append(new_data, [data[t]] , axis=0)
    return new_data

def standardize(data, data_mean=None,data_stdev=None):
    if data_mean is None:
        data_mean = np.nanmean(data)
        if data_stdev is None:
            data_stdev = np.nanstd(data)
            data_norm = (data-data_mean)/data_stdev
            return(data_norm,data_mean,data_stdev)
        else:
            return("ERROR: one missing argument: mean")
    else:
        if data_stdev is None:
            return("ERROR: one missing argument: stdev")
        else:
            return((data-data_mean)/data_stdev)

def destandardize(data_norm,data_mean,data_stdev):
    return(data_norm*data_stdev+data_mean)

def pca_features(training_data, is_standardized=True):
    training_data_available = remove_nan(training_data)
    pca = PCA(n_components=2)
    pca.fit(training_data_available)
    variance = pca.explained_variance_ratio_
    ratio = variance[0]/variance[1]
    cumulative_variance = sum(variance)
    return ratio, cumulative_variance

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
