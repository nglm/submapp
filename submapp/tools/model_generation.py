from sklearn.metrics import mean_squared_error, mean_absolute_error, max_error
import numpy as np
from sklearn.decomposition import PCA
from typing import List, Tuple, TypeVar

def generate_som_shape(
        n_start: int,
        n_end: int,
        ratio: float,
        step: int = 1,
        ) -> List[Tuple[int]]:
    nb_iterations = max(int((n_end - n_start)/step),1)
    l = [()]*nb_iterations 
    for it in range(nb_iterations):
        n = n_start + it*step
        m = int(n*ratio)
        l[it] = (n,m)
    return l