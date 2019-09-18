""" ``som`` module implementing ``Som`` class
"""


import tensorflow as tf
import numpy as np
from copy import copy as built_in_copy
from time import time
import matplotlib.pyplot as plt
import seaborn as sns
from math import sqrt
from numpy import linalg as LA
from statistics import mean, variance, stdev
import pickle
import os
import random
from typing import List, Tuple, TypeVar
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from .tools.data_processing import remove_nan

S = TypeVar('S', bound='Som')


def save(som, filename: str = None, path: str = "") -> None:
    """Save the SOM object
    
    :param filename: 
        filename of the saved SOM. defaults to None, in this case
        the filename is the name of the SOM (eg. "MySom")
    :type filename: str, optional
    :param path:
        Relative path. defaults to "", in this case the SOM
        object is save in the current path
    :type path: str, optional
    :rtype: None

    .. seealso:: :doc:`load <submapp.som.load>`
    """

    som._Som__clear_graph()
    if path:
        os.makedirs(path, exist_ok=True)
    if filename is None:
        filename = som.name
    with open(path + filename, "wb") as f:
        pickle.dump(som, f)

def load(filename: str, path: str = "") -> S :
    """Load a saved SOM
    
    :param filename: filename of the saved SOM
    :type filename: str
    :param path:
        Relative path. defaults to "", in this case the SOM
        object is loaded in the current path
    :type path: str, optional
    :return: The loaded SOM stored at "path/filename"
    :rtype: Som

    .. seealso:: :doc:`save <submapp.som.save>`
    """

    with open(path + filename, "rb") as f:
        mySom = pickle.load(f)
    return mySom

def copy(som) -> S:
    """Return a deep copy of the SOM
    
    :rtype: Som
    """

    new_som = built_in_copy(som)
    new_som._Som__clear_graph()
    return new_som


class Som:
    """Self-Organizing Map (or Kohonen Map)

    Here is a quick summary of the purpose of every attribute and method.
    (Note that some of them may appear in several rows)

    .. csv-table::
       :stub-columns: 1
       :widths: 20, 80
       :delim: ;

       "Shape of the SOM";\
       :ref:`n <submapp.som.Som.n>`,\
       :ref:`m <submapp.som.Som.m>`,\
       :ref:`nb_class <submapp.som.Som.nb_class>`,\
       :ref:`p <submapp.som.Som.p>`,\
       :ref:`shape <submapp.som.Som.shape>`,\
       :ref:`location_classes <submapp.som.Som.location_classes>`,\
       :ref:`distance_matrix <submapp.som.Som.distance_matrix>`
       "Data processing"; \
       :ref:`data_mean <submapp.som.Som.data_mean>`,\
       :ref:`data_stdev <submapp.som.Som.data_stdev>`,\
       :ref:`standardize <submapp.som.Som.standardize>`,\
       :ref:`destandardize <submapp.som.Som.destandardize>`
       "Training"; \
       :ref:`train <submapp.som.Som.train>`,\
       :ref:`nb_training_iterations <submapp.som.Som.nb_training_iterations>`,\
       :ref:`relerr_train <submapp.som.Som.relerr_train>`,\
       :ref:`is_trained <submapp.som.Som.is_trained>`
       "Mapping"; \
       :ref:`map <submapp.som.Som.map>`,\
       :ref:`nb_inputs_mapped <submapp.som.Som.nb_inputs_mapped>`,\
       :ref:`class2weights <submapp.som.Som.class2weights>`,\
       :ref:`clear_map_info <submapp.som.Som.clear_map_info>`
       "Testing";\
       :ref:`weights <submapp.som.Som.weights>`,\
       :ref:`occ_bmu_map <submapp.som.Som.occ_bmu_map>`,\
       :ref:`weights_features <submapp.som.Som.weights_features>`,\
       :ref:`time_info <submapp.som.Som.time_info>`,\
       :ref:`transition <submapp.som.Som.transition>`,\
       :ref:`distance_transitions <submapp.som.Som.distance_transitions>`,\
       :ref:`print_heatmap <submapp.som.Som.print_heatmap>`,\
       :ref:`relerr_test <submapp.som.Som.relerr_test>`

    :var str name: 
        The *Name* of the Som object. This name has no impact on
        the variable name, however it can be usefull when it comes
        to saving the Som object and its related figures, defaults 
        to "MySom"
    """
    # this function is only meant to init the SOM, not necesseraly the graph,
    # ops, etc.
    def __init__(
            self, 
            n: int, 
            m: int, 
            p: int = 1, 
            distance_matrix: np.ndarray = None, 
            weights: np.ndarray = None,
            name: str = "MySom", 
            data_mean: np.ndarray = np.array([0]), 
            data_stdev: np.ndarray = np.array([1]),
            ):
        """Class constructor

        Create a Som object and initialize its basic properties 
        (eg. shape, weights). Please note that it won't be allowed to
        change some attributes (actually all but name) once this
        function is called! If the shape (for instance) doesn't seem
        good after training then create another Som object with a
        different shape.
        
        :param n: 
            The number of rows of the SOM 
            (can't be changed after initialization)
        :type n: int
        :param m: 
            The number of columns of the SOM 
            (can't be changed after initialization)
        :type m: int
        :param p: 
            The size of input vectors (and therefore of the weights
            too). Should be consistent. Can't be changed after
            initialization, defaults to ``1``
        :type p: int, optional
        :param distance_matrix: 
            Matrix of relative distances between every class. 
            Required shape: ``(n*m, n*m)``
            This matrix defines the "shape" of the SOM (eg. hexagonal,
            rectangular, etc.), defaults to None, in this case it
            is a rectangular map (euclidean distance between classes)
            (can't be changed after initialization)
        :type distance_matrix: np.ndarray, optional
        :param weights:
            Initial weights. Required shape: ``(n*m, p)``
            Since the weights are meant to represent
            the data, initializing them with a PCA or with any method
            that could make them more representative of the data given
            our a priori knowlegde (basically the training data set) 
            should be a good option, defaults to None, in this case a
            random initialization is done (uniform distribution).
            (can't be changed after initialization, only the training
            method will be allowed to change their value)
        :type weights: np.ndarray, optional
        :param name: 
            The *Name* of the Som object. This name has no impact on
            the variable name, however it can be usefull when it comes
            to saving the Som object and its related figures, defaults 
            to "MySom"
        :type name: str, optional
        :param data_mean:
            Vector containing means. Required shape: ``(1, )`` or ``(p, )``
            Usually all data are standardized (especially when
            different types of data are used in the same vector). In
            order to standardized, destandardized, compute errors
            effectively the initial (i.e. of the training
            data) mean and standard deviation should be stored. Here
            is the mean of the training data, defaults to [0] (either 
            data are supposed already standardized or no standardization is
            required/wanted in this particular case)
            (can't be changed after initialization)
        :type data_mean: np.ndarray, optional
        :param data_stdev: 
            Same as data_mean but with the standard deviation, 
            (can't be changed after initialization)
            defaults to ``[1]``
        :type data_stdev: np.ndarray, optional
        """

        self.name = name
        self.__graph = None  # Tensorflow graph

        # Usually data are standardized (especially when different types
        # of data are used in the same vector).  In order to 
        # standardized, destandardized, compute errors effectively the
        # initial mean and standard deviation should be stored.
        self.__mean = np.copy(data_mean)
        self.__stdev = np.copy(data_stdev)

        # Shape of the SOM
        # (n,m) size of the map
        # p input dim
        self.__n = abs(int(n))  # number of rows
        self.__m = abs(int(m))  # number of columns
        self.__p = abs(int(p))  # size of input vectors (and weights)

        # Matrix of indexes (i,j) of the kth class within the SOM
        # location_classes[k] = [i, j]
        # k in [|0, n*m - 1|]
        # i in [|0, n-1|]
        # j in [|0, m-1|]
        self.__location_classes = np.zeros((self.nb_class, 2), dtype=int)
        for i in range(self.__n):
            for j in range(self.__m):
                self.__location_classes[i*self.__m + j, 0] = i  # row class k
                self.__location_classes[i*self.__m + j, 1] = j  # and column

        # Matrix of relative distances between classes.
        # This matrix defines the "shape" of the SOM
        # (eg. hexagonal, rectangular, etc.)
        if distance_matrix is None:
            # Default value is a rectangular map (euclidean distance)
            self.__distance_matrix = np.zeros(
                (self.nb_class, self.nb_class), dtype=float
            )
            for i in range(self.nb_class):
                [i1, j1] = self.__location_classes[i, :]
                for j in range(self.nb_class):
                    [i2, j2] = self.__location_classes[j, :]
                    # self.__distance_matrix[i,j] = abs(i1-i2) + abs(j1-j2)
                    self.__distance_matrix[i, j] = sqrt(
                        (i1 - i2)**2 + (j1 - j2)**2
                    )
        else:
            self.__distance_matrix = np.abs(distance_matrix)

        # weights[k] = referent vector of the class k
        # usually weights should be initialized according to the
        # training data. (eg. use a PCA). Otherwise it can be
        # randomly initialized
        if weights is None:
            self.__weights = np.random.uniform(
                low=-2, high=2, size=(self.nb_class, self.__p)
            )
        else:
            self.__weights = np.copy(weights)

        # The following attributes concern only the training phase!
        self.__is_trained = False
        self.__nb_training_iterations = 0
        # Relative error between the input vectors and the referent
        # vectors of the corresponding BMU during the training phase
        self.__relerr_train = np.copy([])
        # Hyperparameters of the SOM:
        # a0, aT: initial and final learning rate
        # s0, sT: initial and final radius of the neighborhood function
        a0 = 0.9
        aT = 0.1
        if self.__n == 1 or self.__m == 1:
            s0 = min(max(self.__n, self.__m)/10.0, 4.0)
        else:
            s0 = max(max(self.__n, self.__m)/10.0, 1.0)
        sT = 0.5
        self.param = (a0, aT, s0, sT)
        self.__current_iteration = 0

        # The following attributes concern only the mapping phase!
        # (From nb_inputs_mapped to relerr_test)
        self.__nb_inputs_mapped = 0
        # occ_bmu_map[k] = occurence class k as BMU
        self.__occ_bmu_map = np.zeros(self.nb_class, dtype=int)
        # t_as_bmu[k] = list of instants t for which k was the BMU
        self.__t_as_bmu = [[] for i in range(self.nb_class)]
        # transition[k1, k2] = occurence of the transition k1 -> k2
        # (k1 BMU instant t-1 and k2 BMU at instant t)
        self.__transition = np.zeros(
            (self.nb_class, self.nb_class), dtype=int)
        # distance_transitions[t] = relative distance between k1 and k2
        # where k1 was BMU at instant t-1 and k2 BMU at instant t
        self.__dist_transition = np.copy([])
        # Relative error between the input vectors and the referent
        # vectors of the corresponding BMU.
        self.__relerr_test = np.copy([])
        # Transition missed in case of missing data
        self.__nb_missing_transitions = 0 

    def __init_graph(
            self, param: Tuple[float, float, float, float] = None) -> None:
        """[summary]
        
        :param param: [description], defaults to None
        :type param: Tuple[float, float, float, float], optional
        :return: [description]
        :rtype: None
        """

        print("Graph initialization")
        self.__current_iteration = 0
        self.__graph = tf.Graph()
        with self.__graph.as_default():

            weights = tf.placeholder(
                tf.float32, (self.nb_class, self.__p), name="Weights")
            vect_input = tf.placeholder(
                tf.float32, (1, self.__p), name="Input")
            # distance_to_bmu[k] = distance_matrix[bmu, k]
            distance_to_bmu = tf.placeholder(
                tf.float32, (self.nb_class, 1), name="Distance_to_BMU")
            curr_it = tf.placeholder(tf.float32, name="Current_Iteration")
            T_train = tf.placeholder(tf.float32, name="T_train")

            # Find the BMU (class whose referent vector minimizes the
            # euclidean distance to the input)
            tf.argmin(
                tf.sqrt(tf.reduce_sum((weights - vect_input)**2, axis=1)),
                name="BMU_op")
            # tf.argmin(tf.reduce_sum(abs(weights - vect_input),axis=1)
            # + tf.sqrt(tf.reduce_sum((weights - vect_input)**2,axis=1)), name="BMU_Op")
            # tf.argmin(tf.reduce_sum(abs(weights - vect_input),axis=1), name="BMU_Op")

            if param is None:
                (a0, aT, s0, sT) = self.param
            else:
                if len(param) != 4:
                    return print(
                        "No parameters were given. \
                        Please specify param=(a0,aT,s0,sT)"
                    )
                else:
                    self.param = param
                    (a0, aT, s0, sT) = param

            # Radius sigma
            # sigma = tf.multiply(s0, tf.exp(-curr_it/ep), name="Sigma_Op")
            sigma = s0 - curr_it*(s0 - sT)/T_train

            # Learning rate r
            r = a0 - curr_it*(a0 - aT)/T_train

            # Neighborhood function h
            h = tf.exp(-(distance_to_bmu) / sigma)

            # training operation
            tf.add(weights, r*h*(vect_input - weights), name="Training_Op")

    def sample_weights_initialization(
            self, 
            data_train: np.ndarray,
            ) -> None:
        """Initialize weights with a sample of the training data

        Allow for NaN values. As this method (re-)initializes
        the weights it also calls the method
        :ref:`clear_map_info <submapp.som.Som.clear_map_info>`
        and re-initializes 
        :ref:`is_trained <submapp.som.Som.is_trained>`
        to ``False``
        
        :param data_train: Training data 
        :type data_train: np.ndarray[float], shape = (_, p)
        :rtype: None

        .. seealso:: 
            - :ref:`random_weights_initialization <submapp.som.Som.random_weights_initialization>`
            - :ref:`pca_weights_initialization <submapp.som.Som.pca_weights_initialization>`
            - :ref:`zeros_weights_initialization <submapp.som.Som.zeros_weights_initialization>`

        """

        # Ignore vectors containing NaN values
        data_train_available = remove_nan(data_train)
        T = len(data_train_available)
        if (T < self.nb_class):
            print("Not enough data: only ", T, " vectors available")
        else:
            ind = random.sample(range(T), self.nb_class)
            self.__weights = data_train_available[ind]

            # Re-initializes the SOM properties
            self.clear_map_info()
            self.is_trained = False

    def pca_weights_initialization(
            self, 
            data_train: np.ndarray,
            ) -> None:
        """Initialize weights with a PCA of the training data

        Allow for NaN values. As this method (re-)initializes
        the weights it also calls the method
        :ref:`clear_map_info <submapp.som.Som.clear_map_info>`
        and re-initializes 
        :ref:`is_trained <submapp.som.Som.is_trained>`
        to ``False``
        
        :param data_train: Training data 
        :type data_train: np.ndarray[float], shape = (_, p)
        :rtype: None

        .. seealso:: 
            - :ref:`sample_weights_initialization <submapp.som.Som.sample_weights_initialization>`
            - :ref:`random_weights_initialization <submapp.som.Som.random_weights_initialization>`
            - :ref:`zeros_weights_initialization <submapp.som.Som.zeros_weights_initialization>`

        """

        # Ignore vectors containing NaN values
        data_train_available = remove_nan(data_train)
        # Projection onto the 2 first principal components
        pca = PCA(n_components=2, whiten=True)
        proj_pca = pca.fit_transform(data_train_available)
        # Group theses points projected into only nb_class clusters 
        kmeans = KMeans(n_clusters=self.nb_class).fit(proj_pca)
        cluster_centers = kmeans.cluster_centers_
        # sort resulting points with a heavier weight on the 1st component
        ind_ax1 = np.argsort(cluster_centers[:,0])
        sorted_centers_ax1 = cluster_centers[ind_ax1]
        ind = np.zeros_like(ind_ax1)
        for k in range(int(self.nb_class/self.__n)):
            start = k*self.__n
            end = min((k+1)*self.__n, self.nb_class)
            tmp_ax1 = ind_ax1[start:end]
            tmp_ax2 = np.argsort(sorted_centers_ax1[start:end,1])
            ind[start:end] = tmp_ax1[tmp_ax2]
        sorted_centers = cluster_centers[ind]
        inverse_pca = pca.inverse_transform(sorted_centers)
        inverse_pca = np.reshape(inverse_pca,self.shape,order='F')
        inverse_pca = np.reshape(inverse_pca,(self.nb_class ,self.__p),order='C')
        self.__weights = inverse_pca

        # Re-initializes the SOM properties
        self.clear_map_info()
        self.is_trained = False

    def random_weights_initialization(
            self, 
            distribution: str = None,

            ) -> None:
        """Initialize weights randomly 

        As this method (re-)initializes
        the weights it also calls the method
        :ref:`clear_map_info <submapp.som.Som.clear_map_info>`
        and re-initializes 
        :ref:`is_trained <submapp.som.Som.is_trained>`
        to ``False``random_weights_initialization
        
        :param distribution: 
            distribution "gaussian" or "uniform". Default to
            None, in this case the distribution is uniform
        :type distribution: string, optional
        :rtype: None

        .. seealso:: 
            - :ref:`sample_weights_initialization <submapp.som.Som.sample_weights_initialization>`
            - :ref:`pca_weights_initialization <submapp.som.Som.pca_weights_initialization>`
            - :ref:`zeros_weights_initialization <submapp.som.Som.zeros_weights_initialization>`

        """
        if distribution == "gaussian":
            self.__weights = np.random.standard_normal(
                size=(self.nb_class, self.__p)
            )
        else:
            self.__weights = np.random.uniform(
                low=-2, high=2, size=(self.nb_class, self.__p)
            )

        # Re-initializes the SOM properties
        self.clear_map_info()
        self.is_trained = False

    def zeros_weights_initialization(
            self, 
            ) -> None:
        """Initialize all weights to zeros vectors

        As this method (re-)initializes
        the weights it also calls the method
        :ref:`clear_map_info <submapp.som.Som.clear_map_info>`
        and re-initializes 
        :ref:`is_trained <submapp.som.Som.is_trained>`
        to ``False``
        
        :rtype: None

        .. seealso:: 
            - :ref:`sample_weights_initialization <submapp.som.Som.sample_weights_initialization>`
            - :ref:`pca_weights_initialization <submapp.som.Som.pca_weights_initialization>`
            - :ref:`random_weights_initialization <submapp.som.Som.random_weights_initialization>`

        """

        self.__weights = np.zeros((self.nb_class,self.p))

        # Re-initializes the SOM properties
        self.clear_map_info()
        self.is_trained = False

    def train(
            self, 
            data_train: np.ndarray, 
            T_train: int = None, 
            batch_size: int = None, 
            param: Tuple[float, float, float, float] = None,
            missing_data: bool = False,
            ) -> None:
        """Train the SOM (ie update the weights) provided training\
        data.

        Update weights - for all input 
        :math:`x[t] \\; t \\in [0, len(data_train)]` - as follows 

        .. math::
        
            weights[k] & = weights[k] 

            & + a(t)\
            \\times exp[\\frac{dist(weights[k],BMU(t))}{ s^{2}(t) } ]\
            \\times [weights[k]-x[t]]
        
        :param data_train: Training data 
        :type data_train: np.ndarray[float], shape = (_, p)
        :param T_train:
            Size of the entire training dataset. Once 
            ``T_train`` vectors
            have been used since the last time a new value of ``param``
            was given, the learning rate and the radius take their
            final value (i.e. ``aT`` and ``sT``). Defaults to None,
            In this case, ``T_train=len(data_train)``.
        :type T_train: int, optional
        :param batch_size: 
            NOT IMPLEMENTED YET.
            Defaults to None, in this case ``batch_size=1``
        :type batch_size: int, optional
        :param param:
            Hyperparameters of the Som for the training phase 
            ``(a0,aT,s0,sT)`` with:

                - ``a0``: initial learning rate (``0<a0<1``)
                - ``aT``: final learning rate (``0<aT<1``)
                - ``s0``: initial radius (``0<s0``)
                - ``sT``: final radius (``0<sT``)

            For each iteration (vector from ``data_train``) the learning
            rate and the radius decrease linearly until they reach
            their final value.

            Defaults to None, in this case the previous value of param
            is used. If this value has never been
            initialized then:

                - ``a0 = 0.9``
                - ``aT = 0.1``
                - 
                    + ``if (n=1 or m=1): s0 = min(max(n, m)/10.0, 4.0)``
                    + ``else: s0 = max(max(n, m)/10.0, 1.0)``
                - ``sT = 0.5``

        :type param: Tuple[float, float, float, float], optional
        :param missing_data: 
            indicates whether ``data_train`` contains missing data
            (represented by ``np.nan``),
            NOT IMPLEMENTED YET
            Defaults to False.

            .. warning:: 

                if ``missing_data=False`` and ``data_train`` does
                contain missing data then results will be 
                inconsistent

        :type missing_data: bool, optional 
        :rtype: None
        """

        if (self.__graph is None or 
           (param is not None and param != self.param)):
           # If graph uninitialized or param has changed:
            self.__init_graph(param)
        sess = tf.Session(graph=self.__graph)

        start = time()
        nb_inputs = len(data_train)
        if T_train is None:
            T_train = nb_inputs

        # Sorted instants 
        sorted_time = np.array(range(nb_inputs))
        # Used to select a new input vector randomly in inputs
        # This is done to prevent the beginning of the period from
        # learning faster than the end of the period
        shuffled_time = np.random.permutation(sorted_time)

        with self.__graph.as_default() as g:

            vect_input = g.get_tensor_by_name("Input" + ":0")
            weights = g.get_tensor_by_name("Weights" + ":0")
            data_train_size = g.get_tensor_by_name("T_train" + ":0")
            curr_it = g.get_tensor_by_name("Current_Iteration" + ":0")
            bmu_op = g.get_tensor_by_name("BMU_op" + ":0")
            training_op = g.get_tensor_by_name("Training_Op" + ":0")
            distance_to_bmu = g.get_tensor_by_name("Distance_to_BMU" + ":0")

            relerr = []  # list of relative error for each input vector

            # Main loop:
            # Train the model (weights) for each vector in data_train
            for t in shuffled_time:
                v_i = data_train[t]  # select input vector (from random instant!)
                v_i = np.reshape(v_i, (1, self.__p))

                # Find the BMU
                bmu_loc = int(sess.run(
                    bmu_op, 
                    feed_dict={vect_input: v_i, weights: self.__weights}
                    ))

                dist_to_bmu = self.__distance_matrix[bmu_loc]
                dist_to_bmu = np.reshape(dist_to_bmu, (self.nb_class, 1))

                # Update weights
                self.__weights = sess.run(training_op, feed_dict={
                    vect_input: v_i,
                    distance_to_bmu: dist_to_bmu,
                    curr_it: self.__current_iteration,
                    weights: self.__weights,
                    data_train_size: T_train})
                self.__current_iteration += 1
                self.__nb_training_iterations += 1

                # Compute relative error
                relerr = np.append(
                    relerr, 
                    self.__compute_relerr(self.__weights[bmu_loc],
                                          v_i, 
                                          is_standardized=True))

            self.__relerr_train = np.append(
                self.__relerr_train, np.nanmean(relerr))
            sess.close()
        end = time()
        print(nb_inputs, " vectors have trained the SOM ", round(end - start, 2), " sec")
        # print(" mean relative error: ", np.mean(relerr)) 
        

    def map(
            self, 
            data_map: np.ndarray,
            ) -> np.ndarray:
        """Map data onto the SOM 

        Associate each input vector to the class whose referent
        vector is the closest to the input vector in the euclidean
        sense.

        Handle missing data: if an entire vector is missing then it is
        associated to the class ``-1``. If only few values are missing
        within an input vector then the BMU is found according to the
        non-missing values remaining
        
        :param data_map: Vectors that have to be mapped (*real* data)
        :type data_map: np.ndarray[float], shape = (_, p)

        :return: 
            Vector of classes representing each vector of ``data_map``
            ie: ``classes[k]`` is the BMU of ``data_map[k]``
        :rtype: np.ndarray[float], shape(len(data_map))
        """

        start = time()
        nb_inputs = len(data_map)
        classes = np.zeros(nb_inputs, dtype=int)
        sess = tf.Session()
        weights = tf.placeholder(tf.float32, (self.nb_class, self.__p))
        vect_input = tf.placeholder(tf.float32, (1, self.__p))

        difference = weights - vect_input
        difference = tf.where(tf.is_nan(difference), 
                                tf.zeros_like(difference), 
                                difference)
        bmu_loc_op = tf.argmin(tf.sqrt(tf.reduce_sum(difference**2, axis=1)))
        
        # bmu_loc_op = tf.argmin(tf.reduce_sum(abs(weights - vect_input),axis=1), name="BMU_Op")

        t = 0
        relerr = []  # list of relative error for each input vector
        nb_missing_data = 0  # nb of missing values in each input vector
        nb_missing_values = 0  # total number of missing values in data_map
        nb_missing_vectors = 0  # total number of missing vectors in data_map

        # Main loop:
        # map (find the BMU) each vector in data_map
        for v_i in data_map:
            v_i = np.reshape(v_i, (1, self.__p))

            nb_missing_data = np.count_nonzero(np.isnan(v_i))
            nb_missing_values += nb_missing_data
                
            # affect default class to bmu_loc if the entire vector is missing
            if nb_missing_data == self.__p:
                bmu_loc = -1
                nb_missing_vectors += 1
            # else find the BMU (ignoring missing values)
            else:
                bmu_loc = int(
                    sess.run(
                        bmu_loc_op,
                        feed_dict={vect_input: v_i, weights: self.__weights},
                    )
                )

            classes[t] = bmu_loc
            self.__nb_inputs_mapped += 1

            # Increment BMU occurence
            # (Helps to detect if classes are homogeneously used)
            if bmu_loc > -1:
                self.__occ_bmu_map[bmu_loc] += 1
                # Keep track of the instants t mapped for this BMU
                # (Helps to detect if classes represent a certain period)
                self.__t_as_bmu[bmu_loc].append(t)

            if t > 0:
                # The following lines help to keep track of info needed
                # to detect if the topology of the SOM is consistent
                if classes[t - 1] == -1 or classes[t] == -1:
                    self.__nb_missing_transitions += 1 
                else:
                    # Increment transition between BMU(t-1) and BMU(t)
                    self.__transition[classes[t - 1], classes[t]] += 1
                    # distance (in the SOM) between BMU(t-1) and BMU(t)
                    # Note that nb_inputs_mapped != len(dist_transition)
                    self.__dist_transition = np.append(
                        self.__dist_transition,
                        self.__distance_matrix[classes[t - 1], classes[t]],
                    )

            # Compute relative error
            if bmu_loc > -1:
                relerr = np.append(
                    relerr, 
                    self.__compute_relerr(self.__weights[bmu_loc], 
                                          v_i, 
                                          is_standardized=True)
                )
            else:
                relerr = np.append(relerr, np.nan)
            t = t + 1

        relerr = np.nanmean(relerr)
        values = self.class2weights(classes)
        true_values = self.destandardize(data_map)
        mse = self.__compute_mse(values, true_values, is_standardized=False)
        self.__relerr_test = np.append(self.__relerr_test, relerr)
        sess.close()
        end = time()
        print(nb_inputs, " vectors mapped in ", round(end - start, 2), " sec")
        if nb_missing_values:
            print("Number of missing values: ", nb_missing_values)
            print("Number of missing vectors: ", nb_missing_vectors)
        # print(" mean relative error: ", relerr)
        # print(" MSE: ", mse)

        return classes

    def __compute_mse(
            self, values: np.ndarray, true_values: np.ndarray, 
            is_standardized: bool = False) -> List[float]:
        # If there is only one input we consider it as a time serie of
        # length one to process it as a normal time serie
        if len(values.shape) < 2:
            values = np.array([values])
            true_values = np.array([true_values]) 
        T = len(values)
        mse = 0
        if T == 1:
            values = [values]
            true_values = [true_values]    

        # In this function we would like to compute errors with
        # non-standardized data
        if is_standardized:
            values = self.destandardize(values)
            true_values = self.destandardize(true_values)
            
        # Handling missing data
        missing_data = np.logical_or(np.isnan(true_values),
                                     np.isnan(values))
                
        values = np.where(np.isnan(values), 
                          np.zeros_like(values), 
                          values)
        true_values = np.where(np.isnan(true_values), 
                          np.zeros_like(true_values), 
                          true_values)

        missing_vectors = 0
        for t in range(T):
            if not np.all(missing_data[t]):
                # There is at least one non-missing point in common
                mse += (
                    np.sum((true_values[t] - values[t])**2)
                    / (self.__p - np.count_nonzero(missing_data[t])) )
            else:
                missing_vectors += 1
        mse = mse / (T - missing_vectors)
        return mse
        
    def __compute_relerr(
            self, values: np.ndarray, true_values: np.ndarray, 
            is_standardized: bool = False) -> List[float]:

        # If there is only one input we consider it as a time serie of
        # length one to process it as a normal time serie
        if len(values.shape) < 2:
            values = np.array([values])
            true_values = np.array([true_values]) 
        T = len(values)
        relerr = np.zeros(T)

        # In this function we would like to compute errors with
        # non-standardized data
        if is_standardized:
            values = self.destandardize(values)
            true_values = self.destandardize(true_values)

        # Handling missing data
        missing_data = np.logical_or(np.isnan(true_values),
                                     np.isnan(values))
        values = np.where(np.isnan(values), 
                          np.zeros_like(values), 
                          values)
        true_values = np.where(np.isnan(true_values), 
                          np.zeros_like(true_values), 
                          true_values)
        
        for t in range(T):
            if not np.all(missing_data[t]):
                # There is at least one non-missing point in common

                # Distinguish vectors from scalars
                if self.__p > 1:
                    relerr[t] = (
                        LA.norm(values[t] - true_values[t])
                        / LA.norm(true_values[t]) )
                else:
                    relerr[t] = (
                        abs(values[t] - true_values[t])
                        / abs(true_values[t]))
            else:
                relerr[t] = np.nan
        return relerr

    def clear_map_info(self) -> None:
        """Clear all data related to the mapping phase

        Data related to the mapping phase can help to understand the
        behavior of the trained SOM and to evaluate its quality.
        However if the SOM has been trained again after having already
        mapped some inputs it could be wise to delete previous map
        information stored in the SOM.
        Similarly, if the same input was mapped twice (or more) map
        info should be cleared and the mapping phase done again.

        List of data cleared:
            - :ref:`nb_inputs_mapped <submapp.som.Som.nb_inputs_mapped>`
            - :ref:`distance_transitions <submapp.som.Som.distance_transitions>`
            - :ref:`transition <submapp.som.Som.transition>`
            - :ref:`relerr_test <submapp.som.Som.relerr_test>`
            - :ref:`occ_bmu_map <submapp.som.Som.occ_bmu_map>`
            - :ref:`time_info <submapp.som.Som.time_info>`
        
        :rtype: None
        """

        self.__nb_inputs_mapped = 0
        self.__occ_bmu_map = np.zeros(self.nb_class, dtype=int)
        self.__t_as_bmu = [[] for i in range(self.nb_class)]
        self.__transition = np.zeros((self.nb_class, self.nb_class), dtype=int)
        self.__dist_transition = np.copy([])
        self.__relerr_test = np.copy([])

    def location_classes(self, classes: np.ndarray = None) -> np.ndarray:
        """ *Locate* classes within the SOM: give their corresponding \
        indexes ``(i,j)``

        ``location_classes[k] = [i, j]``
            - ``k`` in ``[|0,`` \
              :ref:`nb_class <submapp.som.Som.nb_class>` ``- 1|]``
              (kth class)
            - ``i`` in ``[|0,``\
              :ref:`n <submapp.som.Som.n>` ``-1|]`` (located at the
              ith row)
            - ``j`` in ``[|0,``\
              :ref:`m <submapp.som.Som.m>`-1|]`` (and at the jth 
              column)
        
        :param classes: 
            Vector of classes which have to be located. 
            Defaults to None, in this case all classes (from ``0``
            to :ref:`nb_class <submapp.som.Som.nb_class>` ``-1``) 
            will be located
        :type classes: np.ndarray[int], optional
        :return: 
            A matrix containing all the indexes corresponding to the
            given classes.
        :rtype: np.ndarray[int], shape = (_, 2)
        """

        if classes is None:
            loc = np.copy(self.__location_classes)
        else:
            # List objects don't have astype function!
            # classes should be a np.ndarray!
            classes = classes.astype(int)
            loc = self.__location_classes[classes]
        return loc

    def class2weights(self, classes: np.ndarray = None) -> np.ndarray:
        """Matrix of referent vectors associated to the given classes

        ``class2weights[k]`` is the ``k``th weights (referent vector) of 
        the SOM (``k`` in ``[|0, n*m - 1|]`` (``k``th class)
        
        :param classes: 
            Vector of classes corresponding to the referent vectors
            wanted. 
            Defaults to None, in this case all referent vectors 
            (associated to all classes, from ``0`` to ``nb_classes-1``)
            will be returned.
        :type classes: np.ndarray[int], optional
        :return: 
            A matrix containing all the referent vectors corresponding
            to the classes given.
        :rtype: np.ndarray[float], shape = (len(classes), p)
        """

        if classes is None:
            res = self.weights
        else:
            # List objects don't have astype function!
            # classes should be a np.ndarray
            T = len(classes)
            classes = np.reshape(classes.astype(int), (T,1) )

            # handle missing vectors
            np_nan = np.reshape(np.array((T*self.__p)*[np.nan]), (T, self.__p))
            res = np.where(classes == -1, np_nan, self.weights[classes].squeeze())
        return res

    def print_heatmap(
            self, data: np.ndarray = None, vmin: float = None,
            vmax: float = None) -> None:
        """Print the heatmap describing a feature of the classes

        This is a visual way to detect whether referent vectors of
        each area of the SOM share indeed some common features (only
        one feature at a time though...). If within a given aera the
        vectors have very heterogeneous values then either the 
        feature is not relevant with respect to the data or the SOM
        is not adapted to the data. 
        
        :param data: 
            Matrix whose elements describe a feature of the 
            corresponding class. Typically data could be
            ``Som.occ_bmu_map``, ``Som.weights_features``,
            Som.time_info[0] or Som.time_info[1]
            Defaults to None, in this case, ``data=Som.occ_bmu_map``
            and then the heatmap shows how much each aera of the SOM
            has been used during the mapping phase.
        :type data: np.ndarray[float, int], shape = (nb_class)
        :param vmin: 
            min value to anchor the colormap. 
            Defaults to None. In this case ``vmin = np.min(data)``
        :type vmin: float, optional
        :param vmax:
            max value to anchor the colormap. 
            Defaults to None. In this case ``vmax = np.max(data)``
        :type vmax: float, optional
        :rtype: None
        """

        # print a heatmap of the matrix data
        # (typically occ_bmu_map)
        if data is None:
            data = self.__occ_bmu_map
        if vmin is None:
            vmin = np.min(data)
        if vmax is None:
            vmax = np.max(data)

        # compute the matrix hmap of the heatmap related to data
        hmap = np.zeros((self.__n, self.__m))
        for k in range(data.size):
            [i, j] = self.__location_classes[k]
            hmap[i, j] = data[k]

        # print the heatmap of the matrix hmap
        fig = plt.figure(figsize=(15, 15 * (self.__n / self.__m)))
        ax = sns.heatmap(hmap, vmin=vmin, vmax=vmax, annot=False)
        fig.add_axes(ax)
        plt.show()

    def weights_features(
            self, features: str,
            start: int = 0, end: int = None) -> np.ndarray:
        """Extract some features from the weights

        For each weight ``k``, extracts a feature (only *mean* and 
        *standard deviation* are implemented) from
        ``weights[k, start:end]``
        
        :param features: 
            feature to extract, " ``mean`` " for the mean or 
            " ``std`` " for the standard deviation
        :type features: str
        :param start: 
            First row within the weights from which to extract the
            feature. Can be usefull if each row of the weights 
            represents completely different type of data or if
            it represents different levels of depth for instance.
            Defaults to ``0``
        :type start: int, optional
        :param end: 
            Last row within the weights from which to extract the
            feature. Defaults to None, in this case ``end = p``
        :type end: int, optional
        :rtype: np.ndarray[float], shape = (nb_class)
        """

        if end is None:
            end = self.__p
        res = np.zeros((self.nb_class))
        if features == "mean":
            res = np.mean(self.weights[:, start:end], axis=1)
        if features == "std":
            res = np.std(self.weights[:, start:end], axis=1)
        return res

    def __clear_graph(self) -> None :
        """Clear computational graph (Shouldn't be used...)
        
        :rtype: None
        """
        self.__graph = None

    def standardize(self, data: np.ndarray) -> np.ndarray :
        """ Substact ``Som.data_mean`` and divide by ``Som.data_stdev``

        .. Note:: 
            :ref:`destandardize <Som.destandardize>`
            is the inverse operation
        
        .. Warning:: 
            This property should not be necessary under normal use.

        :param data: Data that have to be standardized
        :type data: np.ndarray[float]
        :rtype: np.ndarray

        .. seealso:: 
            - :ref:`data_mean <submapp.som.Som.data_mean>`
            - :ref:`data_stdev <submapp.som.Som.data_stdev>`
            
        """
        return (data - self.__mean) / self.__stdev

    def destandardize(self, data_norm: np.ndarray) -> np.ndarray :
        """Multiply by ``Som.data_stdev`` and add ``Som.data_mean``

        .. Note:: 
            :ref:`standardize <submapp.som.Som.standardize>`
            is the inverse operation
        .. Warning:: 
            This property should not be necessary under normal use.

        :param data_norm: Data that have to be destandardized
        :type data_norm: np.ndarray
        :rtype: np.ndarray

        .. seealso:: 
            - :ref:`data_mean <submapp.som.Som.data_mean>`
            - :ref:`data_stdev <submapp.som.Som.data_stdev>`

        """
        return data_norm * self.__stdev + self.__mean

    @property
    def distance_matrix(self) -> np.ndarray:
        """Matrix of relative distance between each class
        
        This matrix defines the "shape" of the SOM (eg. hexagonal,
        rectangular, etc.). E.g in the case of a euclidean distance
        between classes (rectangular map) the matrix is define as
        follows:
        
        .. math:: 

            distance[k_1, k_2] & = \\sqrt{ (i_1-i_2)^2 + (j_1-j_2)^2 }

            k_1, k_2 & \\in [|0, nb\_class-1 |]

            i_1, i_2 & \\in [|0, n |]

            j_1, j_2 & \\in [|0, m |]

        Note: 
        
            ``(i, j)`` = :ref:`location_classes[k] <submapp.som.Som.location_classes>`

        :rtype: np.ndarray[float], shape = (nb_class, nb_class)
        """
        return np.copy(self.__distance_matrix)

    @property
    def weights(self) -> np.ndarray:
        """ Weights (i.e referent vectors) of the SOM
        
        ``weights[k]`` is the referent vector
        of the class ``k``. It is meant to be representative of
        some *real* data

        :rtype: np.ndarray[float], shape = (nb_class, p) 
        """
        return self.destandardize(self.__weights)

    @property
    def __standardized_weights(self) -> np.ndarray:
        """ *standardized* weights (i.e referent vectors) of the SOM

        ``standardized_weights[k]`` is the *standardized* referent vector
        of the class ``k``. standardized in the sense that they were
        trained with beforehand standardized and centered data

        To access weights that are representative of the *real* data
        either use simply ``Som.weights`` attribute or 
        ``Som.destandardize(Som.standardized_weights)``

        .. Warning:: 
            This property should not be necessary under normal use.
        
        :rtype: np.ndarray[float], shape = (nb_class, p) 
        """
        return np.copy(self.__weights)

    @property
    def occ_bmu_map(self) -> np.ndarray:
        """Occurence of each class as BMU during the mapping phase

        Let ``BMU(t)`` be the BMU at instant ``t`` during the mapping
        phase. Then: 

        .. code-block:: python

            occ_bmu_map[BMU(t)] += 1

        This attribute can help to detect whether some aeras in the SOM
        are never used or not as much as some other aeras. In this case
        it might be due to the non-adapted shape of the SOM or wrong
        set of hyperparameters 
        
        :rtype: np.ndarray[int], shape = (nb_class)

        .. warning::
        
            Should not be confused with
            :doc:`Map2d.occ_bmu_map <../../map2d/Map2d/submapp.map2d.Map2d.occ_bmu_map>`
        """
        return np.copy(self.__occ_bmu_map)

    @property
    def time_info(
            self) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """For each class, info about the instants ``t`` at which it has
        been used

        First are stored all instants ``t`` at which the kth class
        has been used as BMU in ``t_as_bmu[k]``. Then in order to help to
        detect whether each classes represent well a certain period
        the mean and the standard deviation of ``t_as_bmu[k]``
        are computed for all ``k`` in ``[|0,`` 
        :ref:`nb_class <submapp.som.Som.nb_class>` ``-1 |]``.

        Note that ``len(t_as_bmu[k])=``
        :ref:`occ_bmu_map[k] <submapp.som.Som.occ_bmu_map>`

        .. warning::

            Note that this attribute is meaningless if the SOM
            is not used with time-series or if each input has a 
            different scale of time.
        
        :return: 
            (
                - mean(t_as_bmu[k]),
                - standard_deviation(t_as_bmu[k]), 
                - t_as_bmu

            )
        :rtype: Tuple[np.ndarray[float], np.ndarray[float], List[int]]
        """
        t_as_bmu = self.__t_as_bmu.copy()
        mean_time = np.array(
            [
                -1 if t_as_bmu[i] == [] else mean(t_as_bmu[i])
                for i in range(self.nb_class)
            ]
        )
        std_time = np.array(
            [
                -1 if len(t_as_bmu[i]) < 2 else stdev(t_as_bmu[i])
                for i in range(self.nb_class)
            ]
        )
        return (mean_time, std_time, t_as_bmu)

    @property
    def shape(self) -> Tuple[int, int, int]:
        """The shape of the SOM  ``(n,m,p)``
        
        :rtype: Tuple[int, int, int]

        .. seealso:: 

            - :ref:`n <submapp.som.Som.n>` number of rows
            - :ref:`m <submapp.som.Som.m>` number of columns
            - :ref:`nb_class <submapp.som.Som.nb_class>` total number
              of classes ``n*m``
            - :ref:`p <submapp.som.Som.p>` size of the input vectors 
              (and the weights)

        """
        return (self.__n, self.__m, self.__p)

    @property
    def n(self) -> int:
        """The number of rows of the SOM 
        
        :rtype: int

        .. seealso:: 

            - :ref:`m <submapp.som.Som.m>` number of columns
            - :ref:`nb_class <submapp.som.Som.nb_class>` total number
              of classes ``n*m``
            - :ref:`p <submapp.som.Som.p>` size of the input vectors 
              (and the weights)
            - :ref:`shape <submapp.som.Som.shape>` ``(n,m,p)``

        """
        return self.__n

    @property
    def m(self) -> int:
        """The number of columns of the SOM 
        
        :rtype: int

        .. seealso:: 

            - :ref:`n <submapp.som.Som.n>` number of rows
            - :ref:`nb_class <submapp.som.Som.nb_class>` total number
              of classes ``n*m``
            - :ref:`p <submapp.som.Som.p>` size of the input vectors
              (and the weights)
            - :ref:`shape <submapp.som.Som.shape>` ``(n,m,p)``

        """
        return self.__m

    @property
    def p(self) -> int:
        """The size of input vectors (and of the weights too)
        
        :rtype: int

        .. seealso:: 

            - :ref:`n <submapp.som.Som.n>` the number of rows
            - :ref:`m <submapp.som.Som.m>` number of columns
            - :ref:`nb_class <submapp.som.Som.nb_class>` total number
              of classes ``n*m``
            - :ref:`shape <submapp.som.Som.shape>` ``(n,m,p)``
            
        """
        return self.__p

    @property
    def nb_class(self) -> int:
        """The total number of classes of the SOM (``n*m``)
        
        :rtype: int

        .. seealso:: 

            - :ref:`n <submapp.som.Som.n>` the number of rows
            - :ref:`m <submapp.som.Som.m>` number of columns
            - :ref:`p <submapp.som.Som.p>` size of the input vectors (and 
              the weights)
            - :ref:`shape <submapp.som.Som.shape>` ``(n,m,p)``

        """
        return self.__n * self.__m

    @property
    def nb_training_iterations(self) -> int:
        """Total number of vectors used to train the SOM
        
        :rtype: int
        """
        return self.__nb_training_iterations

    @property
    def nb_inputs_mapped(self) -> int:
        """Total number of vectors mapped by the SOM
        
        :rtype: int
        """
        return self.__nb_inputs_mapped

    @property
    def data_mean(self) -> np.ndarray:
        """Mean of the training data

        Usually all data are standardized (especially when
        different types of data are used in the same vector). In
        order to standardized, destandardized, compute errors
        effectively the initial (i.e. of the training
        data) mean and standard deviation is stored. Here
        is the mean of the training data.

        .. seealso:: 

            :ref:`data_stdev <submapp.som.Som.data_stdev>`
            for the standard deviation
        
        :rtype: np.ndarray, shape =(1, ) or (p, )
        """
        return np.copy(self.__mean)

    @property
    def data_stdev(self) -> np.ndarray:
        """Standard deviation of the training data

        Usually all data are standardized (especially when
        different types of data are used in the same vector). In
        order to standardized, destandardized, compute errors
        effectively the initial (i.e. of the training
        data) mean and standard deviation is stored. Here
        is the standard deviation of the training data.

        .. seealso:: 

            :ref:`data_mean <submapp.som.Som.data_mean>`
            for the mean
        
        :rtype: np.ndarray, shape =(1, ) or (p, )
        """
        return np.copy(self.__stdev)

    @property
    def distance_transitions(self) -> np.ndarray:
        """Vector of distance between consecutive BMU mapped by the SOM
        
        Let ``BMU(t-1)`` and ``BMU(t)`` be the BMU at instant 
        ``t-1`` and ``t``
        
        Then: 

        .. code-block:: python

            distance_transitions = np.append(
                distance_transitions,
                distance_matrix[BMU(t-1), BMU(t)])

        Note that ``nb_inputs_mapped != len(dist_transition)`` because
        there is no transition for the first input of each data mapped

        .. warning::

            This attribute is meaningless if the SOM
            is not used with time-series or if there is no correlation
            between consecutive input vectors
        
        :rtype: np.ndarray[float]

        .. warning::
        
            Should not be confused with
            :doc:`Map2d.distance_transitions <../../map2d/Map2d/submapp.map2d.Map2d.distance_transitions>`
        """
        return np.copy(self.__dist_transition)

    @property
    def transition(self) -> np.ndarray:
        """Matrix of occurence of the transition mapped by the SOM

        Let ``BMU(t-1)`` and ``BMU(t)`` be the BMU at instant 
        ``t-1`` and ``t``
        
        Then ``transition[BMU(t-1),BMU(t)] += 1``

        .. warning::

            This attribute is meaningless if the SOM
            is not used with time-series or if each input has a 
            different scale of time.
            
        :rtype: np.ndarray[int], shape = (nb_class, nb_class)

        .. warning::
        
            Should not be confused with
            :doc:`Map2d.transition <../../map2d/Map2d/submapp.map2d.Map2d.transition>`
        """
        return np.copy(self.__transition)


    @property
    def nb_missing_transitions(self) -> int:
        """Number of transitions missed due to missing vectors
        
        :rtype: int
        """
        return self.__nb_missing_transitions

    @property
    def relerr_test(self) -> np.ndarray:
        """Relative error during the mapping phase
        
        Vector of the mean of relative errors computed every time map
        function is called

        .. seealso:: 

            :ref:`relerr_train <submapp.som.Som.relerr_train>`
            for the relative error during the training phase

        :rtype: np.ndarray[float]
        """
        return np.copy(self.__relerr_test)

    @property
    def relerr_train(self) -> np.ndarray:
        """Relative error during the training phase
        
        Vector of the mean of relative errors computed every time
        train function is called

        .. seealso:: 

            :ref:`relerr_test <submapp.som.Som.relerr_test>`
            for the relative error during the mapping phase

        :rtype: np.ndarray[float]
        """
        return np.copy(self.__relerr_train)

    @property
    def is_trained(self) -> bool:
        """Indicate whether the training phase is completed.
        
        :return: 
            Boolean indicating if the training is considered completed 
            by the user. From the point of view of the user the value
            of this boolean is of no consequence (except being a 
            reminder...) however we recommend using it consistently as
            it will automatically delete the computational graph (which
            is now useless) associated to the training phase.
            defaults to True
            Otherwise return False
        :rtype: bool
        """
        return self.__is_trained

    @is_trained.setter
    def is_trained(self, new_bool: bool = True) -> None :
        """Set the new training state of the SOM

        :param new_bool: 
            Boolean indicating if the training is considered completed 
            by the user. From the point of view of the user the value
            of this boolean is of no consequence (except being a 
            reminder...) however we recommend using it consistently as
            it will automatically delete the computational graph (which
            is now useless) associated to the training phase.
            defaults to True
            
        :type new_bool: bool, optional
        :rtype: None
        """
        if new_bool:
            self.__clear_graph()
        self.__is_trained = new_bool
