""" Implements ``Map2d`` class and some functions related to this class

Map2d objects are a convenient way of storing
:ref:`SOM <submapp.som>` and
:ref:`HMM <submapp.hmm>`
outputs. This module provides additional tools to this class of object.

"""

import numpy as np
from copy import copy as built_in_copy
import matplotlib.pyplot as plt
from numpy import linalg as LA
import pickle
import os
import submapp.som 
import sklearn.metrics as skl_m
from typing import List, Tuple
from math import sqrt


def distance_between_labels(map1, map2) -> np.ndarray:
    """Distance (within the SOM) between each label from map1 and map2

    In case of missing vectors (``NaN``) in the dataset and/or 
    their associated class ``-1``: 
    If ``map1.classes[k] == -1 or map2.classes[k] == -1 then
    distance_between_labels[k] == -1``
    
    :param map1: First Map2d
    :type map1: Map2d
    :param map2: 
        Second Map2d (``map1`` and ``map2`` are interchangeable)
    :type map2: Map2d
    :rtype: np.ndarray[float]

    .. Warning:: 

        ``map1`` and ``map2`` should have been constructed by the
        same 
        :ref:`SOM <submapp.som.Som>`)
        Otherwise, the result returned does not make any sense.
        They must have the same number of inputs mapped too.


    .. seealso::
        - :doc:`count_differences <submapp.map2d.count_differences>`
        - :doc:`confusion_matrix <submapp.map2d.confusion_matrix>`

    """

    dist = np.zeros(shape=map1.classes.size)
    for k in range(map1.classes.size):
        i = int(map1.classes[k])
        j = int(map2.classes[k])
        # -1 corresponds to a missing vector
        if i > -1 and j > -1:
            dist[k] = map1.som.distance_matrix[i, j]
        else:
            dist[k] = -1
    return dist

def count_differences(
        map1,
        map2, 
        ignore_missing_vector: bool = True) -> np.ndarray:
    """Number of differences between the classes of map1 and map2
    
    :param map1: First Map2d
    :type map1: Map2d
    :param map2: 
        Second Map2d (``map1`` and ``map2`` are interchangeable)
    :type map2: Map2d
    :param ignore_missing_vector: 
        Determine whether a difference should be ignored if a missing
        vector is involded in ``map1`` or ``map2``. If ``False`` 
        the results is incremented even though either ``map1`` or
        ``map2`` is representing a missing value
        Defaults to ``True`` 
    :type ignore_missing_vector: bool, optional 
    :rtype: int

    .. Warning:: 

        ``map1`` and ``map2`` should have been constructed by the
        same 
        :ref:`SOM <submapp.som.Som>`)
        Otherwise, the result returned does not make any sense.

        They must have the same number of inputs mapped too.

    .. seealso::
    
        - :doc:`distance_between_labels <submapp.map2d.distance_between_labels>`
        - :doc:`confusion_matrix <submapp.map2d.confusion_matrix>`

    """
    if ignore_missing_vector:
        diff = np.where(np.logical_or(map1.classes == -1, map2.classes == -1),
                        np.zeros_like(map1.classes), 
                        map1.classes - map2.classes)
    else:
        diff = map1.classes - map2.classes
    return np.count_nonzero(diff)

def confusion_matrix(
        true_map, 
        est_map,
        ignore_missing_vector: bool = True) -> np.ndarray:
    """Confusion matrix between the classes of true_map and est_map
    
    :param true_map: 
        Map representing the *true values*, whose labels are
        considered as *true labels*
    :type true_map: Map2d

    :param est_map: 
        Map representing the *estimated values*, whose labels are
        considered as *estimated labels*

        ``true_map`` and ``est_map`` are **NOT** 
        interchangeable
    :type est_map: Map2d
    :param ignore_missing_vector: 
        Determine whether a confusion should be ignored if a missing
        vector is involded in ``true_map`` or ``est_map``. If 
        ``False`` the results is updated even though either 
        ``true_map`` or ``est_map`` is representing a missing value
        Defaults to ``True`` 
    :type ignore_missing_vector: bool, optional 
    :rtype: np.ndarray[int]

    .. Warning:: 

        ``true_map`` and ``est_map`` should have been constructed by the
        same 
        :ref:`SOM <submapp.som.Som>`)
        AND represent the same set of data
        Otherwise, the result returned does not make any sense.

        They must have the same number of inputs mapped too.

    .. seealso::
    
        - :doc:`distance_between_labels <submapp.map2d.distance_between_labels>`
        - :doc:`count_differences <submapp.map2d.count_differences>`

    """
    true_classes = true_map.classes
    est_classes = est_map.classes
    if ignore_missing_vector:
        true_classes = np.where(true_classes == -1,
                                np.zeros_like(true_classes),
                                true_classes)
        est_classes = np.where(est_classes == -1,
                               np.zeros_like(est_classes),
                               est_classes)
    confusion = skl_m.confusion_matrix(
        y_true=true_classes,
        y_pred=est_classes,
        labels=np.arange(true_map.nb_class),
    )
    return confusion


def concat_maps(
        maps:List, 
        name=None):
    """ Concatenate a list of maps into a new Map2d
    
    :param maps: 
        List of Map2d object to be concatenated. All the maps
        should have been created with the same SOM
    :type maps: Map2d
    :param name:
        The *Name* of the Map2d object. This name has no impact on
        the variable name, however it can be usefull when it comes
        to saving the Map2d object and its related figures.
        Defaults to "None", in that case the name will be 
        "concat_maps"
    :type path: str, optional
    :rtype: Map2d

    .. seealso:: :doc:`copy <submapp.map2d.copy>`
    """

    if name is None:
        name = "concat_maps"
    newMap = Map2d(maps[0].som, name=name)

    for map in maps:
        newMap._Map2d__add_values(map.values, overwrite=False)
        newMap._Map2d__add_classes(map.classes, overwrite=False)
        if len(map.true_values) == 0:
            new_true_values = np.nan*np.ones_like(map.values)
        else: 
            new_true_values = map.true_values
        newMap.add_true_values(new_true_values, overwrite=False)
    return newMap

def save(
        myMap, 
        filename: str = None,
        path: str = "",
        ) -> None :
    """Save the Map2d object
    
    :param filename: 
        filename of the saved Map2d. defaults to None, in this case
        the filename is the name of the Map2d (eg. "MyMap")
    :type filename: str, optional
    :param path:
        Relative path. defaults to "", in this case the Map2d
        object is save in the current path
    :type path: str, optional
    :rtype: None

    .. seealso:: :doc:`load <submapp.map2d.load>`
    """

    myMap._Map2d__clear_graph()
    if path:
        os.makedirs(path, exist_ok=True)
    if filename is None:
        filename = myMap.name
    with open(path + filename, "wb") as f:
        pickle.dump(myMap, f)


def load(filename: str, path: str = ""):
    """Load a saved Map2d
    
    :param filename: filename of the saved Map2d
    :type filename: str
    :param path:
        Relative path. defaults to "", in this case the Map2d
        object is loaded in the current path
    :type path: str, optional
    :return: The loaded Map2d stored at "path/filename"
    :rtype: Map2d

    .. seealso:: :doc:`save <submapp.map2d.save>`
    """

    with open(path + filename, "rb") as f:
        myMap = pickle.load(f)
    return myMap

def copy(myMap):
    """Return a deep copy of the Map2d
    
    :rtype: Map2d
    """

    new_map = built_in_copy(myMap)
    new_map.som._Som__clear_graph()
    return new_map

class Map2d:
    """A map associated to a Som object representing a time series

    A Map2d object is a convenient way of manipulating
    :ref:`SOM <submapp.som>` and
    :ref:`HMM <submapp.hmm>`
    outputs as it stores in the same object:

    - :ref:`true_values <submapp.map2d.Map2d.true_values>`, 
      np.array[float], shape=(
      :ref:`nb_inputs_mapped <submapp.map2d.Map2d.nb_inputs_mapped>`,
      :ref:`p <submapp.map2d.Map2d.p>`)
          - The *real* values with which the Map2d was constructed
            if the method
            :ref:`map_from_data <submapp.map2d.Map2d.map_from_data>`
            was used. 
          - The *real* values the Map2d tries to
            reconstruct if the method 
            :ref:`map_from_classes <submapp.map2d.Map2d.map_from_classes>`
            was used. In this case 
            :ref:`classes <submapp.map2d.Map2d.classes>` 
            are propably the output of the 
            :ref:`viterbi <submapp.hmm.Hmm.viterbi>`
            method and the true values - if available - have to be
            specified by the user either when calling the method
            :ref:`map_from_classes <submapp.map2d.Map2d.map_from_classes>`
            or with the method 
            :ref:`add_true_values <submapp.map2d.Map2d.add_true_values>`

    - :ref:`classes <submapp.map2d.Map2d.classes>`, np.array[int], shape=(
      :ref:`nb_inputs_mapped <submapp.map2d.Map2d.nb_inputs_mapped>`)
      classes of the 
      :ref:`SOM <submapp.som.Som>` representing ``true_values``
    - :ref:`values <submapp.map2d.Map2d.values>`,
      np.array[float], shape=(
      :ref:`nb_inputs_mapped <submapp.map2d.Map2d.nb_inputs_mapped>`,
      :ref:`p <submapp.map2d.Map2d.p>`)
      Referent vectors associated to ``classes``
    
    In addition, it stores its associated 
    :ref:`Som <submapp.som>` 
    object so that essential properties are easy to access such as:

    - :ref:`distance_matrix <submapp.som.Som.distance_matrix>`, 
      give the distance between 2 classes
    - :ref:`distance_transitions <submapp.som.Som.distance_transitions>`,
      stores the distance of all transitions mapped by the SOM
    - :ref:`weights <submapp.som.Som.weights>`, 
      Referent vectors (weights) associated to all classes

    :var str name:
        The *Name* of the Map2d object. This name has no impact on
        the variable name, however it can be usefull when it comes
        to saving the Map2d object and its related figures, defaults 
        to "MyMap"
    """
    def __init__(
            self,
            som,
            name: str = "MyMap",
            ):
        """Class constructor

        Create a Map2d object and initialize its basic properties.
        Please note that it won't be allowed to
        change the 
        :ref:`som <submapp.map2d.Map2d.som>`
        attribute (see class :ref:`Som <submapp.som.Som>`) once this
        function is called! If the shape (for instance) doesn't seem
        good then create another Map2d object with a
        different shape.
        
        :param som: SOM of the Map2d
        :type som: Som
        :param name: 
            The *Name* of the Map2d object. This name has no impact on
            the variable name, however it can be usefull when it comes
            to saving the Map2d object and its related figures.
            Defaults to "MyMap"
        :type name: str, optional
        """

        self.__true_values = np.array([])
        self.__values = np.array([])
        self.__classes = np.array([])
        self.__som = submapp.som.copy(som)
        self.name = name

    def map_from_data(
            self,
            inputs: np.ndarray,
            overwrite: bool = False,
            ) -> None:
        """Construct the Map2d from real data

        .. note:: 

            Missing values are represented by ``NaN``
            and the class associated to a missing vector is
            ``-1``
        
        :param inputs: 
            Real data that are going to be represented by the Map2d
        :type inputs: np.ndarray[float], shape = (_, p)
        :param overwrite: 
            Indicate whether the previous values (
            :ref:`values <submapp.map2d.Map2d.values>`, 
            :ref:`true_values <submapp.map2d.Map2d.true_values>`, 
            :ref:`classes <submapp.map2d.Map2d.classes>`, 
            ) should be cleared.
            Defaults to False, in this case, the new values are
            appended to the current ones.
        :type overwrite: bool, optional
        :rtype: None

        .. seealso:: 

            :ref:`map_from_classes <submapp.map2d.Map2d.map_from_classes>`

        """
        
        # We use self.som rather than self.__som not to change map_info
        new_classes = self.som.map(inputs)
        new_values = self.__som.class2weights(new_classes).squeeze()
        true_values = self.__som.destandardize(inputs)

        self.__add_classes(new_classes, overwrite=overwrite)
        self.__add_values(new_values, overwrite=overwrite)
        self.add_true_values(true_values, overwrite=overwrite)


    def map_from_classes(
            self,
            classes: np.ndarray,
            true_values: np.ndarray = [],
            overwrite: bool = False,
            ) -> None:
        """Construct the Map2d from a vector of classes

        .. note:: 
        
            Missing values are represented by ``NaN``
            and the class associated to an entire missing vector is
            ``-1``

        :param classes: Classes representing some *real* data
        :type classes: np.ndarray[int]
        :param true_values: 
            *Real* values. Values represented by the Map2d.
            Defaults to []. In this case the true values are supposed
            unkown. In this case, the relative error
            :ref:`relerr <submapp.map2d.Map2d.relerr>`
            of the Map2d can't be computed
        :type true_values: np.ndarray[float], optional
        :param overwrite: 
            Indicate whether the previous values (
            :ref:`values <submapp.map2d.Map2d.values>`, 
            :ref:`true_values <submapp.map2d.Map2d.true_values>`, 
            :ref:`classes <submapp.map2d.Map2d.classes>`, 
            ) should be cleared.
            Defaults to False, in this case, the new values are
            appended to the current ones.
        :type overwrite: bool, optional
        :rtype: None

        .. seealso:: 

            :ref:`map_from_data <submapp.map2d.Map2d.map_from_data>`

        """

        new_values = self.__som.class2weights(classes)
        self.__add_classes(classes, overwrite=overwrite)
        self.__add_values(new_values, overwrite=overwrite)
        self.add_true_values(true_values, overwrite=overwrite)

    def add_true_values(
            self,
            new_true_values: np.ndarray,
            overwrite: bool = True,
            ) -> None:
        """Add true values to the Map2d
        
        :param new_true_values: 
            New
            :ref:`true_values <submapp.map2d.Map2d.true_values>`
            value
        :type new_true_values: np.ndarray[float]
        :param overwrite: 
            Determine whether ``new_true_values`` should overwrite the 
            current 
            :ref:`true_values <submapp.map2d.Map2d.true_values>`
            Defaults to True
        :type overwrite: bool, optional
        :rtype: None
        """
        if self.true_values.size == 0 or overwrite:
            self.__true_values = np.copy(new_true_values)
        else:
            self.__true_values = np.concatenate(
                (self.__true_values, new_true_values)
            )            

    def __add_values(
            self,
            new_values: np.ndarray,
            overwrite: bool = False,
            ) -> None:
        """Add values to the Map2d
        
        :param new_values: 
            New 
            :ref:`values <submapp.map2d.Map2d.values>`
            value
        :type new_values: np.ndarray[float]
        :param overwrite: 
            Determine whether ``new_value`` should overwrite the 
            current 
            :ref:`values <submapp.map2d.Map2d.values>`
            Defaults to False
        :type overwrite: bool, optional
        :rtype: None
        """
        if self.__values.size == 0 or overwrite:
            self.__values = np.copy(new_values)
        else:
            self.__values = np.concatenate(
                (self.__values, new_values)
            )

    def __add_classes(
            self,
            new_classes: np.ndarray,
            overwrite: bool = False,
            ) -> None:
        """Add classes to the Map2d
        
        :param new_classes: New
        :ref:`classes <submapp.map2d.Map2d.classes>`
        :type new_classes: np.ndarray[float]
        :param overwrite: 
            Determine whether ``new_classes`` should overwrite the 
            current 
            :ref:`classes <submapp.map2d.Map2d.classes>`
            Defaults to False
        :type overwrite: bool, optional
        :rtype: None
        """
        if self.__classes.size == 0 or overwrite:
            self.__classes = np.copy(new_classes)
        else:
            self.__classes = np.concatenate(
                (self.__classes, new_classes)
            )

    def __clear_graph(self):
        """Clear computational graph (Shouldn't be used...)
        
        :rtype: None
        """
        if self.__som is not None:
            self.__som._Som__clear_graph()

    @property
    def shape(self) -> Tuple[int,int,int]:
        """The shape of the SOM  ``(n,m,p)``
        
        :rtype: Tuple[int, int, int]

        .. seealso:: 

            - :ref:`n <submapp.map2d.Map2d.n>` number of columns
            - :ref:`m <submapp.map2d.Map2d.m>` number of columns
            - :ref:`nb_class <submapp.map2d.Map2d.nb_class>` total number
              of classes ``n*m``
            - :ref:`p <submapp.map2d.Map2d.p>` size of the input vectors 
              (and the weights)
            - :ref:`som <submapp.map2d.Map2d.som>` SOM of the Map2d
            - :ref:`Som <submapp.som.Som>` class **Som**

        """

        return (self.__som.n, self.__som.m, self.__som.p)

    @property
    def n(self) -> int:
        """The number of rows of the SOM 
        
        :rtype: int

        .. seealso:: 

            - :ref:`m <submapp.map2d.Map2d.m>` number of columns
            - :ref:`nb_class <submapp.map2d.Map2d.nb_class>` total number
              of classes ``n*m``
            - :ref:`p <submapp.map2d.Map2d.p>` size of the input vectors 
              (and the weights)
            - :ref:`shape <submapp.map2d.Map2d.shape>` ``(n,m,p)``
            - :ref:`som <submapp.map2d.Map2d.som>` SOM of the Map2d
            - :ref:`Som <submapp.som.Som>` class **Som**

        """

        return self.__som.n

    @property
    def m(self) -> int:
        """The number of columns of the SOM 
        
        :rtype: int

        .. seealso:: 

            - :ref:`n <submapp.map2d.Map2d.n>` number of rows
            - :ref:`nb_class <submapp.map2d.Map2d.nb_class>` total number
              of classes ``n*m``
            - :ref:`p <submapp.map2d.Map2d.p>` size of the input vectors
              (and the weights)
            - :ref:`shape <submapp.map2d.Map2d.shape>` ``(n,m,p)``
            - :ref:`som <submapp.map2d.Map2d.som>` SOM of the Map2d
            - :ref:`Som <submapp.som.Som>` class **Som**

        """

        return self.__som.m

    @property
    def p(self) -> int:
        """The size of input vectors (and of the weights too)
        
        :rtype: int

        .. seealso:: 

            - :ref:`n <submapp.map2d.Map2d.n>` the number of rows
            - :ref:`m <submapp.map2d.Map2d.m>` number of columns
            - :ref:`nb_class <submapp.map2d.Map2d.nb_class>` total number
              of classes ``n*m``
            - :ref:`shape <submapp.map2d.Map2d.shape>` ``(n,m,p)``
            - :ref:`som <submapp.map2d.Map2d.som>` SOM of the Map2d
            - :ref:`Som <submapp.som.Som>` class **Som**
            
        """

        return self.__som.p

    @property
    def nb_class(self) -> int:
        """The total number of classes of the SOM (``n*m``)

        .. note::

            In case of missing vectors the special class ``-1``
            exists too
        
        :rtype: int

        .. seealso:: 

            - :ref:`n <submapp.map2d.Map2d.n>` the number of rows
            - :ref:`m <submapp.map2d.Map2d.m>` number of columns
            - :ref:`p <submapp.map2d.Map2d.p>` size of the input vectors (and 
              the weights)
            - :ref:`shape <submapp.map2d.Map2d.shape>` ``(n,m,p)``
            - :ref:`som <submapp.map2d.Map2d.som>` SOM of the Map2d
            - :ref:`Som <submapp.som.Som>` class **Som**

        """

        return self.__som.nb_class

    @property
    def som(self):
        """SOM of the Map2d
        
        :rtype: Som

        .. seealso::

            - :ref:`Som <submapp.som.Som>` class **Som**

        """

        return submapp.som.copy(self.__som)

    @property
    def classes(self) -> np.ndarray:
        """Vector of classes used by the Map2d

        For all element ``c`` in ``classes``, -1 <= c <= 
        :doc:`nb_class <submapp.map2d.Map2d.nb_class>`

        .. note:: ``-1`` represents a missing vector in the data set
        
        :rtype: np.ndarray[int]
        """

        return np.copy(self.__classes)

    @property
    def values(self) -> np.ndarray:
        """Referent vectors associated to the classes used by the Map2d

        .. note:: Missing values are represented by ``NaN``
        
        :rtype: np.ndarray[float], shape(nb_inputs_mapped, p)
        """

        return np.copy(self.__values)

    @property
    def true_values(self) -> np.ndarray:
        """Values of the real data represented by the Map2d

        .. note:: Missing values are represented by ``NaN``

        .. note::

            If the map has been constructed with 
            :ref:`map_from_classes <submapp.map2d.Map2d.map_from_classes>`
            and not with
            :ref:`map_from_data <submapp.map2d.Map2d.map_from_data>`
            ``true_values`` could be empty either because it was not
            specified at this moment even though true values are known
            or because the true values are unknown. It can be the case
            if the Map2d is used to reconstruct new data using 
            :ref:`viterbi <submapp.hmm.Hmm.viterbi>`
            from the class
            :ref:`HMM <submapp.hmm.Hmm>`

        .. seealso::

            :ref:`add_true_values <submapp.map2d.Map2d.add_true_values>`

        :rtype: np.ndarray[float], shape(_, p)
        """

        return np.copy(self.__true_values)

    @property
    def relerr(self) -> np.ndarray:
        """Relative error between ``values`` and ``true_values`` of\
        the Map2d

        .. note::

            If the map has been constructed with 
            :ref:`map_from_classes <submapp.map2d.Map2d.map_from_classes>`
            and not with
            :ref:`map_from_data <submapp.map2d.Map2d.map_from_data>`
            then 
            :ref:`true_values <submapp.map2d.Map2d.true_values>`
            could be empty either because it was not
            specified at this moment even though true values are known
            or because the true values are unknown. It can be the case
            if the Map2d is used to reconstruct new data using 
            :ref:`viterbi <submapp.hmm.Hmm.viterbi>`
            from the class
            :ref:`HMM <submapp.hmm.Hmm>`

            In this case relerr can not be computed.

        .. warning:: 
        
            Error on missing values are ignored so long as one
            common point between 
            :doc:`true_values[t] <submapp.map2d.Map2d.true_values>`
            and 
            :doc:`values[t] <submapp.map2d.Map2d.values>` exists.
            Otherwise ``relerr[t] = Nan``

        :rtype: np.ndarray[float]
        """

        missing_data = np.logical_or(np.isnan(self.__true_values),
                                     np.isnan(self.__values))
        values = np.where(np.isnan(self.__values), 
                          np.zeros_like(self.__values), 
                          self.__values)
        true_values = np.where(np.isnan(self.__true_values), 
                          np.zeros_like(self.__true_values), 
                          self.__true_values)
        T = len(self.__values)
        new_relerr = np.zeros(T)
        for t in range(T):
            if not np.all(missing_data[t]):
                # There is at least one non-missing point in common
                if self.__som.p > 1:
                    new_relerr[t] = LA.norm(
                        values[t] - true_values[t]
                    ) / LA.norm(true_values[t])
                else:
                    new_relerr[t] = (
                        abs(values[t] - true_values[t])
                        / abs(true_values[t]))
            else:
                new_relerr[t] = np.nan
        return new_relerr

    @property
    def occ_bmu_map(self) -> np.ndarray:
        """Occurence of each class as BMU within the map

        Let ``BMU(t)`` be the BMU at instant ``t`` then: 

        .. code-block:: python

            occ_bmu_map[BMU(t)] += 1
        
        :rtype: np.ndarray[int], shape = (nb_class)

        .. warning::
        
            Should not be confused with
            :doc:`Som.occ_bmu_map <../../som/Som/submapp.som.Som.occ_bmu_map>`
        """
        
        occ = np.zeros((self.nb_class))
        u, counts = np.unique(self.__classes, return_counts=True)
        occ[u[1:]] = counts[1:]
        return occ

    @property
    def distance_transitions(self) -> np.ndarray:
        """Vector of distance between consecutive BMU within the map
        
        Let ``BMU(t-1)`` and ``BMU(t)`` be the BMU at instant 
        ``t-1`` and ``t``
        
        Then: 

        .. code-block:: python

            distance_transitions = np.append(
                distance_transitions,
                distance_matrix[BMU(t-1), BMU(t)])

        .. warning::

            This attribute is meaningless if the map
            is not used with time-series or if there is no correlation
            between consecutive input vectors

        :rtype: np.ndarray[float]

        .. warning::
        
            Should not be confused with
            :doc:`Som.distance_transitions <../../som/Som/submapp.som.Som.distance_transitions>`

        .. seealso::

            - :ref:`nb_missing_transitions <submapp.map2d.Map2d.nb_missing_transitions>`
            - :ref:`transition <submapp.map2d.Map2d.transition>`
        """

        nb_inputs = self.__classes.size
        if nb_inputs > 1:
            tr = np.zeros((nb_inputs-1))
            for t in range(0, nb_inputs-1):
                c_i = self.__classes[t]
                c_j = self.__classes[t + 1]
                if c_i > -1 and c_j > -1:
                    tr[t] = self.__som.distance_matrix[c_i,c_j]
        else:
            tr = np.array([])
        return tr

    @property
    def transition(self) -> np.ndarray:
        """Matrix of occurence of transitions 

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
            :doc:`Som.transition <../../som/Som/submapp.som.Som.transition>`

        .. seealso::

            - :ref:`nb_missing_transitions <submapp.map2d.Map2d.nb_missing_transitions>`
            - :ref:`distance_transitions <submapp.map2d.Map2d.distance_transitions>`
        """

        nb_inputs = self.__classes.size
        if nb_inputs > 1:
            tr = np.zeros((self.nb_class, self.nb_class))
            for t in range(0, nb_inputs-1):
                c_i = self.__classes[t]
                c_j = self.__classes[t + 1]
                if c_i > -1 and c_j > -1:
                    tr[c_i, c_j] += 1
        else:
            tr = np.array([])
        return tr

    @property
    def nb_missing_transitions(self) -> int:
        """Number of transitions missed due to missing vectors
        
        :rtype: int

        .. seealso::

            - :ref:`nb_missing_values <submapp.map2d.Map2d.nb_missing_values>`
            - :ref:`nb_missing_vectors <submapp.map2d.Map2d.nb_missing_vectors>`
            - :ref:`transition <submapp.map2d.Map2d.transition>`
            - :ref:`distance_transitions <submapp.map2d.Map2d.distance_transitions>`
            - :doc:`nb_inputs_mapped <submapp.map2d.Map2d.nb_inputs_mapped>`
        """
        n = np.size(self.__classes)
        res = 0
        if n>1:
            for t in range(1,n):
                if self.__classes[t - 1] == -1 or self.__classes[t] == -1:
                    res += 1    
        return res

    @property
    def nb_missing_vectors(self) -> int:
        """Number of missing vectors in the data mapped
        
        :rtype: int

        .. seealso::

            - :ref:`nb_missing_values <submapp.map2d.Map2d.nb_missing_values>`
            - :ref:`nb_missing_transitions <submapp.map2d.Map2d.nb_missing_transitions>`
            - :doc:`nb_inputs_mapped <submapp.map2d.Map2d.nb_inputs_mapped>`
        """
        n = np.size(self.__classes)
        res = 0
        if n>0:
            for c in self.__classes:
                if c == -1:
                    res += 1    
        return res

    @property
    def nb_missing_values(self) -> int:
        """Number of missing values in the data mapped
        
        :rtype: int

        .. seealso::

            - :ref:`nb_missing_vectors <submapp.map2d.Map2d.nb_missing_vectors>`
            - :ref:`nb_missing_transitions <submapp.map2d.Map2d.nb_missing_transitions>`
            - :doc:`nb_inputs_mapped <submapp.map2d.Map2d.nb_inputs_mapped>`
        """
        return np.count_nonzero(np.isnan(self.__values))

    @property
    def nb_classes_used(self) -> int:
        """Number of different classes used by the Map2d

        .. note::
        
            In case of missing vectors the associated class ``-1``
            is counted like any other classes.
        
        :rtype: int

        .. seealso::

            - :doc:`nb_missing_vectors <submapp.map2d.Map2d.nb_missing_vectors>`
            - :doc:`nb_missing_transitions <submapp.map2d.Map2d.nb_missing_transitions>`
            - :doc:`nb_missing_values <submapp.map2d.Map2d.nb_missing_values>`
            - :doc:`nb_inputs_mapped <submapp.map2d.Map2d.nb_inputs_mapped>`
        """

        u = np.unique(self.__classes)
        return u.size

    @property
    def nb_inputs_mapped(self) -> int:
        """Number of inputs (classes or vectors) mapped by the Map2

        .. note::
        
            Missing vectors or classes are counted like any other
            regular vectors or classes.
                
        :rtype: int

        .. seealso::

            - :doc:`nb_missing_vectors <submapp.map2d.Map2d.nb_missing_vectors>`
            - :doc:`nb_missing_transitions <submapp.map2d.Map2d.nb_missing_transitions>`
            - :doc:`nb_missing_values <submapp.map2d.Map2d.nb_missing_values>`
            - :doc:`nb_classes_used <submapp.map2d.Map2d.nb_classes_used>`
        """
        return np.size(self.__classes)
