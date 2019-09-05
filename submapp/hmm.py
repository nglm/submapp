""" ``hmm`` module implementing ``Hmm`` class

"""

from math import exp, log, isnan, nan, inf
import numpy as np
from numpy import linalg as LA
from copy import copy as built_in_copy
import matplotlib.pyplot as plt
from time import time
import tensorflow as tf
from statistics import mean, variance
# from scipy.misc import logsumexp  # outdated, use the line below
from scipy.special import logsumexp
import pickle
import os
import sys
#from .map2d import Map2d
from typing import List, Tuple, TypeVar

def save(hmm, filename: str = None, path: str = "") -> None:
    """Save the HMM object
    
    :param filename: 
        filename of the saved HMM. defaults to None, in this case
        the filename is the name of the HMM (eg. "MyHmm")
    :type filename: str, optional
    :param path:
        Relative path. defaults to "", in this case the HMM
        object is save in the current path
    :type path: str, optional
    :rtype: None

    .. seealso:: :doc:`load <submapp.hmm.load>`
    """

    hmm._Hmm__graph = None
    if path:
        os.makedirs(path, exist_ok=True)
    if filename is None:
        filename = hmm.name
    with open(path + filename, "wb") as f:
        pickle.dump(hmm, f)

def load(filename: str, path: str = ""):
    """Load a saved HMM
    
    :param filename: filename of the saved HMM
    :type filename: str
    :param path:
        Relative path. defaults to "", in this case the HMM
        object is loaded in the current path
    :type path: str, optional
    :return: The loaded HMM stored at "path/filename"
    :rtype: Hmm

    .. seealso:: :doc:`save <submapp.hmm.save>`
    """

    with open(path + filename, "rb") as f:
        myHmm = pickle.load(f)
    return myHmm

def copy(hmm):
    """Return a deep copy of the HMM
    
    :rtype: Hmm
    """

    new_hmm = built_in_copy(hmm)
    new_hmm._Hmm__graph = None
    return new_hmm


class Hmm:
    """Hidden Markov Model



    :var str name: 
        The *Name* of the Hmm object. This name has no impact on
        the variable name, however it can be usefull when it comes
        to saving the Hmm object and its related figures, defaults 
        to "MyHmm"
    """
    def __init__(
            self,
            n_obs: int,
            n_dis: int ,
            name: str = "MyHMM",
            ):
        """Class constructor
        
        :param n_obs: Total number of observable classes
        :type n_obs: int
        :param n_dis: Total number of hidden classes
        :type n_dis: int
        :param name: 
            The *Name* of the Hmm object. This name has no impact on
            the variable name, however it can be usefull when it comes
            to saving the Hmm object and its related figures, defaults 
            to "MyHmm"
        :type name: str, optional
        """
        self.__n_obs = int(abs(n_obs))
        self.__n_dis = int(abs(n_dis))
        self.__Em = np.zeros((self.__n_obs, self.__n_dis))
        self.__Tr = np.zeros((self.__n_dis, self.__n_dis))
        self.__pi = np.zeros((self.__n_dis), dtype=np.float32)
        self.__ln_Tr_eps = np.zeros((self.__n_dis, self.__n_dis))
        self.__ln_Em_eps = np.zeros((self.__n_obs, self.__n_dis))
        self.__ln_pi_eps = np.zeros((self.__n_dis))
        self.__sum_ln_Tr_eps = np.zeros((self.__n_dis))
        self.__sum_ln_Em_eps = np.zeros((self.__n_dis))
        self.__sum_ln_pi_eps = 0

        self.name = name
        self.__graph = None

    def init_model(
            self,
            classes_obs: np.ndarray,
            classes_dis: np.ndarray,
            ln_eps: float = None,
            ) -> None:
        """Initializes the model ``(Tr, Em, pi)`` and pseudo-counts
        
        :param classes_obs: 
            Observable classes used as a priori knowlegde about
            the model (*training data*)
        :type classes_obs: np.ndarray[int]
        :param classes_dis:
            Hidden classes used as a priori knowlegde about
            the model (*training data*)
        :type classes_dis: np.ndarray[int]
        :param ln_eps: 
            Order of magnitude (logarithm, <0) of the pseudo counts used to prevent
            null probabilities within the model.
            Defaults to None, in this case ``ln_eps = -20``
        :type ln_eps: float, optional
        :rtype: None
        """

        c_dis = classes_dis.astype(int)
        c_obs = classes_obs.astype(int)
        T = c_obs.size

        # Initialization (t=0), => doesn't concern the transition
        if c_dis[0] > -1:
            self.__pi[c_dis[0]] += 1
            if c_obs[0] > -1:
                self.__Em[c_obs[0], c_dis[0]] += 1
        # For (t>0) => doesn't concern pi anymore
        for t in range(1, T):
            if c_dis[t] > -1:
                if c_obs[t] > -1:
                    self.__Em[c_obs[t], c_dis[t]] += 1
                if c_dis[t] > -1:
                    self.__Tr[c_dis[t - 1], c_dis[t]] += 1

        # Add very low probabilities to prevent null probabilities
        self.__Em += 1e-1
        self.__Tr += 1e-1
        self.__pi += 1e-1

        # See the corresponding setters for more details!
        self.Em = self.__Em
        self.Tr = self.__Tr
        self.pi = self.__pi

        # Pseudo-counts used in Baum-Welch algorithm to prevent null
        # probabilities
        if ln_eps is None:
            ln_eps = -20
        self.ln_Em_eps = np.random.uniform(
            ln_eps - 1, ln_eps, (self.__n_obs, self.__n_dis)
        )
        self.ln_Tr_eps = np.random.uniform(
            ln_eps - 1, ln_eps, (self.__n_dis, self.__n_dis)
        )

        self.ln_pi_eps = np.random.uniform(
            ln_eps - 1, ln_eps, (self.__n_dis)
        )

    def __forward_procedure(self, Tr, Em, pi, obs):
        T = tf.size(obs)
        ln_alpha = pi + Em[obs[0], :]
        ln_alpha = tf.reshape(ln_alpha, (1, self.__n_dis))
        for t in range(1, T):
            ln_alpha = tf.concat(
                [
                    ln_alpha,
                    [
                        Em[obs[t]]
                        + tf.reduce_logsumexp(
                            tf.reshape(ln_alpha[-1], (self.__n_dis, 1)) + Tr,
                            axis=0,
                        )
                    ],
                ],
                axis=0,
            )
        # self.__check_norm_alpha(ln_alpha)
        return ln_alpha

    def __backward_procedure(self, Tr, Em, obs):
        T = tf.size(obs)
        ln_beta = tf.concat(
            [
                [tf.reduce_logsumexp(Tr + Em[obs[-1]], axis=1)],
                tf.zeros(shape=(1, self.__n_dis), dtype=np.float32),
            ],
            axis=0,
        )

        for t in range(T - 3, -1, -1):
            ln_beta = tf.concat(
                [
                    [
                        tf.reduce_logsumexp(
                            ln_beta[0] + Tr + Em[obs[t + 1]], axis=1
                        )
                    ],
                    ln_beta,
                ],
                axis=0,
            )
        # self.__check_norm_beta(ln_beta[:-1])
        return ln_beta

    def __aux(self, ln_alpha, ln_beta):
        prod = ln_alpha + ln_beta
        sum_prod = tf.reduce_logsumexp(prod, axis=1)
        return [prod, sum_prod]

    def __update_gamma(self, prod, sum_prod):
        (T, _) = prod.shape
        ln_gamma = prod - tf.reshape(sum_prod, (T, 1))
        sum_ln_gamma = tf.reduce_logsumexp(ln_gamma, axis=0)
        # self.__check_norm_gamma(ln_gamma)
        return [ln_gamma, sum_ln_gamma]

    def __update_xi(self, Tr, Em, ln_beta, ln_alpha, sum_prod, obs):
        T = tf.size(obs)
        num = -inf * np.ones(
            shape=(T - 1, self.__n_dis, self.__n_dis), dtype=np.float32
        )

        # sum_ln_xi = tf.cast(-inf*np.ones(shape=(self.__n_dis,self.__n_dis)),tf.float32)
        num[0] = (
            tf.reshape(ln_alpha[0], (self.__n_dis, 1))
            + tf.reshape(ln_beta[1], (1, self.__n_dis))
            + tf.reshape(Em[obs[1]], (1, self.__n_dis))
            + Tr
            - sum_prod[0]
        )

        for t in range(1, T - 1):
            num[t] = (
                tf.reshape(ln_alpha[t], (self.__n_dis, 1))
                + tf.reshape(ln_beta[t + 1], (1, self.__n_dis))
                + tf.reshape(Em[obs[t + 1]], (1, self.__n_dis))
                + Tr
                - sum_prod[t]
            )

        tmp = tf.cast(tf.reduce_logsumexp(num, axis=0), tf.float32)
        return tmp

    def __update_Tr(self, sum_ln_xi, sum_ln_gamma, ln_gamma):
        num = tf.log(
            tf.exp(sum_ln_xi) + tf.exp(tf.cast(self.__ln_Tr_eps, tf.float32))
        )
        den = tf.log(
            tf.reshape(tf.exp(sum_ln_gamma), (self.__n_dis, 1))
            + tf.reshape(
                tf.exp(tf.cast(self.__sum_ln_Tr_eps, tf.float32)),
                (self.__n_dis, 1),
            )
            - tf.reshape(tf.exp(ln_gamma[-1, :]), (self.__n_dis, 1))
        )
        return num - den

    def __update_Em(self, sum_ln_gamma, ln_gamma, obs):
        T = tf.size(obs)
        # num_list contains numerators such that Em[i,:] = num_list[i]/den
        num_list = []
        tmp_aux = np.ones(shape=(T, self.__n_dis), dtype=np.float32)
        tmp_aux = -inf * tmp_aux
        for i in range(self.__n_obs):
            num_list.append(
                tf.reduce_logsumexp(
                    tf.where(tf.equal(obs, i), ln_gamma, tmp_aux), axis=0
                )
            )
        del tmp_aux

        # then pseudocounts have to be added to num and den
        num = tf.log(
            tf.exp(tf.stack(num_list))
            + tf.exp(tf.cast(self.__ln_Em_eps, tf.float32))
        )
        den = tf.log(
            tf.exp(tf.reshape(sum_ln_gamma, (1, self.__n_dis)))
            + tf.reshape(
                tf.exp(tf.cast(self.__sum_ln_Em_eps, tf.float32)),
                (1, self.__n_dis),
            )
        )
        return num - den

    def __update_pi(self, ln_gamma_0):
        return ln_gamma_0

    def __init_graph(self):
        print("Graph initialization")
        self.__graph = tf.Graph()

        with self.__graph.as_default():
            Em = tf.placeholder(
                tf.float32, shape=(self.__n_obs, self.__n_dis), name="Em"
            )
            Tr = tf.placeholder(
                tf.float32, shape=(self.__n_dis, self.__n_dis), name="Tr"
            )
            pi = tf.placeholder(tf.float32, shape=self.__n_dis, name="pi")
            obs = tf.placeholder(tf.int32, shape=[None], name="Classes_obs")

            # forward procedure
            alpha_op = tf.contrib.eager.py_func(
                self.__forward_procedure,
                inp=[Tr, Em, pi, obs],
                Tout=tf.float32,
            )

            # backward procedure
            beta_op = tf.contrib.eager.py_func(
                self.__backward_procedure, inp=[Tr, Em, obs], Tout=tf.float32
            )

            # compute alpha*beta and sum(alpha*beta)
            [aux_op, sum_aux_op] = tf.contrib.eager.py_func(
                self.__aux,
                inp=[alpha_op, beta_op],
                Tout=[tf.float32, tf.float32],
            )

            # Compute sum_ln_xi
            xi_op = tf.contrib.eager.py_func(
                self.__update_xi,
                inp=[Tr, Em, beta_op, alpha_op, sum_aux_op, obs],
                Tout=tf.float32,
            )

            # Compute ln_gamma and sum_ln_gamma
            [gamma_op, sum_gamma_op] = tf.contrib.eager.py_func(
                self.__update_gamma,
                inp=[aux_op, sum_aux_op],
                Tout=[tf.float32, tf.float32],
            )

            # tf.contrib.eager.py_func(self.__check_norm_xi,
            # inp=[xi_op, gamma_op, sum_gamma_op],Tout=tf.float32,
            # name="Check_xi-gamma")

            # Update Em
            tf.contrib.eager.py_func(
                self.__update_Em,
                inp=[sum_gamma_op, gamma_op, obs],
                Tout=tf.float32,
                name="Em_op",
            )
            self.__check_norm_Em()

            # Update Tr
            tf.contrib.eager.py_func(
                self.__update_Tr,
                inp=[xi_op, sum_gamma_op, gamma_op],
                Tout=tf.float32,
                name="Tr_op",
            )

            # Update pi
            tf.contrib.eager.py_func(
                self.__update_pi,
                inp=[gamma_op[0]],
                Tout=tf.float32,
                name="pi_op",
            )

    def bw(
            self,
            classes_obs: np.ndarray,
            maxit: int = 1
            ) -> None:
        """Apply Baum-Welch algorithm to the Hmm
        
        :param classes_obs: 
            Observable classes used to find the parameters
            (:ref:`Tr <submapp.hmm.Hmm.Tr>`,
            :ref:`Em <submapp.hmm.Hmm.Em>`,
            :ref:`pi <submapp.hmm.Hmm.pi>`)
            that maximize the likelihood of the model
        :type classes_obs: np.ndarray[float]
        :param maxit: 
            Number of iteration of the Baum-Welch algorithm
            NOT IMPLEMENTED YET. 
            Defaults to 1
        :type maxit: int, optional
        :rtype: None
        """
        start = time()
        c_obs = classes_obs.astype(int)
        nb_inputs = c_obs.size

        if self.__graph is None:
            self.__init_graph()

        sess = tf.Session(graph=self.__graph)
        with self.__graph.as_default() as g:
            Em_op = g.get_tensor_by_name("Em_op" + ":0")
            Tr_op = g.get_tensor_by_name("Tr_op" + ":0")
            pi_op = g.get_tensor_by_name("pi_op" + ":0")
            Em = g.get_tensor_by_name("Em" + ":0")
            Tr = g.get_tensor_by_name("Tr" + ":0")
            pi = g.get_tensor_by_name("pi" + ":0")
            obs = g.get_tensor_by_name("Classes_obs" + ":0")
            # check_xi_gamma = g.get_operation_by_name("Check_xi-gamma")

            # un = sess.run(check_xi_gamma,feed_dict={
            #    Em: self.__Em, Tr: self.__Tr, pi: self.__pi, obs: c_obs})

            # Update Tr, Em and pi
            self.__Tr, self.__Em, self.__pi = sess.run(
                [Tr_op, Em_op, pi_op],
                feed_dict={
                    Em: self.__Em,
                    Tr: self.__Tr,
                    pi: self.__pi,
                    obs: c_obs,
                },
            )
            # self.__check_norm_Em()
            # self.__check_norm_Tr()
            # self.__check_norm_pi()

            sess.close()
        end = time()
        print(
            "Baum-Welch algorithm done in ",
            round(end - start, 2),
            " sec, using ",
            nb_inputs,
            " inputs",
        )

    def viterbi(self, classes_obs: np.ndarray) -> np.ndarray:
        """Viterbi algorithm: retrieve hidden data from observable data
        
        :param classes_obs: 
            *New* observable sequence from which the hidden sequence
            is inferred using the Viterbi algorithm
        :type classes_obs: np.ndarray[int]
        :return: 
            *New* hidden sequence inferred using the Viterbi
            algorithm
        :rtype: np.ndarray[int]
        """
        start = time()
        obs = classes_obs.astype(int)
        T = obs.size
        ln_prob = np.empty([T, self.__n_dis])
        path = np.empty([T, self.__n_dis], dtype=int)

        # initialization
        ln_prob[0] = self.__pi + self.__Em[obs[0]]
        path[0] = 0

        # recursion
        for t in range(1, T):
            path[t] = np.argmax(
                np.reshape(ln_prob[t - 1], (self.__n_dis, 1)) + self.__Tr,
                axis=0,
            )
            for j in range(self.__n_dis):
                k = path[t, j]
                ln_prob[t, j] = (
                    ln_prob[t - 1, k]
                    + self.__Tr[k, j]
                    + self.__Em[obs[t], j]
                    - log(4.0)
                )
        classes_dis = np.zeros(T, dtype=int)
        classes_dis[-1] = np.argmax(ln_prob[-1, :])
        # path backtracking
        for t in range(T - 2, -1, -1):
            classes_dis[t] = path[t + 1, classes_dis[t + 1]]

        end = time()
        print(
            "Viterbi algorithm done in ",
            round(end - start, 2),
            " sec, using ",
            T,
            " inputs",
        )

        return classes_dis


    def neighborhood(
            self, 
            sigma: float, 
            distance_matrix: np.ndarray = None,
            ) -> None:
        """
        “Propagate” the probability of a hidden class to its neighbours
        
        :param sigma: 
            Radius of the neighbouring function
        :type sigma: float
        :param distance_matrix: 
            Matrix of relative distance between each class in the SOM
            This matrix defines the "shape" of the SOM (eg. hexagonal,
            rectangular) and should be the same as the one used in the
            SOM!
        :type distance_matrix: 
            np.ndarray[float], shape = (
            :ref:`n_dis <submapp.hmm.Hmm.n_dis>`,
            :ref:`n_dis <submapp.hmm.Hmm.n_dis>`)
        :rtype: None

        .. seealso::

            - :ref:`smooth_transitions <submapp.hmm.Hmm.smooth_transitions>`        
        """

        old_Em = self.Em
        new_Em = np.zeros_like(old_Em)
        old_Tr = self.Tr
        new_Tr = np.zeros_like(self.Tr)
        if distance_matrix is None:
            print(
                "Warning you are using an arbitrary distance between classes:"
            )
            print("Distance[class_i,class_j] = abs(i-j)")
            distance_matrix = np.array(
                [
                    abs(i - j)
                    for i in range(self.__n_dis)
                    for j in range(self.__n_dis)
                ]
            ).reshape((self.__n_dis, self.__n_dis))

        for j in range(self.__n_dis):
            for k in range(self.__n_dis):
                dist = distance_matrix[j, k]
                new_Em[:, j] += (
                    exp(-dist / sigma) * old_Em[:, k]
                )
        self.Em = new_Em

        for j in range(self.__n_dis):
            for k in range(self.__n_dis):
                dist = distance_matrix[j, k]
                new_Tr[:, j] += (
                    exp(-dist / sigma) * old_Tr[:, k]
                )
        self.Tr = new_Tr


    def smooth_transitions(
            self, 
            sigma: float, 
            distance_matrix: np.ndarray = None,
            ) -> None:
        """
        Penalize (i.e decrease probability) long distance transitions 
        
        :param sigma: 
            radius of the neighbouring function
        :type sigma: float
        :param distance_matrix: 
            Matrix of relative distance between each class in the SOM
            This matrix defines the "shape" of the SOM (eg. hexagonal,
            rectangular) and should be the same as the one used in the
            SOM!
        :type distance_matrix: 
            np.ndarray[float], shape = (
            :ref:`n_dis <submapp.hmm.Hmm.n_dis>`,
            :ref:`n_dis <submapp.hmm.Hmm.n_dis>`)
        :rtype: None

        .. seealso::

            - :ref:`neighborhood <submapp.hmm.Hmm.neighborhood>`        
        """

        old_Tr = self.Tr
        new_Tr = np.zeros_like(self.Tr)
        if distance_matrix is None:
            print(
                "Warning you are using an arbitrary distance between classes:"
            )
            print("Distance[class_i,class_j] = abs(i-j)")
            distance_matrix = np.array(
                [
                    abs(i - j)
                    for i in range(self.__n_dis)
                    for j in range(self.__n_dis)
                ]
            ).reshape((self.__n_dis, self.__n_dis))

        for i in range(self.__n_dis):
            for j in range(self.__n_dis):
                dist = distance_matrix[i, j]
                new_Tr[i, j] = exp(-dist / sigma) * old_Tr[i, j]
        self.Tr = new_Tr


    def __check_norm_Em(self, tol=1e-2):
        norm = LA.norm(self.Em, ord=1, axis=0)
        norm = np.reshape(norm, (1, self.__n_dis))
        if np.any(norm > 1 + tol) or np.any(norm < 1 - tol):
            print("WARNING! norm of Em is not consistent: ", norm)
        return norm

    def __check_norm_Tr(self, tol=1e-2):
        norm = LA.norm(self.Tr, ord=1, axis=1)
        norm = np.reshape(norm, (self.__n_dis, 1))
        if np.any(norm > 1 + tol) or np.any(norm < 1 - tol):
            print("WARNING! norm of Tr is not consistent: ", norm)
        return norm

    def __check_norm_pi(self, tol=1e-2):
        norm = LA.norm(self.pi, ord=1)
        if np.any(norm > 1 + tol) or np.any(norm < 1 - tol):
            print("WARNING! norm of pi is not consistent: ", norm)
        return norm

    def __check_norm_alpha(self, alpha, tol=1e-2, ignore=True):
        norm = LA.norm(tf.exp(alpha), ord=1, axis=1)
        if np.any(norm > 1 + tol) or np.any(norm < -tol):
            if ignore:
                print("WARNING! norm of alpha is not consistent: ", norm)
            else:
                print("NORM DE ALPHA: ", norm)
                sys.exit("ERROR! norm of alpha is not consistent: ")
        else:
            print("Norm of alpha is Ok!")

    def __check_norm_beta(self, beta, tol=1e-2, ignore=True):
        norm = LA.norm(tf.exp(beta), ord=1, axis=1)
        if np.any(norm > 1 + tol) or np.any(norm < -tol):
            if ignore:
                print("WARNING! norm of beta is not consistent: ", norm)
            else:
                print("NORM DE BETA: ", norm)
                sys.exit("ERROR! norm of beta is not consistent: ")
        else:
            print("Norm of beta is OK!")

    def __check_norm_gamma(self, gamma, tol=1e-2, ignore=True):
        norm = LA.norm(tf.exp(gamma), ord=1, axis=1)
        if np.any(norm > 1 + tol) or np.any(norm < 1 - tol):
            if ignore:
                print("WARNING! norm of gamma is not consistent: ", norm)
            else:
                print("NORM DE GAMMA: ", norm)
                sys.exit("ERROR! norm of gamma is not consistent: ")
        else:
            print("Norm of gamma is OK!")

    def __check_norm_xi(self, sum_xi, gamma, sum_gamma, tol=1e-2, ignore=True):
        (T, _) = tf.shape(gamma)
        T = tf.cast(T, tf.float32)
        gamma = tf.exp(gamma)
        sum_gamma = tf.exp(sum_gamma)
        sum_xi = tf.exp(sum_xi)

        # sum_i sum_j sum_xi[i,j] = 1
        norm = np.sum(sum_xi) - (T - 1.0)
        if abs(norm) > (T - 1.0) * tol:
            if ignore:
                print("WARNING! norm_tot of sum_xi is not consistent: ", norm)
            else:
                print("NORM TOT DE SUM_XI: ", norm)
                sys.exit("ERROR! norm_tot of sum_xi is not consistent: ")
        else:
            print("Norm_tot of sum_xi is OK!")

        # sum_j sum_xi[:,j] = sum_gamma[:] - gamma[T-1,:] (for all i)
        norm = LA.norm(sum_xi, ord=1, axis=1) - sum_gamma + gamma[-1]
        if np.any(abs(norm) > tol):
            if ignore:
                print(
                    "WARNING! norm of sum_j sum_xi is not consistent: ", norm
                )
            else:
                print("NORM SUM_J DE SUM_XI: ", norm)
                sys.exit("ERROR! norm of sum_j sum_xi is not consistent: ")
        else:
            print("Norm of sum_j sum_xi is OK!")

        # sum_i sum_xi[i,:] = sum_gamma[:] - gamma[0,:] (for all j)
        norm = LA.norm(sum_xi, ord=1, axis=0) - sum_gamma + gamma[0]
        if np.any(abs(norm) > tol):
            if ignore:
                print(
                    "WARNING! norm of sum_i sum_xi is not consistent: ", norm
                )
            else:
                print("NORM SUM_I DE SUM_XI: ", norm)
                sys.exit("ERROR! norm of sum_i sum_xi is not consistent: ")
        else:
            print("Norm of sum_i sum_xi is OK!")
        return 1.0






    @property
    def Tr(self) -> np.ndarray:
        """Transition matrix

        Store the probability of getting from one hidden class
        (at instant ``t-1``) to another one (at instant ``t``)

        .. math::
        
            Tr[i,j] = p(c^{t}_{hid} = j | c^{t-1}_{hid} = i)
     
        with: 0 <= i,j <= 
        :ref:`n_dis <submapp.hmm.Hmm.n_dis>`    

        :rtype: np.ndarray[float], shape = (n_dis, n_dis)

        .. seealso::

            - :ref:`(Tr,Em,pi) <submapp.hmm.Hmm.model>` 
            - :ref:`pi <submapp.hmm.Hmm.pi>`        
            - :ref:`Em <submapp.hmm.Hmm.Em>`

        """
        return np.exp(self.__Tr)

    @property
    def Em(self) -> np.ndarray:
        """Emission matrix

        Store the probability of getting a observable class
        given a hidden class at the same instant ``t``

        .. math::
        
            Em[i,j] = p(c^{t}_{obs} = i | c^{t}_{hid} = j)
     
        with: 
        
        - 0 <= i <= 
          :ref:`n_obs <submapp.hmm.Hmm.n_obs>` 
        - 0 <= j <= 
          :ref:`n_dis <submapp.hmm.Hmm.n_dis>`      

        :rtype: np.ndarray[float], shape = (n_obs, n_dis)

        .. seealso::

            - :ref:`(Tr,Em,pi) <submapp.hmm.Hmm.model>` 
            - :ref:`Tr <submapp.hmm.Hmm.Tr>`        
            - :ref:`pi <submapp.hmm.Hmm.pi>`
        """
        return np.exp(self.__Em)

    @property
    def pi(self) -> np.ndarray:
        """Initialization matrix

        Store the probability of getting hidden classes 
        at instant ``0``

        .. math::
        
            pi[i] = p(c^{0}_{hid} = i)
     
        with: 
        
        - 0 <= i <= 
          :ref:`n_dis <submapp.hmm.Hmm.n_dis>`  

        :rtype: np.ndarray[float], shape = (n_dis)

        .. seealso::

            - :ref:`(Tr,Em,pi) <submapp.hmm.Hmm.model>` 
            - :ref:`Tr <submapp.hmm.Hmm.Tr>`        
            - :ref:`Em <submapp.hmm.Hmm.Em>`
        """
        return np.exp(self.__pi)

    @Tr.setter
    def Tr(self, new_Tr):
        if new_Tr[0, 0] < 0:
            new_Tr = np.exp(new_Tr)
        norm_Tr = LA.norm(new_Tr, ord=1, axis=1)
        norm_Tr = np.reshape(norm_Tr, (self.__n_dis, 1))
        self.__Tr = np.log(new_Tr / norm_Tr)

    @Em.setter
    def Em(self, new_Em):
        if new_Em[0, 0] < 0:
            new_Em = np.exp(new_Em)
        norm_Em = LA.norm(new_Em, ord=1, axis=0)
        norm_Em = np.reshape(norm_Em, (1, self.__n_dis))
        self.__Em = np.log(new_Em / norm_Em)

    @pi.setter
    def pi(self, new_pi):
        if new_pi[0] < 0:
            new_pi = np.exp(new_pi)
        norm_pi = LA.norm(new_pi, ord=1)
        if norm_pi == 0:
            new_pi = np.ones_like(new_pi)
            norm_pi = LA.norm(new_pi, ord=1)
        self.__pi = np.log(new_pi / norm_pi)

    @property
    def model(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Model ``(Tr, Em, pi)`` of the HMM

        The tuple (
        :ref:`Tr <submapp.hmm.Hmm.Tr>`,    
        :ref:`Em <submapp.hmm.Hmm.Em>`,
        :ref:`pi <submapp.hmm.Hmm.pi>`        
        ) defines all the probabilities required to build a HMM
        model. Once these 3 matrices are estimated one can find the
        most likely hidden state at instant ``t`` knowing the
        observable state at the same instant ``t`` and the previous
        hidden state at instant ``t-1`` (calculated recursively)

        The model has to be trained first. See:
        
        - :ref:`init_model <submapp.hmm.Hmm.init_model>`
        - :ref:`neighborhood <submapp.hmm.Hmm.neighborhood>`
        - :ref:`bw <submapp.hmm.Hmm.bw>`
        - :ref:`smooth_transitions <submapp.hmm.Hmm.smooth_transitions>`

        Once trained the
        :ref:`viterbi <submapp.hmm.Hmm.viterbi>` 
        algorithm can estimate the most likely hidden time series
        associated to an observable time series based on this model

        :rtype: 
            Tuple[np.ndarray[float], np.ndarray[float],\
            np.ndarray[float]]

        .. seealso::

            - :ref:`Tr <submapp.hmm.Hmm.Tr>`        
            - :ref:`Em <submapp.hmm.Hmm.Em>`
            - :ref:`pi <submapp.hmm.Hmm.pi>`

        """
        return (np.exp(self.__Tr), np.exp(self.__Em), np.exp(self.__pi))

    @property
    def ln_Tr_eps(self) -> np.ndarray:
        """Pseudo-counts associated to the Transition matrix
        
        :rtype: np.ndarray[float], shape = (n_dis, n_dis)

        .. seealso::

            - :ref:`Tr <submapp.hmm.Hmm.Tr>`        
            - :ref:`ln_Em_eps <submapp.hmm.Hmm.ln_Em_eps>`
            - :ref:`ln_pi_eps <submapp.hmm.Hmm.ln_pi_eps>`

        """
        return np.copy(self.__ln_Tr_eps)

    @property
    def ln_Em_eps(self) -> np.ndarray:
        """Pseudo-counts associated to the Emission matrix
        
        :rtype: np.ndarray[float], shape = (n_obs, n_dis)

        .. seealso::

            - :ref:`Em <submapp.hmm.Hmm.Em>`        
            - :ref:`ln_Tr_eps <submapp.hmm.Hmm.ln_Tr_eps>`
            - :ref:`ln_pi_eps <submapp.hmm.Hmm.ln_pi_eps>`

        """
        return np.copy(self.__ln_Em_eps)

    @property
    def ln_pi_eps(self) -> np.ndarray:
        """Pseudo-counts associated to the Initialization matrix
        
        :rtype: np.ndarray[float], shape = (n_dis)

        .. seealso::

            - :ref:`pi <submapp.hmm.Hmm.pi>`        
            - :ref:`ln_Em_eps <submapp.hmm.Hmm.ln_Em_eps>`
            - :ref:`ln_Tr_eps <submapp.hmm.Hmm.ln_Tr_eps>`

        """
        return np.copy(self.__ln_pi_eps)

    @ln_Tr_eps.setter
    def ln_Tr_eps(self, ln_Tr_eps):
        self.__ln_Tr_eps = np.copy(ln_Tr_eps)
        self.__sum_ln_Tr_eps = logsumexp(self.__ln_Tr_eps, axis=1)

    @ln_Em_eps.setter
    def ln_Em_eps(self, ln_Em_eps):
        self.__ln_Em_eps = np.copy(ln_Em_eps)
        self.__sum_ln_Em_eps = logsumexp(self.__ln_Em_eps, axis=0)

    @ln_pi_eps.setter
    def ln_pi_eps(self, ln_pi_eps):
        self.__ln_pi_eps = np.copy(ln_pi_eps)
        self.__sum_ln_pi_eps = logsumexp(self.__ln_pi_eps)


    @property
    def n_dis(self) -> int:
        """Total number of hidden classes
        
        :rtype: int

        .. seealso::

            - :ref:`n_obs <submapp.hmm.Hmm.n_obs>`     

        """
        return self.__n_dis

    @property
    def n_obs(self) -> int:
        """Total number of observable classes
        
        :rtype: int

        .. seealso::

            - :ref:`n_dis <submapp.hmm.Hmm.n_dis>`     

        """
        return self.__n_obs

