from math import exp,log,isnan,nan,inf,sqrt
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import time
import tensorflow as tf
import copy
import matplotlib.pyplot as plt
import seaborn as sns
from statistics import mean, variance, stdev


def eexp(x) :
    res = 0
    if isnan(x) :
        res = 0
    else : 
        res = exp(x)
    return(res)
        
    
def eln(x) :
    res = 0
    if x==0 :
        res = nan
    elif x > 0: 
        res = log(x)
    else :
        raise Exception('negative input error')
    return(res)
    
def elnsum(eln_x, eln_y) :
    res = 0
    if isnan(eln_x) or isnan(eln_y) :
        if isnan(eln_x) :
            res = eln_y
        else :
            res = eln_x
    else :
        if eln_x > eln_y :
            res = eln_x + eln(1 + exp(eln_y-eln_x))
        else :
            res = eln_y + eln(1 + exp(eln_x-eln_y))
    return(res)

def elndiff(eln_x, eln_y) :
    res = 0
    if isnan(eln_x) or isnan(eln_y) :
        if isnan(eln_y) :
            res = eln_x
        else :
            res = eln_y
    else :
        if eln_x > eln_y :
            res = eln_y + eln(-1 + exp(eln_x-eln_y))
        elif eln_x == eln_y :
            res = nan
        else :
            raise Exception('negative input error')
    return(res)


def elnprod(eln_x,eln_y) :
    res = 0
    if isnan(eln_x) or isnan(eln_y) :
        res = nan
    else :
        res = eln_x + eln_y
    return(res)

