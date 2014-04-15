import numpy as np

import theano
import theano.tensor as T


class RNN(object):
    """Class to use Recursive Neural Network on Tree

    Usage
    -----


    Methods
    -------

    """
    def __init__(self, dim, K):
        self.dim = dim
        self.K = K

        #Initiate V, the tensor operator
        self.V = np.zeros((2*dim, 2*dim, dim))

        #Initiate W, the linear operator
        W_0 = np.zeros((dim, 2*dim))

    def compute(self, X_tree):
        for p1, a, b in X_tree.parcours():
            X = a.X
