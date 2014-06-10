# -*- coding: utf8 -*-
import numpy as np


class Node(object):
    """Class that implement the node of the parsing tree"""

    def __init__(self, word=None, label=None):
        if label < 0:
            label = None
        self.y = label
        if label is not None:
            self.ypred = np.ones(len(label)) / len(label)
        else:
            self.ypred = None
        self.X = None
        self.word = word
        self.parent = None
        self.childrens = []
        self.d = None  # Vecteur d'erreur

    def set_label(self, label):
        label = float(label)
        if label < 0 or label > 1:
            self.y = None
        else:
            self.y = np.zeros(2)
            self.y[1] = label
            self.y[0] = 1 - self.y[1]

    def have_label(self):
        return (self.y is not None)

    def cost(self):
        if self.have_label():
            return -np.sum(self.y * np.log(self.ypred))
        else:
            return 0

    @staticmethod
    def getSoftLabel(l):
        if l <= 0.2:
            label = 0
        elif 0.2 < l <= 0.4:
            label = 1
        elif 0.4 < l <= 0.6:
            label = 2
        elif 0.6 < l <= 0.8:
            label = 3
        else:
            label = 4
        return label

    def score_fine(self):
        if self.have_label():
            return Node.getSoftLabel(self.ypred[1]) == Node.getSoftLabel(self.y[1])
        else:
            return 0

    def score_binary(self, inc_neut=False):
        if self.have_label():
            return ((self.ypred[1] <= 0.5 and self.y[1] <= 0.5) or
                    (self.ypred[1] > 0.5 and self.y[1] > 0.5))\
                * (inc_neut or not (0.4 < self.y[1] <= 0.6))
        else:
            return 0

    def score_eps(self, eps):
        if self.have_label():
            return (abs(self.ypred[1] - self.y[1]) <= eps)
        else:
            return 0
