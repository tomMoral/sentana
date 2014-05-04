# -*- coding: utf8 -*-
import numpy as np


class Node(object):
    """Class that implement the node of the parsing tree"""
    def __init__(self, word=None, label=None):
        self.y = label
        if label is not None:
            self.ypred = np.ones(len(label))/len(label)
        else:
            self.ypred = None
        self.X = None
        self.word = word
        self.parent = None
        self.childrens = []
        self.d = None  # Vecteur d'erreur
