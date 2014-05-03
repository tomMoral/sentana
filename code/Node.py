# -*- coding: utf8 -*-
import numpy as np

class Node(object):
    """Class that implement the node of the parsing tree"""
    def __init__(self, word=None, label=None):
        super(Node, self).__init__() #??
        self.y = label
        if label!=None:
            self.ypred= np.ones(len(label))/len(label)
        else:
            self.ypred=None
        self.X = None
        self.word = word
        self.parent = None
        self.childrens = []
        self.d=None #Vecteur d'erreur
