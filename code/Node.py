import numpy as np

#Define class parameter
r = 0.0001


class Node(object):
    """Class that implement the node of the parsing tree"""
    def __init__(self, label, d):
        super(Node, self).__init__()
        self.label = label
        self.X = np.random.uniform(-r, r, size=(d, 1))
