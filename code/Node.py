

class Node(object):
    """Class that implement the node of the parsing tree"""
    def __init__(self, word=None, label=None):
        super(Node, self).__init__()
        self.label = label
        self.X = None
        self.word = word
        self.parent = None
        self.childrens = []
