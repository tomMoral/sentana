from Node import Node

#Define class parameter
r = 0.0001


class Tree(object):
    """docstring for Tree"""
    def __init__(self, sentence, structure):
        self.sentence = sentence
        self.structure = structure

        wc = len(sentence)

        self.nodes = []
        self.leaf = []

        for i in range(2*wc-1):
            self.nodes.append(Node())

        for i, w in enumerate(sentence):
            node = self.nodes[i]
            node.word = w
            self.leaf.append(node)

        parc = {}
        for i, (n, p) in enumerate(zip(self.nodes, structure)):
            n.parent = p-1
            self.nodes[p-1].childrens.append(i)
            l = parc.get(p-1, [])
            l.append(i)
            parc[p-1] = l
        self.parcours = parc.items()[:-1]
