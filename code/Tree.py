# -*- coding: utf8 -*-
from Node import Node


class Tree(object):

    """docstring for Tree"""

    def __init__(self, sentence, structure, label=None):
        self.sentence = sentence
        self.structure = structure

        wc = len(sentence)
        self.word_count = wc

        self.nodes = []
        self.leaf = []

        for i in range(2 * wc - 1):
            self.nodes.append(Node(isLeaf=i < wc))

        self.root = self.nodes[-1]

        for i, w in enumerate(sentence):
            node = self.nodes[i]
            node.word = w
            node.order = i
            self.leaf.append(node)

        parc = {}
        for i, (n, p) in enumerate(zip(self.nodes, structure)):
            n.parent = p - 1
            self.nodes[p - 1].childrens.append(i)
            l = parc.get(p - 1, [])
            l.append(i)
            parc[p - 1] = l
        parc.pop(-1)
        self.parcours = parc.items()
        self.parcours.sort()

        def sortAB(p, (a, b)):
            if(l[0] < l[1]):
                return (p, (a, b))
            else:
                return (p, (b, a))

        self.parcours.map(sortAB)

        self.size = len(structure)
        self.weight = 0

        if label is not None:
            for p, [a, b] in self.parcours:
                aT = self.nodes[a]
                bT = self.nodes[b]
                pT = self.nodes[p]
                # aT.order < bT.order:
                pT.word = ' '.join([aT.word, bT.word])
                pT.set_label(label.get(pT.word))
                pT.order = aT.order

    def strip_labels(self, leaf=False, middle=False, root=False):
        for p, [a, b] in self.parcours:
            if middle:
                # a < self.word_count == a.isLeaf
                if leaf or a >= self.word_count:
                    self.nodes[a].strip_label()
                if leaf or b >= self.word_count:
                    self.nodes[b].strip_label()
        if root:
            self.root.strip_label()

    def getRootIndex(self):
        result = self.nodes[-1]
        if result.parent != -1:
            raise Exception("oops")
        return len(self.nodes) - 1

    def getDepth(self, node=None):
        if node is None:
            node = self.getRootIndex()
        result = 0
        for child in self.root.childrens:
            if child != node:
                result = max(result, self.getDepth(child))
        return result + 1

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
