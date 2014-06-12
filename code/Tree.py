# -*- coding: utf8 -*-
from Node import Node


class Tree(object):

    """docstring for Tree"""

    def __init__(self, sentence, structure, label=None,
                 rll=False,
                 rml=False):
        self.sentence = sentence
        self.structure = structure

        wc = len(sentence)

        self.nodes = []
        self.leaf = []

        for i in range(2 * wc - 1):
            self.nodes.append(Node())

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

        if label is not None:
            for p, [a, b] in self.parcours:
                aT = self.nodes[a]
                bT = self.nodes[b]
                pT = self.nodes[p]
                if aT.order < bT.order:
                    pT.word = ' '.join([aT.word, bT.word])
                else:
                    pT.word = ' '.join([bT.word, aT.word])
                pT.set_label(label.get(pT.word))

                if rml:
                    aT.y = None
                    bT.y = None
                pT.order = aT.order

            if not rll:
                for n in self.leaf:
                    n.set_label(label.get(n.word))
