import numpy as np
from os import path

from Tree import Tree

DATASET = '../data'


def load(remove_leaf_label=False,
         remove_middle_label=False):
    print 'Load Trees...'
    with open(path.join(DATASET, 'STree.txt')) as f:
        trees = []
        for line in f.readlines():
            tree = line.split('|')
            tree = np.array(tree).astype(int)
            trees.append(tree)

    print 'Load Sentences...'
    with open(path.join(DATASET, 'SOStr.txt')) as f:
        sentences = []
        lexicon = set()
        for line in f.readlines():
            sent = line.strip().split('|')
            sentences.append(sent)
            lexicon = lexicon.union(sent)

    print 'Load data split'
    with open(path.join(DATASET, 'datasetSplit.txt')) as f:
        whichSet = []
        f.readline()
        for line in f.readlines():
            whichSet.append(int(line.strip().split(',')[1]))

    print 'Load Index...'
    with open(path.join(DATASET, 'dictionary.txt')) as f:
        index = {}
        for line in f.readlines():
            phrase = line.split('|')
            index[int(phrase[1])] = phrase[0]

    print 'Load Labels...'
    with open(path.join(DATASET, 'sentiment_labels.txt')) as f:
        f.readline()
        labels = {}
        for line in f.readlines():
            id_p, y = line.split('|')
            labels[index[int(id_p)]] = float(y)

    print 'Build Trees...'
    X_trees_train = []
    X_trees_dev = []
    X_trees_test = []
    for s, t, k in zip(sentences, trees, whichSet):
        if k == 1:
            X_trees_train.append(Tree(s, t, labels, rll=remove_leaf_label, rml=remove_middle_label))
        elif k == 2:
            X_trees_test.append(Tree(s, t, labels, rll=remove_leaf_label, rml=remove_middle_label))
        elif k == 3:
            X_trees_dev.append(Tree(s, t, labels, rll=remove_leaf_label, rml=remove_middle_label))
        else:
            raise(Exception('Erreur dans le parsing train/test/dev'))
    return lexicon, X_trees_train, X_trees_dev, X_trees_test, labels
