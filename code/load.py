# -*- coding: utf8 -*-
import numpy as np
from os import path

from Tree import Tree

DATASET = '../data'
if __name__ == '__main__':

    print 'Load Trees...'
    with open(path.join(DATASET, 'STree.txt')) as f:
        trees = []
        for line in f.readlines():
            tree = line.split('|')
            tree = np.array(tree).astype(int)
            trees.append(tree)

    print 'Load Sentences...'
    #Changement pour SOStr parce que encodage merdique sur l'autre
    '''
    with open(path.join(DATASET, 'datasetSentences.txt')) as f:
        f.readline()
        sentences = []
        lexicon = set()
        for line in f.readlines():
            sent = line.split('\t')[1]
            sent = sent.replace('\n', '').split(' ')
            sentences.append(sent)
            lexicon = lexicon.union(sent)
    '''
    with open(path.join(DATASET, 'SOStr.txt')) as f:
        sentences = []
        lexicon = set()
        for line in f.readlines():
            sent = line.strip().split('|')
            sentences.append(sent)
            lexicon = lexicon.union(sent)
            
    print 'Load data split'
    with open(path.join(DATASET,'datasetSplit.txt')) as f:
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
        if k==1:
            X_trees_train.append(Tree(s, t, labels))
        elif k==2:
            X_trees_test.append(Tree(s,t,labels))
        elif k==3:
            X_trees_dev.append(Tree(s,t,labels))
        else:
            raise(Exception('Erreur dans le parsing train/test/dev'))
    '''
    Deja dans l'init du RNN
    vocab = {}
    for i, w in enumerate(lexicon):
        vocab[w] = i
    '''
    from RNN import RNN
    model = RNN(vocab=lexicon)

    #rpz = model.compute(X_trees[0])
    #print rpz
