# -*- coding: utf8 -*-
import numpy as np
from os import path

from Tree import Tree

DATASET = '../data'
if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser('Perform a GridSearch for this model')
    parser.add_argument('--reg', action='store_true',
                        help='For the regularisation parameter')
    parser.add_argument('--lr', action='store_true',
                        help='For the learning rate')
    parser.add_argument('--mb', action='store_true',
                        help='For the mini batch size')
    args = parser.parse_args()

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
            X_trees_train.append(Tree(s, t, labels))
        elif k == 2:
            X_trees_test.append(Tree(s, t, labels))
        elif k == 3:
            X_trees_dev.append(Tree(s, t, labels))
        else:
            raise(Exception('Erreur dans le parsing train/test/dev'))
    '''
    Deja dans l'init du RNN
    vocab = {}
    for i, w in enumerate(lexicon):
        vocab[w] = i
    '''
    from RNN import RNN
    curve = []
    if args.reg:
        for reg in np.logspace(-2, 3, 10):
            model = RNN(vocab=lexicon, reg=reg)
            l1, l2 = model.train(X_trees_train, max_iter=1000, val_set=X_trees_dev,
                                 strat='AdaGrad', mini_batch_size=30)
            curve.append(l2)
        np.save('reg_curve', curve)
    elif args.mb:
        for mb in np.linspace(20, 50, 10):
            model = RNN(vocab=lexicon, reg=1)
            l1, l2 = model.train(X_trees_train, max_iter=1000, val_set=X_trees_dev,
                                 strat='AdaGrad', mini_batch_size=mb)
            curve.append(l2)
        np.save('mb_curve', curve)
    else:
        for lr in np.logspace(-2, 3, 10):
            model = RNN(vocab=lexicon, reg=1)
            l1, l2 = model.train(X_trees_train, max_iter=1000, val_set=X_trees_dev,
                                 strat='AdaGrad', mini_batch_size=30, learning_rate=lr)
            curve.append(l2)
        np.save('lr_curve', curve)

    sa_trn, sr_trn = model.score_fine(X_trees_train)
    sa_val, sr_val = model.score_fine(X_trees_dev)
    sa_tst, sr_tst = model.score_fine(X_trees_test)
    print 'Fine grain\tTrain\tTest\tValidation'
    print 'Overall\t\t{:.3}\t{:.3}\t{:.3}'.format(sa_trn, sa_tst, sa_val)
    print 'Root\t\t{:.3}\t{:.3}\t{:.3}'.format(sr_trn, sr_tst, sr_val)
