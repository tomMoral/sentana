# -*- coding: utf8 -*-
import numpy as np
from os import path

from Tree import Tree

DATASET = '../data'
if __name__ == '__main__':

    import argparse

    parser = argparse.ArgumentParser('Run a RNN sentiment tree analysis'
                                     ' over the stanfor tree bank')
    parser.add_argument('--strat', type=str, default='AdaGrad',
                        help='Strategie for the learning. {AdaGrad, Rmsprop}')
    parser.add_argument('--iter', type=int, default=400,
                        help='Nb max d\'iteration {default: 400')
    parser.add_argument('--bin', action='store_true',
                        help='Perform a binary classification')
    parser.add_argument('--mb_size', type=int, default=27,
                        help='Size of the mini-batch')
    parser.add_argument('--reset_freq', type=int, default=-1,
                        help='Frequence of reset for adagrad')

    parser.add_argument('--save_tmp',type=str, default='tmp.pkl',
                        help='Tmp file save')

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
            X_trees_train.append(Tree(s, t, labels, binl=args.bin))
        elif k == 2:
            X_trees_test.append(Tree(s, t, labels, binl=args.bin))
        elif k == 3:
            X_trees_dev.append(Tree(s, t, labels, binl=args.bin))
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
    l1, l2 = model.train(X_trees_train, max_iter=args.iter, val_set=X_trees_dev,
                         strat=args.strat, mini_batch_size=args.mb_size,
                         reset_freq=args.reset_freq,save_tmp=args.save_tmp)

    model.save('../data/exp1')

    sa_trn, sr_trn = model.score_fine(X_trees_train)
    sa_val, sr_val = model.score_fine(X_trees_dev)
    sa_tst, sr_tst = model.score_fine(X_trees_test)
    print 'Fine grain\tTrain\tTest\tValidation'
    print 'Overall\t\t{:.3}\t{:.3}\t{:.3}'.format(sa_trn, sa_tst, sa_val)
    print 'Root\t\t{:.3}\t{:.3}\t{:.3}'.format(sr_trn, sr_tst, sr_val)
