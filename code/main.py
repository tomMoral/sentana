# -*- coding: utf8 -*-

DATASET = '../data/stanfordSentimentTreebank'
if __name__ == '__main__':

    import argparse
    import os

    #print os.environ['PATH']

    parser = argparse.ArgumentParser('Run a RNN sentiment tree analysis'
                                     ' over the stanfor tree bank')
    parser.add_argument('--strat', type=str, default='AdaGrad',
                        help='Strategie for the learning. {AdaGrad, Rmsprop}')
    parser.add_argument('--epoch', type=int, default=100,
                        help='Nb max d\'epoch {default: 100')
    parser.add_argument('--bin', action='store_true',
                        help='Perform a binary classification')
    parser.add_argument('--mb_size', type=int, default=27,
                        help='Size of the mini-batch')
    parser.add_argument('--reset_freq', type=int, default=-1,
                        help='Frequence of reset for adagrad')
    parser.add_argument('--n_stop', type=int, default=8,
                        help='Threshold for early stopping -1 to avoid ES')
    parser.add_argument('--reg', type=float, default=1,
                        help='Regularisation factor scaling')
    parser.add_argument('--n_check', type=float, default=32,
                        help='Frequence of checking early stopping')
    parser.add_argument('--lr', type=float, default=0.01,
                        help='Learning rate')
    parser.add_argument('--save_tmp', type=str, default='tmp.pkl',
                        help='Tmp file save')
    parser.add_argument('--model_dump', type=str, default='model/model',
                        help='Where to dump the succesives model of the NN')
    parser.add_argument('--wroot', type=int, default=1,
                        help='Relative weight of the root compared \
                        to the other nodes')
    parser.add_argument('--rae', action='store_true',
                        help='Use the RAE')

    args = parser.parse_args()

    import cPickle as pickle
    if not os.path.exists('trees.pkl'):
        from load import load
        lexicon, X_trees_train, X_trees_dev, X_trees_test, lab = load()
        with open('trees.pkl', 'wb') as output:
            pickle.dump((lexicon, X_trees_train, X_trees_dev, X_trees_test, lab), output, -1)
    else:
        with open('trees.pkl', 'rb') as input:
            lexicon, X_trees_train, X_trees_dev, X_trees_test, lab = pickle.load(input)

    if args.rae:
        from RAE import RAE
        model = RAE(vocab=lexicon, reg=args.reg)
        l1, l2 = model.train([t for t in X_trees_train if t.getDepth() == 2],
                             max_epoch=args.epoch,
                             val_set=X_trees_dev,
                             strat=args.strat,
                             mini_batch_size=args.mb_size,
                             reset_freq=args.reset_freq,
                             save_tmp=args.save_tmp,
                             n_stop=args.n_stop,
                             w_root=args.wroot,
                             learning_rate=args.lr)
    else:
        from RNN import RNN
        import numpy as np
        model = RNN(vocab=lexicon, reg=args.reg)
        warm_start = False
        maxDepth = max([t.getDepth() for t in X_trees_train])
        minDepth = 2
        #minDepth = maxDepth
        #for depth in range(minDepth, maxDepth + 1):
        for depth in [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]:
            print "=====> depth", depth
            train_set = [t for t in X_trees_train if t.getDepth() <= depth]
            dev_set = [t for t in X_trees_dev if t.getDepth() <= depth]
            # train_set = np.concatenate((set, [t for t in X_trees_dev if t.getDepth() <= depth]))
            l1, l2 = model.train(train_set,
                                 max_epoch=args.epoch,
                                 val_set=dev_set,
                                 strat=args.strat,
                                 mini_batch_size=args.mb_size,
                                 reset_freq=args.reset_freq,
                                 save_tmp=args.save_tmp,
                                 n_stop=args.n_stop,
                                 modelPath=args.model_dump,
                                 learning_rate=args.lr,
                                 warm_start=warm_start)

            model.save('../data/exp{0}'.format(depth))

            if depth >= 10:
                sa_trn, sr_trn = model.score_fine(X_trees_train)
                sa_val, sr_val = model.score_fine(X_trees_dev)
                sa_tst, sr_tst = model.score_fine(X_trees_test)
                print 'Fine grain\tTrain\tTest\tValidation'
                print 'Overall\t\t{:.3}\t{:.3}\t{:.3}'.format(sa_trn, sa_tst, sa_val)
                print 'Root\t\t{:.3}\t{:.3}\t{:.3}'.format(sr_trn, sr_tst, sr_val)

                sa_trn, sr_trn = model.score_binary(X_trees_train)
                sa_val, sr_val = model.score_binary(X_trees_dev)
                sa_tst, sr_tst = model.score_binary(X_trees_test)
                print 'Binary\tTrain\tTest\tValidation'
                print 'Overall\t\t{:.3}\t{:.3}\t{:.3}'.format(sa_trn, sa_tst, sa_val)
                print 'Root\t\t{:.3}\t{:.3}\t{:.3}'.format(sr_trn, sr_tst, sr_val)
            warm_start = True

    colors = {}
    n_gram = {}
    for x in X_trees_test:
        for N in x.nodes:
            n = len(N.word.split(' '))
            l = n_gram.get(n, [])
            l.append(N.X)
            n_gram[n] = l
            y = lab[N.word]
            l = colors.get(n, [])
            if y < 0.2:
                l.append(0)
            elif y < 0.4:
                l.append(1)
            elif y < 0.6:
                l.append(2)
            elif y < 0.8:
                l.append(3)
            else:
                l.append(4)
            colors[n] = l

    for k in n_gram.keys():
        n_gram[k] = np.array(n_gram[k])
