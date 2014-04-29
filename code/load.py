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
    with open(path.join(DATASET, 'datasetSentences.txt')) as f:
        f.readline()
        sentences = []
        lexicon = set()
        for line in f.readlines():
            sent = line.split('\t')[1]
            sent = sent.replace('\n', '').split(' ')
            sentences.append(sent)
            lexicon = lexicon.union(sent)

    print 'Load Index...'
    with open(path.join(DATASET, 'dictionary.txt')) as f:
        index = {}
        lexicon = set()
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
    X_trees = []
    for s, t in zip(sentences, trees):
        X_trees.append(Tree(s, t, labels))

    vocab = {}
    for i, w in enumerate(lexicon):
        vocab[w] = i

    from RNN import RNN
    model = RNN(dim=20, vocab=vocab)

    rpz = model.compute(X_trees[0])
    print rpz
