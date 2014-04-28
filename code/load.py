import numpy as np
from os import path

from Tree import Tree

DATASET = '../data'
if __name__ == '__main__':
    with open(path.join(DATASET, 'STree.txt')) as f:
        trees = []
        for line in f.readlines():
            tree = line.split('|')
            tree = np.array(tree).astype(int)
            trees.append(tree)

    with open(path.join(DATASET, 'datasetSentences.txt')) as f:
        sent1 = f.readline()
        sentences = []
        lexicon = set()
        for line in f.readlines():
            sent = line.split('\t')[1]
            sent = sent.replace('\n', '').split(' ')
            sentences.append(sent)
            lexicon = lexicon.union(sent)

    X_trees = []
    for s, t in zip(sentences, trees):
        X_trees.append(Tree(s, t))

    vocab = {}
    for i, w in enumerate(lexicon):
        vocab[w] = i

    from RNN import RNN
    model = RNN(dim=20, vocab=vocab)

    rpz = model.compute(X_trees[0])
    print rpz
