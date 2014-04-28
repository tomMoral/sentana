import numpy as np


class RNN(object):
    """Class to use Recursive Neural Network on Tree

    Usage
    -----


    Methods
    -------

    """
    def __init__(self, dim, vocab, r=5):
        self.dim = dim

        #Initiate V, the tensor operator
        self.V = 1e-5*np.ones((dim, 2*dim, 2*dim))

        #Initiate W, the linear operator
        self.W = 1e-5*np.ones((dim, 2*dim))

        #Initiate Ws, the linear operator
        self.Ws = 1e-5*np.ones((5, 2*dim))

        #Initiate L, the Lexicon representation
        self.L = np.random.uniform(-r, r, size=(len(vocab), dim))
        self.vocab = {}
        for i, w in enumerate(vocab):
            self.vocab[w] = i

        self.f = lambda X: np.tanh(X.T.dot(self.V).dot(X) + self.W.dot(X))
        self.grad = lambda x: x

        self.y = lambda x: max(np.exp(x) / np.exp(x).sum())

    def compute(self, X_tree):
        for n in X_tree.leaf:
            n.X = self.L[self.vocab[n.word]]

        for p, [a, b] in X_tree.parcours:
            aT = X_tree.nodes[a]
            bT = X_tree.nodes[b]
            pT = X_tree.nodes[p]
            X = np.append(aT.X, bT.X).reshape((-1, 1))
            pT.X = self.f(X)

        E = sum([(self.y(n.X) - n.y) for n in X_tree.nodes])
        print E

        return self.Ws.dot(pT.X)
