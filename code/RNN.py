# -*- coding: utf8 -*-
import numpy as np
import pickle
from itertools import ifilter, imap
from Node import Node
from math import sqrt


class RNN(object):

    """Class to use Recursive Neural Network on Tree

    Usage
    -----


    Methods
    -------

    """

    def __init__(self, vocab={}, dim=30, r=0.0001, reg=1):
        self.dim = dim

        # TODO improve initialisation --> check ?
        # TODO add bias
        # Initiate V, the tensor operator
        self.V = init_tensor(dim, 2 * dim, 2 * dim, r=r)

        # Initiate W, the linear operator
        self.W = init_matrix(dim, 2 * dim, bias=True)

        # Initiate Ws, the sentiment linear operator
        self.Ws = init_matrix(2, dim, bias=True)
        print self.Ws.shape

        # Regularisation
        self.regV = reg * 0.001
        self.regW = reg * 0.001
        self.regWs = reg * 0.0001
        self.regL = reg * 0.0001

        # Initiate L, the Lexicon representation
        self.L = init_matrix(len(vocab), dim, bias=False)
        self.vocab = {}
        for i, w in enumerate(vocab):
            self.vocab[w] = i

        self.f = lambda X: np.tanh(self.combine_with_bias(X))
        self.grad = lambda f: 1 - f ** 2

        self.y = lambda x: self.sentiment_with_bias(x)

    def combine(self, X):
        return X.T.dot(self.V).dot(X) + self.W.dot(X)

    def combine_with_bias(self, X):
        return X.T.dot(self.V).dot(X) + self.W[::, :-1].dot(X) + self.W[::, -1]

    def sentiment(self, x):
        y = np.exp(self.Ws.dot(x).clip(-500, 700))
        return y / np.sum(y)

    def sentiment_with_bias(self, x):
        y = np.exp((self.Ws[::, :-1].dot(x) + self.Ws[::, -1]).clip(-500, 700))
        return y / np.sum(y)

    def save(self, saveFile):
        with open(saveFile, 'wb') as output:
            pickle.dump(self.dim, output, -1)
            pickle.dump(self.V, output, -1)
            pickle.dump(self.W, output, -1)
            pickle.dump(self.Ws, output, -1)
            pickle.dump(self.regV, output, -1)
            pickle.dump(self.regW, output, -1)
            pickle.dump(self.regWs, output, -1)
            pickle.dump(self.regL, output, -1)
            pickle.dump(self.L, output, -1)
            pickle.dump(self.vocab, output, -1)

    def load(self, loadFile):
        with open(loadFile, 'rb') as input:
            self.dim = pickle.load(input)
            self.V = pickle.load(input)
            self.W = pickle.load(input)
            self.Ws = pickle.load(input)
            self.regV = pickle.load(input)
            self.regW = pickle.load(input)
            self.regWs = pickle.load(input)
            self.regL = pickle.load(input)
            self.L = pickle.load(input)
            self.vocab = pickle.load(input)

    def error(self, val_set):
        '''
        Calcule l'erreur moyenne sur le set val_set avec les parametres actuel
        '''
        errorVal = 0.0
        for X_tree in val_set:
            errorVal += self.forward_pass(X_tree)
        return errorVal / len(val_set)

    def forward_pass(self, X_tree):
        '''
        Effectue la passe forward
        et calcule l'erreur actuelle pour l'arbre X_tree
        '''
        errorVal = 0.0
        for n in X_tree.leaf:
            # Met a jour le mot avec le Lexicon courant
            n.X = self.L[self.vocab[n.word]]
            n.ypred = self.y(n.X)  # Mise a jour du label predit
            n.ypred += 1e-300 * (n.ypred == 0)
            assert (n.ypred != 0).all()
            if n.cost() < 0:
                print "cost: %4f, ypred: %4f, y: %4f" % (n.cost(), n.ypred[1], n.y[1])
            errorVal += n.cost()

        for p, [a, b] in X_tree.parcours:
            # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            aT = X_tree.nodes[a]
            bT = X_tree.nodes[b]
            pT = X_tree.nodes[p]
            if aT.order < bT.order:
                X = np.append(aT.X, bT.X)
            else:
                X = np.append(bT.X, aT.X)
            pT.X = self.f(X)  # Mise a jour du descripteur du parent
            pT.ypred = self.y(pT.X)  # Mise a jour du label predit
            pT.ypred += 1e-300 * (pT.ypred == 0)
            errorVal += pT.cost()
            if pT.cost() < 0:
                print "cost: %4f, ypred: %4f, y: %4f" % (pT.cost(), pT.ypred[1], pT.y[1])
        #E = sum([(self.y(n.X) - n.y) for n in X_tree.nodes])
        # print E
        # return self.Ws.dot(pT.X) -> Pas besoin de retourner le lbel, il faut
        # le maj aussi?
        return errorVal

    def backward_pass(self, X_tree, root=False):
        '''
        Retourne le gradient du a l'erreur commise sur l'arbre X_tree
        Attention: suppose la forward_pass faite
        '''
        # Initialise les gradients
        dWs = np.zeros(self.Ws.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        dL = np.zeros(self.L.shape)

        # Initialise les deltas
        for n in X_tree.nodes:
            if n.have_label():
                n.d = self.Ws.T.dot(n.ypred - n.y)[:self.dim]
            else:
                n.d = np.zeros(self.dim)

        # Descend dans l arbre
        for p, [a, b] in X_tree.parcours[::-1]:
            # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            aT = X_tree.nodes[a]
            bT = X_tree.nodes[b]
            pT = X_tree.nodes[p]
            # Propagation des erreurs vers le bas
            gX = self.grad(pT.X)
            if aT.order < bT.order:  # Si aT est le noeud de gauche
                X = np.append(aT.X, bT.X)
                ddown = (
                    self.W.T.dot(pT.d * gX)[:self.dim]
                    + 2 * (pT.d * gX).dot(self.V.dot(X)))
                aT.d += ddown[:self.dim]
                bT.d += ddown[self.dim:]
            else:  # aT est a droite
                X = np.append(bT.X, aT.X)
                ddown = (
                    self.W.T.dot(pT.d * gX)[:self.dim]
                    + 2 * (pT.d * gX).dot(self.V.dot(X)))
                aT.d += ddown[self.dim:]
                bT.d += ddown[:self.dim]
            # Contribution aux gradients du pT
            if pT.have_label():
                # if (pT.y is None) or (pT.ypred is None):
                #     print "ypred: %4f, y: %4f" % (pT.ypred[1], pT.y[1])
                dWs += np.outer(pT.ypred - pT.y, pT.X)
            dV += np.tensordot(pT.d * gX, np.outer(X, X), axes=0)
            dW += np.outer(pT.d * gX, X)

        # Contribution des feuilles
        for n in X_tree.leaf:
            dL[self.vocab[n.word]] += n.d
            if n.have_label():
                dWs += np.outer(n.ypred - n.y, n.X)

        return dWs, dV, dW, dL

    #TODO use the w_root argument ??
    def train(self, X_trees, learning_rate=0.01, mini_batch_size=27,
              warm_start=False, r=0.0001, max_epoch=1000, val_set=[],
              n_check=8, strat='AdaGrad',
              bin=False, reset_freq=-1,

              modelPath='model/model',
              save_tmp='tmp.pkl', n_stop=4):
        '''
        Training avec AdaGrad (Dutchi et al.),
        prends en entrée une liste d'arbres X_trees
        '''
        # Remise à zero du modele
        if not warm_start:
            print 'reseting the RNN'
            dim = self.dim
            self.V = init_tensor(dim, 2 * dim, 2 * dim, r=r)
            self.W = init_matrix(dim, 2 * dim, bias=True)
            self.Ws = init_matrix(2, dim, bias=True)
            self.L = init_matrix(len(self.vocab), dim, bias=False)

        self.printPerformanceSummary(val_set)
        self.print_confusion_matrix(val_set)

        # Liste pour erreurs
        errMB = []
        errVal = []

        # Condition d'arret
        n_epoch = 1
        gradNorm = 1.0
        if val_set != []:
            prevError = self.error(val_set)
            prevError += self.regWs * np.sum(self.Ws * self.Ws) / 2.0
            prevError += self.regW * np.sum(self.W * self.W) / 2.0
            prevError += self.regV * np.sum(self.V * self.V) / 2.0
            prevError += self.regL * np.sum(self.L * self.L) / 2.0
            iniError = prevError
            minError = prevError  # optimal error so far
            glError = 0
            upStop = 0
            errVal.append(prevError)

        # Normalisation pour AdaGrad/Rms-prop
        eta = learning_rate
        dWsHist = np.zeros(self.Ws.shape)
        dVHist = np.zeros(self.V.shape)
        dWHist = np.zeros(self.W.shape)
        dLHist = np.zeros(self.L.shape)

        # Adaptative LR for RMSprop
        dWsMask = np.ones(self.Ws.shape)
        dVMask = np.ones(self.V.shape)
        dWMask = np.ones(self.W.shape)
        dLMask = np.ones(self.L.shape)

        dWsPrev = np.zeros(self.Ws.shape)
        dVPrev = np.zeros(self.V.shape)
        dWPrev = np.zeros(self.W.shape)
        dLPrev = np.zeros(self.L.shape)

        early_stop = False
        n_trees = len(X_trees)
        learn_by_epoch = True

        if learn_by_epoch:
            mini_batch_numbers = (n_trees + mini_batch_size - 1) / mini_batch_size
            print "nb of batchs %i, batch size %i"\
                % (mini_batch_numbers, mini_batch_size)
        else:
            mini_batch_numbers = 1

        while (not early_stop) and n_epoch < max_epoch:
            print "Starting epoch %i" % n_epoch

            if learn_by_epoch:
                X_trees_perm = np.random.permutation(X_trees)
                print "Permutation computed"

            # Reset adagrad rates cf Socher
            if n_epoch % reset_freq == 0 and reset_freq > 0:
                dWsHist = np.zeros(self.Ws.shape)
                dVHist = np.zeros(self.V.shape)
                dWHist = np.zeros(self.W.shape)
                dLHist = np.zeros(self.L.shape)

            for k in range(mini_batch_numbers):
                #print "Starting round %i" % k
                if learn_by_epoch:
                    #take mini batch according to the current permutation
                    mini_batch_samples = X_trees_perm[
                        k * mini_batch_size:
                        min(n_trees, (k + 1) * mini_batch_size)
                    ]
                else:
                    # Choose mini batch randomly
                    mini_batch_samples = np.random.choice(
                        X_trees, size=mini_batch_size)

                # Initialize gradients to 0
                dWsCurrent = np.zeros(self.Ws.shape)
                dVCurrent = np.zeros(self.V.shape)
                dWCurrent = np.zeros(self.W.shape)
                dLCurrent = np.zeros(self.L.shape)

                # Mini batch pour gradient
                currentMbe = 0.0
                for X_tree in mini_batch_samples:
                    currentMbe += self.forward_pass(X_tree)
                    dWs, dV, dW, dL = self.backward_pass(X_tree)
                    dWsCurrent += dWs
                    dVCurrent += dV
                    dWCurrent += dW
                    dLCurrent += dL

                currentMbe /= mini_batch_size
                currentMbe += self.regWs * np.sum(self.Ws * self.Ws) / 2.0
                currentMbe += self.regW * np.sum(self.W * self.W) / 2.0
                currentMbe += self.regV * np.sum(self.V * self.V) / 2.0
                currentMbe += self.regL * np.sum(self.L * self.L) / 2.0

                # Division par le nombre de sample + regularisation
                dWsCurrent = dWsCurrent / mini_batch_size + self.regWs * self.Ws
                dVCurrent = dVCurrent / mini_batch_size + self.regV * self.V
                dWCurrent = dWCurrent / mini_batch_size + self.regW * self.W
                dLCurrent = dLCurrent / mini_batch_size + self.regL * self.L

                # Mise a jour des poids et calcul des pas
                if strat == 'AdaGrad':
                    eps = 0.001  # Adagrad >0 ? cf Socher
                    dWsHist += dWsCurrent * dWsCurrent
                    dVHist += dVCurrent * dVCurrent
                    dWHist += dWCurrent * dWCurrent
                    dLHist += dLCurrent * dLCurrent

                    dWsCurrent = eta * dWsCurrent / np.sqrt(dWsHist + eps)
                    dWCurrent = eta * dWCurrent / np.sqrt(dWHist + eps)
                    dVCurrent = eta * dVCurrent / np.sqrt(dVHist + eps)
                    dLCurrent = eta * dLCurrent / np.sqrt(dLHist + eps)
                else:
                    eps = 0.001
                    dWsHist = 0.9 * dWsHist + 0.1 * dWsCurrent * dWsCurrent
                    dVHist = 0.9 * dVHist + 0.1 * dVCurrent * dVCurrent
                    dWHist = 0.9 * dWHist + 0.1 * dWCurrent * dWCurrent
                    dLHist = 0.9 * dLHist + 0.1 * dLCurrent * dLCurrent

                    dWsMask *= .7 * (dWsPrev * dWsCurrent >= 0) + .5
                    dWMask *= .7 * (dWPrev * dWCurrent >= 0) + .5
                    dVMask *= .7 * (dVPrev * dVCurrent >= 0) + .5
                    dLMask *= .7 * (dLPrev * dLCurrent >= 0) + .5

                    dWsCurrent = eta * \
                        dWsMask.clip(1e-6, 50, out=dWsMask) * \
                        dWsCurrent / np.sqrt(dWsHist + eps)
                    dWCurrent = eta * \
                        dWMask.clip(1e-6, 50, out=dWMask) * \
                        dWCurrent / np.sqrt(dWHist + eps)
                    dVCurrent = eta * \
                        dVMask.clip(1e-6, 20, out=dVMask) * \
                        dVCurrent / np.sqrt(dVHist + eps)
                    dLCurrent = eta * \
                        dLMask.clip(1e-6, 20, out=dLMask) * \
                        dLCurrent / np.sqrt(dLHist + eps)

                # Calcul de la norme du gradient (critere d'arret)
                gradNorm = np.sum(np.abs(dWsCurrent))
                gradNorm += np.sum(np.abs(dWCurrent))
                gradNorm += np.sum(np.abs(dVCurrent))
                gradNorm += np.sum(np.abs(dLCurrent))

                # Keep previous gradient
                dWsPrev = dWsCurrent
                dWPrev = dWCurrent
                dVPrev = dVCurrent
                dLPrev = dLCurrent

                # Descente
                self.Ws -= dWsCurrent
                self.W -= dWCurrent
                self.V -= dVCurrent
                self.L -= dLCurrent

                if k % 10 == 0:
                    print "Epoch %i, mini batch %i, error : %.4f, gradient norm : %.4f" \
                        % (n_epoch, k, currentMbe, gradNorm)
                    errMB.append(currentMbe)

            print "End of epoch %i." % n_epoch

            # Maj de la condition d'arret
            if val_set != [] and (n_epoch % n_check) == 0:

                currentError = self.error(val_set)

                regularisationCost = self.regWs * np.sum(self.Ws * self.Ws) / 2.0
                regularisationCost += self.regW * np.sum(self.W * self.W) / 2.0
                regularisationCost += self.regV * np.sum(self.V * self.V) / 2.0
                regularisationCost += self.regL * np.sum(self.L * self.L) / 2.0

                errVal.append(currentError)
                print('Error+regularisation cost, on validation set at epoch {0} : '
                      '{1} + {2} | (previous : {3})'
                      .format(n_epoch, currentError, regularisationCost, prevError))

                currentError += regularisationCost

                self.printPerformanceSummary(val_set)
                self.print_confusion_matrix(val_set)

                with open(save_tmp, 'wb') as output:
                    pickle.dump(errVal, output, -1)
                    pickle.dump(errMB, output, -1)

                self.save(modelPath + "-%i.pkl" % n_epoch)

                # Early stopping
                minError = min(minError, currentError)
                glError = 100 * ((currentError / minError) - 1.0)
                if currentError > prevError:
                    upStop += 1
                else:
                    upStop = 0
                # UP criterion
                early_stop = (upStop >= n_stop) and (n_stop > 0)
                prevError = currentError

            # Maj epoch
            n_epoch += 1

        if val_set != []:
            print('Error on training set before and after training'
                  '({2} epoch) : {0}->{1}\n'
                  .format(iniError, currentError, n_epoch))
            print('Generalization error : {0}'.format(glError))
        return errMB, errVal

    def printPerformanceSummary(self, X_trees):
        sa, sr = self.score_fine(X_trees)
        sa_bin, sr_bin = self.score_binary(X_trees)
        print '\t\tFine grain\tBinary'
        print 'Overall\t| %3f\t%3f' % (sa, sa_bin)
        print 'Root\t| %3f\t%3f' % (sr, sr_bin)
        self.print_layers_norm()

    def score(self, function, X_trees):
        scRoot = (0.0, 0.0)  # Score and count for roots
        scAll = (0.0, 0.0)
        for X_tree in X_trees:
            self.forward_pass(X_tree)
            scTree = reduce(aggregate, imap(function, X_tree.nodes))
            scAll = aggregate(scAll, scTree)
            scRoot = aggregate(scRoot, function(X_tree.root))

        return scAll[0] / scAll[1], scRoot[0] / scRoot[1]

    def score_fine(self, X_trees):
        return self.score(Node.score_fine, X_trees)

    def score_binary(self, X_trees, inc_neut=False):
        s = lambda n: n.score_binary(inc_neut=inc_neut)
        return self.score(s, X_trees)

    def score_eps(self, X_trees, eps):
        s = lambda n: n.score_eps(eps)
        return self.score(s, X_trees)

    def check_derivative(self, X_tree, eps=1e-6):
        '''
        Fait une comparaison dérivee / differences finies
        '''
        error1 = self.forward_pass(X_tree)
        dWs, dV, dW, dL = self.backward_pass(X_tree)
        dirW = np.random.uniform(size=self.W.shape)
        dirWs = np.random.uniform(size=self.Ws.shape)
        dirL = np.random.uniform(size=self.L.shape)
        dirV = np.random.uniform(size=self.V.shape)

        self.Ws += eps * dirWs
        self.W += eps * dirW
        self.L += eps * dirL
        self.V += eps * dirV

        error2 = self.forward_pass(X_tree)
        diff = (error2 - error1) / eps
        diff2 = np.sum(dW * dirW) + np.sum(
            dWs * dirWs) + np.sum(dL * dirL) + np.sum(dV * dirV)

        return np.abs(diff - diff2)

    def confusion_matrix(self, X_trees):
        confAll = np.zeros((5, 5))
        confRoot = np.zeros((5, 5))
        for tree in X_trees:
            for n in ifilter(Node.have_label, tree.nodes):
                lp = Node.getSoftLabel(n.ypred[1])
                l = Node.getSoftLabel(n.y[1])
                confAll[l, lp] += 1
            if tree.root.have_label():
                lp = Node.getSoftLabel(tree.root.ypred[1])
                l = Node.getSoftLabel(tree.root.y[1])
                confRoot[l, lp] += 1
        # for lp in range(5):
        #     confAll[lp, :] /= np.sum(confAll[lp, :])
        #     confRoot[lp, :] /= np.sum(confRoot[lp, :])
        return confAll, confRoot

    def print_confusion_matrix(self, X_trees):
        confAll, confRoot = self.confusion_matrix(X_trees)
        print 'confusion on root'
        print confRoot
        print 'confusion on nodes'
        print confAll

    def print_layers_norm(self):
        norm = lambda W: np.linalg.norm(W) / W.size
        print "tensor norm: %f, max: %f" % (norm(self.V), np.max(self.V))
        print "combinator norm: %f, max: %f" % (norm(self.W), np.max(self.W))
        print "sentiment norm: %f, max: %f" % (norm(self.Ws), np.max(self.Ws))
        print "lexicon norm: %f, max: %f" % (norm(self.L), np.max(self.L))

    def plot_words_2D(self, labels):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
        pca = PCA(n_components=2)
        X = pca.fit_transform(self.L)
        revert_vocab = {i: (w, labels[w]) for (w, i) in self.vocab.iteritems()}
        y = np.zeros(self.L.shape[0])
        for i in range(self.L.shape[0]):
            _, y[i] = revert_vocab[i]
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.jet)


def aggregate(scTotal, sc):
    return (scTotal[0] + sc[0], scTotal[1] + sc[1])


def init_matrix(nRows, nCols, bias=True, r=None, method="Bengio"):
    mat = np.zeros((nRows, nCols + bias))
    print mat.shape
    if r is None:
        if method == "Bengio":
            r = sqrt(6) / sqrt(nRows + nCols)
        if method == "Classic":
            r = 1.0 / sqrt(nCols)
        if method == "Marteens":
            for i in range(nRows):
                mat[i, np.random.choice(range(nCols), 15)] = np.random.normal(1, 1, 15)
                if bias:
                    mat[::, -1] = 0.5
            return mat
        mat[::, :nCols] = np.random.uniform(-r, r, size=(nRows, nCols))
    return mat


def init_tensor(nRows, nCols, nLayers, r=0.0001):
    V = np.random.uniform(-r, r, size=(nRows, nCols, nLayers))
    V = (V + np.transpose(V, axes=[0, 2, 1])) / 2
    return V
