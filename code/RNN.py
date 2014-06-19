# -*- coding: utf8 -*-
import numpy as np
import pickle
import Tree


class RNN(object):

    """Class to use Recursive Neural Network on Tree

    Usage
    -----


    Methods
    -------

    """

    def __init__(self, vocab={}, dim=30, r=0.0001, reg=1):
        self.sparse = True
        self.dim = dim

        # Initiate V, the tensor operator
        #self.V = np.random.uniform(-r, r, size=(dim, 2 * dim, 2 * dim))
        #self.V = (self.V + np.transpose(self.V, axes=[0, 2, 1])) / 2

        # Initiate W, the linear operator
        ##self.W = np.random.uniform(-r, r, size=(dim, 2 * dim))
        #self.W = np.random.uniform(-1.0, 1.0, size=(dim, 2 * dim))
        # self.sparsify(self.W)
        # self.W = np.concatenate((self.W, np.zeros((self.W.shape[0], 1))),
        # axis=1)       # joc: additional column for bias (init = 0)

        # Initiate Ws, the sentiment linear operator
        ##self.Ws = np.random.uniform(-r, r, size=(2, dim))
        ##self.Ws = np.random.uniform(-1.0, 1.0, (2, dim))
        # self.sparsify(self.Ws)
        #self.Ws = np.zeros((2, dim))
        # self.Ws = np.concatenate((self.Ws, np.zeros((self.Ws.shape[0], 1))),
        # axis=1)    # joc: additional column for bias (init = 0)

        # Regularisation
        self.regV = reg * 0.001
        self.regW = reg * 0.001
        self.regWs = reg * 0.0001
        self.regL2 = reg * 0.0001

        # Initiate L, the Lexicon representation
        #self.L = np.random.uniform(-r, r, size=(len(vocab), dim))
        #self.L = np.random.uniform(-0.5, 0.5, size=(len(vocab), dim))

        self.vocab = {}
        for i, w in enumerate(vocab):
            self.vocab[w] = i

        #self.f = lambda X: np.tanh(X.T.dot(self.V).dot(X) + self.W.dot(X))
        self.f = lambda X: np.tanh(
            X.T.dot(self.V).dot(X) + self.W.dot(np.append(X, 1)))  # joc
        self.grad = lambda f: 1 - f ** 2

        self.f2 = lambda X: np.tanh(self.L2.dot(np.append(X, 1)))
        self.grad2 = lambda f: 1 - f ** 2

        #self.y = lambda x: np.exp(self.Ws.dot(x).clip(-500, 700)) \
        #    / sum(np.exp(self.Ws.dot(x).clip(-500, 700)))
        self.y = lambda x: np.exp(self.Ws.dot(np.append(x, 1)).clip(-500, 700)) \
            / sum(np.exp(self.Ws.dot(np.append(x, 1)).clip(-500, 700)))

    def sparsify(self, W):
        for row in range(0, W.shape[0]):
            n = np.random.permutation(range(0, W.shape[1]))[:-15]
            W[row, n] = 0
        return W

    def save(self, saveFile):
        with open(saveFile, 'wb') as output:
            pickle.dump(self.dim, output, -1)
            pickle.dump(self.V, output, -1)
            pickle.dump(self.W, output, -1)
            pickle.dump(self.Ws, output, -1)
            pickle.dump(self.regV, output, -1)
            pickle.dump(self.regW, output, -1)
            pickle.dump(self.regWs, output, -1)
            pickle.dump(self.regL2, output, -1)
            pickle.dump(self.L2, output, -1)
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
            self.regL2 = pickle.load(input)
            self.L2 = pickle.load(input)
            self.vocab = pickle.load(input)

    def error(self, val_set):
        '''
        Calcule l'erreur moyenne sur le set val_set avec les parametres actuel
        '''
        #errorVal = 0.0
        # for X_tree in val_set:
        #    errorVal += self.forward_pass(X_tree)
        # return errorVal / len(val_set)
        return sum([self.forward_pass(X_tree) for X_tree in val_set]) / len(val_set)

    def forward_pass(self, X_tree):
        '''
        Effectue la passe forward
        et calcule l'erreur actuelle pour l'arbre X_tree
        '''
        errorVal = 0.0
        for n in X_tree.leaf:
            # Met a jour le mot avec le Lexicon courant
            #n.X = self.L[self.vocab[n.word]]
            n.X = self.f2(self.embeddings[self.vocab[n.word]])

            n.ypred = self.y(n.X)  # Mise a jour du label predit
            n.ypred += 1e-300 * (n.ypred == 0)
            assert (n.ypred != 0).all()
            errorVal += -np.sum(n.y * np.log(n.ypred))

        for p, [a, b] in X_tree.parcours:
            aT = X_tree.nodes[a]  #
            # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            bT = X_tree.nodes[b]
            pT = X_tree.nodes[p]
            if aT.order < bT.order:
                X = np.append(aT.X, bT.X)
            else:
                X = np.append(bT.X, aT.X)
            pT.X = self.f(X)  # Mise a jour du decripteur du parent
            pT.ypred = self.y(pT.X)  # Mise a jour du label predit
            pT.ypred += 1e-300 * (pT.ypred == 0)
            errorVal += -np.sum(pT.y * np.log(pT.ypred))
        #E = sum([(self.y(n.X) - n.y) for n in X_tree.nodes])
        # print E
        # return self.Ws.dot(pT.X) -> Pas besoin de retourner le lbel, il faut
        # le maj aussi?
        return errorVal / (len(X_tree.leaf) + len(X_tree.parcours))

    def backward_pass(self, X_tree, root=False):
        '''
        Retourne le gradient du a l'erreur commise sur l'arbre X_tree
        Attention: suppose la forward_pass faite
        '''
        # Initialise les gradients
        dWs = np.zeros(self.Ws.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        dL2 = np.zeros(self.L2.shape)

        # Initialise les deltas
        for n in X_tree.nodes:
            if self.sparse and n.parent != -1:
                n.d = np.zeros((self.dim))
            else:
                n.d = self.Ws[:, 0:self.dim].T.dot(n.ypred - n.y)

        # Descend dans l arbre
        for p, [a, b] in X_tree.parcours[::-1]:
            # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            aT = X_tree.nodes[a]
            bT = X_tree.nodes[b]
            pT = X_tree.nodes[p]
            # if np.max(pT.d) == 0:
            #    raise Exception("oho")

            # Propagation des erreurs vers le bas
            gX = self.grad(pT.X)
            if aT.order < bT.order:  # Si aT est le noeud de gauche
                X = np.append(aT.X, bT.X)
                ddown = (
                    self.W[0:self.dim, 0:2 * self.dim].T.dot(pT.d * gX)
                    + 2 * (pT.d * gX).dot(self.V.dot(X)))
                aT.d += ddown[:self.dim]
                bT.d += ddown[self.dim:]
            else:  # aT est a droite
                X = np.append(bT.X, aT.X)
                ddown = (
                    self.W[0:self.dim, 0:2 * self.dim].T.dot(pT.d * gX)
                    + 2 * (pT.d * gX).dot(self.V.dot(X)))
                aT.d += ddown[self.dim:]
                bT.d += ddown[:self.dim]
            # Contribution aux gradients du pT
            if (not self.sparse) or pT.parent == -1:
                #dWs += np.outer(pT.ypred - pT.y, pT.X)
                # delta * X
                dWs += np.outer(pT.ypred - pT.y, np.append(pT.X, 1))

            dV += np.tensordot(pT.d * gX, np.outer(X, X), axes=0)
            #dW += np.outer(pT.d * gX, X)
            # joc           # d = nextW * nextDelta => d * gx = delta => delta
            # * X
            dW += np.outer(pT.d * gX, np.append(X, 1))

        # Contribution des feuilles
        # L(10 000, 30) = embeddings(10 000, 300) x L2(300, 30)
        for n in X_tree.leaf:
            #dL[self.vocab[n.word]] += n.d
            gX = self.grad(n.X)

            X = self.embeddings[self.vocab[n.word]]

            dL2 += np.outer(n.d * gX, np.append(X, 1))
            if (not self.sparse) or n.parent == -1:
                #dWs += np.outer(n.ypred - n.y, n.X)
                dWs += np.outer(n.ypred - n.y, np.append(n.X, 1))

        return dWs, dV, dW, dL2

    # TODO use the w_root argument ??
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
        #max_epoch = 1000
        # Remise à zero du modele
        if not warm_start:
            print 'reseting the RNN'

            dim = self.dim
            # Initiate V, the tensor operator
            self.V = np.random.uniform(-r, r, size=(dim, 2 * dim, 2 * dim))
            self.V = (self.V + np.transpose(self.V, axes=[0, 2, 1])) / 2

            # Initiate W, the linear operator
            #self.W = np.random.uniform(-r, r, size=(dim, 2 * dim))
            self.W = np.random.uniform(-1.0, 1.0, size=(dim, 2 * dim))
            self.sparsify(self.W)
            self.W = np.concatenate(
                (self.W, np.zeros((self.W.shape[0], 1))), axis=1)  # joc: additional column for bias (init = 0)

            # Initiate Ws, the linear operator
            #self.Ws = np.random.uniform(-r, r, size=(2, dim))
            #self.Ws = np.random.uniform(-1.0, 1.0, size=(2, dim))
            # self.sparsify(self.Ws)
            self.Ws = np.zeros((2, dim))
            self.Ws = np.concatenate(
                (self.Ws, np.zeros((self.Ws.shape[0], 1))), axis=1)  # joc: additional column for bias (init = 0)

            # Initiate L, the Lexicon representation
            #self.L = np.random.uniform(-r, r, size=(len(self.vocab), dim))
            #self.L = np.random.uniform(-0.5, 0.5, size=(len(self.vocab), dim))

            # L(len(vocab), dim) = embeddings(len(vocab), 300) x L2(300, dim)
            from Mikolov import load_mikolov
            self.embeddings = load_mikolov(self.vocab)
            self.L2 = np.random.uniform(
                -0.5, 0.5, size=(dim, self.embeddings.shape[1]))
            self.sparsify(self.L2)
            self.L2 = np.concatenate(
                (self.L2, np.zeros((self.L2.shape[0], 1))), axis=1)  # joc: additional column for bias (init = 0)

        self.printPerformanceSummary(val_set)
        self.printConfusion_matrix(val_set)

        # Liste pour erreurs
        errMB = []
        errVal = []

        # Condition d'arret
        n_epoch = 1
        gradNorm = 1.0
        if val_set != []:
            prevError = self.error(val_set)
            #prevError += self.regWs * np.sum(self.Ws * self.Ws) / 2.0
            #prevError += self.regW * np.sum(self.W * self.W) / 2.0
            #prevError += self.regV * np.sum(self.V * self.V) / 2.0
            #prevError += self.regL * np.sum(self.L * self.L) / 2.0
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
        dL2Hist = np.zeros(self.L2.shape)

        # Adaptative LR for RMSprop
        dWsMask = np.ones(self.Ws.shape)
        dVMask = np.ones(self.V.shape)
        dWMask = np.ones(self.W.shape)
        dL2Mask = np.ones(self.L2.shape)

        dWsPrev = np.zeros(self.Ws.shape)
        dVPrev = np.zeros(self.V.shape)
        dWPrev = np.zeros(self.W.shape)
        dL2Prev = np.zeros(self.L2.shape)

        early_stop = False
        n_trees = len(X_trees)
        learn_by_epoch = True

        if learn_by_epoch:
            # TODO learn by epoch
            mini_batch_numbers = (
                n_trees + mini_batch_size - 1) / mini_batch_size
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
                dL2Hist = np.zeros(self.L2.shape)

            for k in range(mini_batch_numbers):
                # print "Starting round %i" % k
                if learn_by_epoch:
                    # take mini batch according to the current permutation
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
                dL2Current = np.zeros(self.L2.shape)

                # Mini batch pour gradient
                currentMbe = 0.0
                for X_tree in mini_batch_samples:
                    currentMbe += self.forward_pass(X_tree)
                    dWs, dV, dW, dL2 = self.backward_pass(X_tree)
                    dWsCurrent += dWs
                    dVCurrent += dV
                    dWCurrent += dW
                    dL2Current += dL2

                n = len(mini_batch_samples)
                currentMbe /= n
                #currentMbe += self.regWs * np.sum(self.Ws * self.Ws) / 2.0
                #currentMbe += self.regW * np.sum(self.W * self.W) / 2.0
                #currentMbe += self.regV * np.sum(self.V * self.V) / 2.0
                #currentMbe += self.regL * np.sum(self.L * self.L) / 2.0

                # Division par le nombre de sample + regularisation
                dWsCurrent = dWsCurrent / n + self.regWs * self.Ws
                dVCurrent = dVCurrent / n + self.regV * self.V
                dWCurrent = dWCurrent / n + self.regW * self.W
                dL2Current = dL2Current / n + self.regL2 * self.L2

                # Mise a jour des poids et calcul des pas
                if strat == 'AdaGrad':
                    eps = 0.001  # Adagrad >0 ? cf Socher
                    dWsHist += dWsCurrent * dWsCurrent
                    dVHist += dVCurrent * dVCurrent
                    dWHist += dWCurrent * dWCurrent
                    dL2Hist += dL2Current * dL2Current

                    dWsCurrent = eta * dWsCurrent / np.sqrt(dWsHist + eps)
                    dWCurrent = eta * dWCurrent / np.sqrt(dWHist + eps)
                    dVCurrent = eta * dVCurrent / np.sqrt(dVHist + eps)
                    dL2Current = eta * dL2Current / np.sqrt(dL2Hist + eps)

                    #eta = 0.005
                    #dWsCurrent = eta * dWsCurrent
                    #dWCurrent = eta * dWCurrent
                    #dVCurrent = eta * dVCurrent
                    #dL2Current = eta * dL2Current
                else:
                    eps = 0.001
                    dWsHist = 0.9 * dWsHist + 0.1 * dWsCurrent * dWsCurrent
                    dVHist = 0.9 * dVHist + 0.1 * dVCurrent * dVCurrent
                    dWHist = 0.9 * dWHist + 0.1 * dWCurrent * dWCurrent
                    dL2Hist = 0.9 * dL2Hist + 0.1 * dL2Current * dL2Current

                    dWsMask *= .7 * (dWsPrev * dWsCurrent >= 0) + .5
                    dWMask *= .7 * (dWPrev * dWCurrent >= 0) + .5
                    dVMask *= .7 * (dVPrev * dVCurrent >= 0) + .5
                    dL2Mask *= .7 * (dL2Prev * dL2Current >= 0) + .5

                    dWsCurrent = eta * \
                        dWsMask.clip(1e-6, 50, out=dWsMask) * \
                        dWsCurrent / np.sqrt(dWsHist + eps)
                    dWCurrent = eta * \
                        dWMask.clip(1e-6, 50, out=dWMask) * \
                        dWCurrent / np.sqrt(dWHist + eps)
                    dVCurrent = eta * \
                        dVMask.clip(1e-6, 20, out=dVMask) * \
                        dVCurrent / np.sqrt(dVHist + eps)
                    dL2Current = eta * \
                        dL2Mask.clip(1e-6, 20, out=dL2Mask) * \
                        dL2Current / np.sqrt(dL2Hist + eps)

                # Calcul de la norme du gradient (critere d'arret)
                gradNorm = np.sum(np.abs(dWsCurrent))
                gradNorm += np.sum(np.abs(dWCurrent))
                gradNorm += np.sum(np.abs(dVCurrent))
                gradNorm += np.sum(np.abs(dL2Current))

                # Keep previous gradient
                dWsPrev = dWsCurrent
                dWPrev = dWCurrent
                dVPrev = dVCurrent
                dL2Prev = dL2Current

                # Descente
                self.Ws -= dWsCurrent
                self.W -= dWCurrent
                self.V -= dVCurrent
                self.L2 -= dL2Current

                if k % 10 == 0:
                    print "Epoch %i, mini batch %i, error : %.4f, gradient norm : %.4f" \
                        % (n_epoch, k, currentMbe, gradNorm)
                    errMB.append(currentMbe)

            print "End of epoch %i." % n_epoch

            # Maj de la condition d'arret
            if val_set != [] and (n_epoch % n_check) == 0:

                currentError = self.error(val_set)

                regularisationCost = self.regWs * \
                    np.sum(self.Ws * self.Ws) / 2.0
                regularisationCost += self.regW * np.sum(self.W * self.W) / 2.0
                regularisationCost += self.regV * np.sum(self.V * self.V) / 2.0
                regularisationCost += self.regL2 * \
                    np.sum(self.L2 * self.L2) / 2.0

                errVal.append(currentError)
                print('Error+regularisation cost, on validation set at epoch {0} : '
                      '{1} + {2} | (current: {3}, previous : {4})'
                      .format(n_epoch, currentError, regularisationCost, currentError, prevError))
                print('Error on training set at epoch {0} : '
                      .format(self.error(X_trees)))

                #currentError += regularisationCost

                self.printPerformanceSummary(val_set)
                self.printConfusion_matrix(val_set)

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
        print 'Fine grain\tBinary'
        print 'Overall\t\t{:.3}\t{:.3}'.format(sa, sa_bin)
        print 'Root\t\t{:.3}\t{:.3}'.format(sr, sr_bin)

    def score_fine(self, X_trees):
        '''
        Score sur les predictions MAP avec 5 label
        '''
        countAll = 0
        countRoot = 0
        scAll = 0.0
        scRoot = 0.0
        for X_tree in X_trees:
            self.forward_pass(X_tree)
            for n in X_tree.nodes:
                countAll += 1
                scAll += (Tree.Tree.getSoftLabel(
                    n.ypred[1]) == Tree.Tree.getSoftLabel(n.y[1]))
            countRoot += 1
            n = X_tree.nodes[-1]
            scRoot += (Tree.Tree.getSoftLabel(
                n.ypred[1]) == Tree.Tree.getSoftLabel(n.y[1]))
        return scAll / countAll, scRoot / countRoot

    def score_binary(self, X_trees, inc_neut=False):
        '''
        Score sur les prediction MAP pos/neg
        '''
        countAll = 0
        countRoot = 0
        scAll = 0.0
        scRoot = 0.0
        for X_tree in X_trees:
            self.forward_pass(X_tree)
            for n in X_tree.nodes:
                countAll += 1 * (inc_neut or not (0.4 < n.y[1] <= 0.6))
                scAll += ((n.ypred[1] <= 0.5 and n.y[1] <= 0.5) or
                          (n.ypred[1] > 0.5 and n.y[1] > 0.5)
                          ) * (inc_neut or not (0.4 < n.y[1] <= 0.6))
            n = X_tree.nodes[-1]
            countRoot += 1 * (inc_neut or not (0.4 < n.y[1] <= 0.6))
            scRoot += ((n.ypred[1] <= 0.5 and n.y[1] <= 0.5) or
                       (n.ypred[1] > 0.5 and n.y[1] > 0.5)
                       ) * (inc_neut or not (0.4 < n.y[1] <= 0.6))
        return scAll / countAll, scRoot / countRoot

    def printConfusion_matrix(self, X_trees):
        confusionNode = np.zeros((5, 5))
        confusionRoot = np.zeros((5, 5))
        for X_tree in X_trees:
            self.forward_pass(X_tree)
            for n in X_tree.nodes:
                gold_label = Tree.Tree.getSoftLabel(n.y[1])
                predicted_label = Tree.Tree.getSoftLabel(n.ypred[1])
                confusionNode[gold_label, predicted_label] += 1

            gold_label = Tree.Tree.getSoftLabel(n.y[1])
            predicted_label = Tree.Tree.getSoftLabel(n.ypred[1])
            confusionRoot[gold_label, predicted_label] += 1
        print 'confusion on root'
        print confusionRoot
        print 'confusion on nodes'
        print confusionNode

    def check_derivative(self, X_tree, eps=1e-6):
        '''
        Fait une compaaison dérivee / differences finies
        '''
        error1 = self.forward_pass(X_tree)
        dWs, dV, dW, dL = self.backward_pass(X_tree)
        dirW = np.random.uniform(size=self.W.shape)
        dirWs = np.random.uniform(size=self.Ws.shape)
        dirL = np.random.uniform(size=self.L2.shape)
        dirV = np.random.uniform(size=self.V.shape)

        self.Ws += eps * dirWs
        self.W += eps * dirW
        self.L += eps * dirL
        self.V += eps * dirV

        error2 = self.forward_pass(X_tree)
        diff = (error2 - error1) / eps
        diff2 = np.sum(dW * dirW) + np.sum(dWs * dirWs) \
            + np.sum(dL * dirL) + np.sum(dV * dirV)

        return np.abs(diff - diff2)

    def confusion_matrix(self, X_trees):
        confAll = np.zeros((5, 5))
        confRoot = np.zeros((5, 5))
        for tree in X_trees:
            for n in tree.nodes:
                lp = Tree.Tree.getSoftLabel(n.ypred[1])
                l = Tree.Tree.getSoftLabel(n.y[1])
                confAll[l, lp] += 1
            lp = Tree.Tree.getSoftLabel(tree.nodes[-1].ypred[1])
            l = Tree.Tree.getSoftLabel(tree.nodes[-1].y[1])
            confRoot[l, lp] += 1
        for lp in range(5):
            confAll[lp, :] /= np.sum(confAll[lp, :])
            confRoot[lp, :] /= np.sum(confRoot[lp, :])
        return confAll, confRoot

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

    def score_eps(self, X_trees, eps):
        '''Score sur les predictions MAP avec 5 label
        '''
        countAll = 0
        countRoot = 0
        scAll = 0.0
        scRoot = 0.0
        for X_tree in X_trees:
            self.forward_pass(X_tree)
            for n in X_tree.nodes:
                countAll += 1
                scAll += (abs(n.ypred[1] - n.y[1]) <= eps)
            countRoot += 1
            n = X_tree.nodes[-1]
            scRoot += (abs(n.ypred[1] - n.y[1]) <= eps)
        return scAll / countAll, scRoot / countRoot
