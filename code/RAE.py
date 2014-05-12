# -*- coding: utf8 -*-
import numpy as np
import pickle
import Tree


class RAE(object):
    """Class to use Recursive Neural Network on Tree

    Usage
    -----


    Methods
    -------

    """
    def __init__(self, vocab={}, dim=30, r=0.0001, reg=1):
        self.dim = dim

        #Initiate V, the tensor operator
        self.V = np.random.uniform(-r, r, size=(dim+1, 2*dim+2, 2*dim+2))
        self.V = (self.V+np.transpose(self.V, axes=[0, 2, 1]))/2

        #Initiate W, the linear operator
        self.W = np.random.uniform(-r, r, size=(dim+1, 2*dim+2))

        #Initiate Ws, the linear operator
        self.Ws = np.random.uniform(-r, r, size=(2, dim+1))

        #Initiate VE, the encoder tensor
        self.Ve = np.random.uniform(-r, r, size=(2*dim+2, dim+1, dim+1))
        self.Ve = (self.Ve+np.transpose(self.Ve, axes=[0, 2, 1]))/2

        #Initiate VE, the encoder tensor
        self.We = np.random.uniform(-r, r, size=(2*dim+2, dim+1))

        #Initiate L, the Lexicon representation
        self.L = np.random.uniform(-r, r, size=(len(vocab), dim))

        #Parameter holder
        self.params = {}
        self.params['V'] = self.V
        self.params['W'] = self.W
        self.params['Ws'] = self.Ws
        self.params['L'] = self.L
        self.params['Ve'] = self.Ve
        self.params['We'] = self.We

        #Regularisation
        self.reg = {'V': 0.001*reg, 'W': 0.001*reg, 'Ws': 0.0001*reg, 'L': 0.0001*reg,
                    'We': reg*0.0001, 'Ve': reg*0.0001}

        self.vocab = {}
        self.index = {}
        for i, w in enumerate(vocab):
            self.vocab[w] = i
            self.index[i] = w

        self.f = lambda X: np.tanh(X.T.dot(self.V).dot(X) + self.W.dot(X))
        self.grad = lambda f: 1-f**2
        self.dec = lambda X: np.tanh(X.T.dot(self.Ve).dot(X) + self.We.dot(X))

        self.y = lambda x: np.exp(self.Ws.dot(x).clip(-500, 700)) \
            / sum(np.exp(self.Ws.dot(x).clip(-500, 700)))

        self.norm = lambda X: np.sum(X*X)

    def save(self, saveFile):
        with open(saveFile, 'wb') as output:
            pickle.dump(self.dim, output, -1)
            pickle.dump(self.vocab, output, -1)
            for k in self.params.keys():
                pickle.dump(self.params[k], output, -1)
            for k in self.params.keys():
                pickle.dump(self.reg[k], output, -1)

    def load(self, loadFile):
        with open(loadFile, 'rb') as input:
            self.dim = pickle.load(input)
            self.vocab = pickle.load(input)
            for k in self.params.keys():
                self.params[k] = pickle.load(input)
            for k in self.params.keys():
                self.reg[k] = pickle.load(input)

    def error(self, val_set):
        '''
        Calcule l'erreur moyenne sur le set val_set avec les parametres actuel
        '''
        errorVal = 0.0
        for X_tree in val_set:
            errorVal += self.forward_pass(X_tree)
        return errorVal/len(val_set)

    def forward_pass(self, X_tree):
        '''
        Effectue la passe forward et calcule l'erreur actuelle pour l'arbre X_tree
        '''
        errorVal = 0.0
        for n in X_tree.leaf:
            # Met a jour le mot avec le Lexicon courant
            n.X = np.append(self.L[self.vocab[n.word]], 0)
            n.ypred = self.y(n.X)  # Mise a jour du label predit
            n.ypred += 1e-300*(n.ypred == 0)
            assert (n.ypred != 0).all()
            errorVal += -np.sum(n.y*np.log(n.ypred))

        for p, [a, b] in X_tree.parcours:
            aT = X_tree.nodes[a]  #
            bT = X_tree.nodes[b]  # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            pT = X_tree.nodes[p]
            if aT.order < bT.order:
                X = np.append(aT.X, bT.X)
            else:
                X = np.append(bT.X, aT.X)
            pT.X = self.f(X)  # Mise a jour du decripteur du parent
            pT.X[-1] = 1

            pT.ypred = self.y(pT.X)  # Mise a jour du label predit
            pT.ypred += 1e-300*(pT.ypred == 0)
            errorVal += -np.sum(pT.y*np.log(pT.ypred))
        pT.c = pT.X
        for p, [a, b] in X_tree.parcours[::-1]:
            aT = X_tree.nodes[a]  #
            bT = X_tree.nodes[b]  # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            pT = X_tree.nodes[p]
            C = self.dec(pT.c)
            #Propagation des erreurs vers le bas
            if aT.order < bT.order:  # Si aT est le noeud de gauche
                aT.c = C[:self.dim+1]
                bT.c = C[self.dim+1:]
            else:  # aT est a droite
                bT.c = C[:self.dim+1]
                aT.c = C[self.dim+1:]

        nn = len(X_tree.nodes)
        for n in X_tree.nodes:
            errorVal += np.sum((n.c-n.X)**2) / nn
        #E = sum([(self.y(n.X) - n.y) for n in X_tree.nodes])
        #print E
        #return self.Ws.dot(pT.X) -> Pas besoin de retourner le lbel, il faut le maj aussi?
        return errorVal

    def backward_pass(self, X_tree, w_root=1):
        '''
        Retourne le gradient du a l'erreur commise sur l'arbre X_tree
        Attention: suppose la forward_pass faite
        '''
        #Initialise les gradients
        grad = {}
        for k in self.params.keys():
            grad[k] = np.zeros(self.params[k].shape)

        #Initialise les deltas
        for n in X_tree.nodes:
            gX = self.grad(n.X)
            gC = self.grad(n.c)
            n.d = self.Ws.T.dot(n.ypred-n.y)*gX*w_root
            n.dr = 2*(n.c-n.X)*gC

        n.d += self.Ws.T.dot(n.ypred-n.y)*gX*(1-w_root)

        err_rec = 0

        for p, [a, b] in X_tree.parcours:
            aT = X_tree.nodes[a]  #
            bT = X_tree.nodes[b]  # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            pT = X_tree.nodes[p]
            if aT.order < bT.order:
                dR = np.append(aT.dr, bT.dr)
            else:
                dR = np.append(bT.dr, aT.dr)
            pT.dr += dR.dot(self.We) + 2*dR.dot(self.Ve.dot(pT.X))
            grad['We'] += np.outer(dR, pT.X)
            grad['Ve'] += np.tensordot(dR, np.outer(pT.X, pT.X), axes=0)

        #Descend dans l arbre
        for p, [a, b] in X_tree.parcours[::-1]:
            aT = X_tree.nodes[a]  #
            bT = X_tree.nodes[b]  # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            pT = X_tree.nodes[p]
            #Propagation des erreurs vers le bas
            if aT.order < bT.order:  # Si aT est le noeud de gauche
                X = np.append(aT.X, bT.X)
                ddown = (self.W.T.dot(pT.d+pT.dr)+2*(pT.d+pT.dr).dot(self.V.dot(X)))
                aT.d += ddown[:self.dim+1]
                bT.d += ddown[self.dim+1:]
            else:  # aT est a droite
                X = np.append(bT.X, aT.X)
                ddown = (self.W.T.dot(pT.d+pT.dr)+2*(pT.d+pT.dr).dot(self.V.dot(X)))
                aT.d += ddown[self.dim+1:]
                bT.d += ddown[:self.dim+1]
            err_rec += np.sum((aT.c-aT.X)**2)
            err_rec += np.sum((bT.c-bT.X)**2)
            #Contribution aux gradients du pT
            grad['Ws'] += np.outer(pT.ypred-pT.y, pT.X)
            grad['V'] += np.tensordot(pT.d + pT.dr, np.outer(X, X), axes=0)
            grad['W'] += np.outer(pT.d + pT.dr, X)

        #Contribution des feuilles
        for n in X_tree.leaf:
            grad['L'][self.vocab[n.word]] += n.d[:-1] + n.dr[:-1]
            grad['Ws'] += np.outer(n.ypred-n.y, n.X)

        return grad

    def train(self, X_trees, learning_rate=0.01, mini_batch_size=27,
              warm_start=True, r=0.0001, max_iter=1000, val_set=[],
              n_check=100, strat='AdaGrad', w_root=1,
              bin=False, reset_freq=-1, save_tmp='tmp.pkl', n_stop=4):
        '''
        Training avec AdaGrad (Dutchi et al.), prends en entrée une liste d'arbres X_trees
        '''
        #Remise à zero du modele
        if not warm_start:
            dim = self.dim
            #Initiate V, the tensor operator
            self.params['V'] = np.random.uniform(-r, r, size=(dim+1, 2*dim+2, 2*dim+2))
            self.params['V'] = (self.V+np.transpose(self.V, axes=[0, 2, 1]))/2

            #Initiate W, the linear operator
            self.params['W'] = np.random.uniform(-r, r, size=(dim+1, 2*dim+2))

            #Initiate Ws, the linear operator
            self.params['Ws'] = np.random.uniform(-r, r, size=(2, dim+1))

            #Initiate L, the Lexicon representation
            self.params['L'] = np.random.uniform(-r, r, size=(len(self.vocab), dim))

        #Liste pour erreurs
        self.errMB = []
        self.errVal = []
        errMB = self.errMB
        errVal = self.errVal

        #Condition d'arret
        n_iter = 1
        gradNorm = 1.0
        if val_set != []:
            prevError = self.error(val_set) / len(val_set)
            for k in self.params.keys():
                prevError += self.reg[k]*self.norm(self.params[k])/2.0
            iniError = prevError
            minError = prevError  # optimal error so far
            glError = 0
            upStop = 0
            errVal.append(prevError)

        # Normalisation pour AdaGrad/Rms-prop
        eta = learning_rate
        dHist = {}
        dMask = {}
        dPrev = {}
        for k in self.params.keys():
            sh = self.params[k].shape
            dHist[k] = np.zeros(sh)
            dMask[k] = np.ones(sh)
            dPrev[k] = np.zeros(sh)

        early_stop = False
        while (not early_stop) and n_iter < max_iter:  # Critere moins random
            if n_iter % reset_freq == 0 and reset_freq > 0:  # Remise a zero des rates cf Socher
                for k in self.params.keys():
                    dHist[k] = np.zeros(self.params[k].shape)

            #Choose mini batch randomly
            mini_batch_samples = np.random.choice(X_trees, size=mini_batch_size)

            #Initialize gradients to 0
            dCurrent = {}
            for k in self.params.keys():
                    dCurrent[k] = np.zeros(self.params[k].shape)

            #Mini batch pour gradient
            currentMbe = 0.0
            for X_tree in mini_batch_samples:
                currentMbe += self.forward_pass(X_tree)
                grad = self.backward_pass(X_tree, w_root=w_root)
                for k in self.params.keys():
                    dCurrent[k] += grad[k]

            currentMbe /= mini_batch_size
            for k in self.params.keys():
                currentMbe += self.reg[k]*self.norm(self.params[k])/2.0

            #Division par le nombre de sample + regularisation
            for k in self.params.keys():
                dCurrent[k] = dCurrent[k]/mini_batch_size + self.reg[k]*self.params[k]

            #Mise a jour des poids et calcul des pas
            if strat == 'AdaGrad':
                eps = 0.001  # Adagrad >0 ? cf Socher
                for k in self.params.keys():
                    dHist[k] += dCurrent[k]*dCurrent[k]
                    dCurrent[k] = eta*dCurrent[k]/np.sqrt(dHist[k]+eps)
            else:
                eps = 0.001
                for k in self.params.keys():
                    dHist[k] = 0.9*dHist[k] + 0.1*dCurrent[k]*dCurrent[k]
                    dMask[k] *= .7*(dPrev[k]*dCurrent[k] >= 0) + .5
                    dCurrent[k] = eta*dMask[k].clip(1e-6, 50, out=dMask[k]) *\
                        dCurrent[k]/np.sqrt(dHist[k]+eps)

            #Calcul de la norme du gradient (critere d'arret)
            gradNorm = 0
            for k in self.params.keys():
                gradNorm += np.sum(np.abs(dCurrent[k]))

            #Keep previous gradient
            for k in self.params.keys():
                dPrev[k] = dCurrent[k]

            #Descente
            for k in self.params.keys():
                self.params[k] -= dCurrent[k]

            #Maj de la condition d'arret
            if val_set != [] and (n_iter % n_check) == 0:
                currentError = self.error(val_set)
                for k in self.params.keys():
                    currentError += self.reg[k]*self.norm(self.params[k])/2.0

                errVal.append(currentError)
                errMB.append(currentMbe)
                print('Error on validation set at iter {0} : {1} '
                      '(previous : {2})'.format(n_iter, currentError, prevError))
                print('Error on mini batch at iter {0} : {1} '
                      '(Gradient norm : {2})'.format(n_iter, currentMbe, gradNorm))

                with open(save_tmp, 'wb') as output:
                    pickle.dump(errVal, output, -1)
                    pickle.dump(errMB, output, -1)

                #Early stopping
                minError = min(minError, currentError)
                glError = 100*((currentError/minError)-1.0)
                if currentError > prevError:
                    upStop += 1
                else:
                    upStop = 0
                early_stop = (upStop >= n_stop) and (n_stop > 0)  # UP criterion
                prevError = currentError
            else:
                print('Error on mini batch at iter {0} : {1} '
                      '(Gradient norm : {2})'.format(n_iter, currentMbe, gradNorm))
                errMB.append(currentMbe)

            #Maj iter
            n_iter += 1

        if val_set != []:
            print('Error on training set before and after training'
                  '({2} iter) : {0}->{1}\n'.format(iniError, currentError, n_iter))
            print('Generalization error : {0}'.format(glError))
        return errMB, errVal

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
                scAll += (Tree.Tree.getSoftLabel(n.ypred[1]) == Tree.Tree.getSoftLabel(n.y[1]))
            countRoot += 1
            n = X_tree.nodes[-1]
            scRoot += (Tree.Tree.getSoftLabel(n.ypred[1]) == Tree.Tree.getSoftLabel(n.y[1]))
        return scAll/countAll, scRoot/countRoot

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
                          (n.ypred[1] > 0.5 and n.y[1] > 0.5)) *\
                         (inc_neut or not (0.4 < n.y[1] <= 0.6))
            n = X_tree.nodes[-1]
            countRoot += 1 * (inc_neut or not (0.4 < n.y[1] <= 0.6))
            scRoot += ((n.ypred[1] <= 0.5 and n.y[1] <= 0.5) or
                      (n.ypred[1] > 0.5 and n.y[1] > 0.5)) *\
                      (inc_neut or not (0.4 < n.y[1] <= 0.6))
        return scAll/countAll, scRoot/countRoot

    def check_derivative(self, X_tree, eps=1e-6):
        '''
        Fait une compaaison dérivee / differences finies
        '''
        error1 = self.forward_pass(X_tree)
        dWs, dV, dW, dL = self.backward_pass(X_tree)
        dirW = np.random.uniform(size=self.W.shape)
        dirWs = np.random.uniform(size=self.Ws.shape)
        dirL = np.random.uniform(size=self.L.shape)
        dirV = np.random.uniform(size=self.V.shape)

        self.Ws += eps*dirWs
        self.W += eps*dirW
        self.L += eps*dirL
        self.V += eps*dirV

        error2 = self.forward_pass(X_tree)
        diff = (error2-error1)/eps
        diff2 = np.sum(dW*dirW)+np.sum(dWs*dirWs)+np.sum(dL*dirL)+np.sum(dV*dirV)

        return np.abs(diff-diff2)

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

    def plot_words_2D(self, labels, N=-1):
        from sklearn.decomposition import PCA
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm

        if N == -1:
            N = self.L.shape[0]

        pca = PCA(n_components=2)
        X = pca.fit_transform(self.L[:N])
        revert_vocab = {i: (w, labels[w]) for (w, i) in self.vocab.iteritems()}
        y = np.zeros(N)
        for i in range(N):
            _, y[i] = revert_vocab[i]
        plt.scatter(X[:, 0], X[:, 1], c=y, cmap=cm.jet)

    def sample_w(self, p):
        dist = ((self.L-p)**2).sum(axis=1)
        i0 = dist.argmin()
        return self.index[i0]

    def sample(self, p):
        queue = [p]
        while len(queue) != 0:
            curr = queue.pop()
            self.dec(curr)
            C = self.dec(curr)
            ac = C[:self.dim+1:]
            bc = C[self.dim+1:]
            if ac[-1] > 0.5:
                queue.append(ac)
            else:
                print self.sample_w(ac[:-1])
            if bc[-1] > 0.5:
                queue.append(bc)
            else:
                print self.sample_w(bc[:-1])
