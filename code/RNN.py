# -*- coding: utf8 -*-
import numpy as np
import pickle


class RNN(object):
    """Class to use Recursive Neural Network on Tree

    Usage
    -----


    Methods
    -------

    """
    def __init__(self, vocab={}, dim=30, r=0.0001):
        self.dim = dim

        #Initiate V, the tensor operator
        self.V = np.random.uniform(-r,r,size=(dim, 2*dim, 2*dim))
        self.V=(self.V+np.transpose(self.V,axes=[0,2,1]))/2

        #Initiate W, the linear operator
        self.W = np.random.uniform(-r,r,size=(dim, 2*dim))

        #Initiate Ws, the linear operator
        self.Ws = np.random.uniform(-r,r,size=(5, dim))
        
        #Un biais ca coute pas cher...
        self.B = np.random.uniform(-r,r,size=(dim))

        #Regularisation
        self.regV = 0.001
        self.regW = 0.001
        self.regWs = 0.0001
        self.regL = 0.0001
        self.regB = 0.001

        #Initiate L, the Lexicon representation
        self.L = np.random.uniform(-r, r, size=(len(vocab), dim))
        self.vocab = {}
        for i, w in enumerate(vocab):
            self.vocab[w] = i

        self.f = lambda X: np.tanh(X.T.dot(self.V).dot(X) + self.W.dot(X)+self.B)
        self.grad = lambda f: 1-f**2

        self.y = lambda x: np.exp(self.Ws.dot(x).clip(-500, 700)) \
            / sum(np.exp(self.Ws.dot(x).clip(-500, 700)))

    def save(self, saveFile):
        with open(saveFile, 'wb') as output:
            pickle.dump(self.dim, output, -1)
            pickle.dump(self.V, output, -1)
            pickle.dump(self.W, output, -1)
            pickle.dump(self.Ws, output, -1)
            pickle.dump(self.B, output, -1)
            pickle.dump(self.regV, output, -1)
            pickle.dump(self.regW, output, -1)
            pickle.dump(self.regWs, output, -1)
            pickle.dump(self.regL, output, -1)
            pickle.dump(self.regB, output, -1)
            pickle.dump(self.L, output, -1)
            pickle.dump(self.vocab, output, -1)

    def load(self, loadFile):
        with open(loadFile, 'rb') as input:
            self.dim = pickle.load(input)
            self.V = pickle.load(input)
            self.W = pickle.load(input)
            self.Ws = pickle.load(input)
            self.B = pickle.load(input)
            self.regV = pickle.load(input)
            self.regW = pickle.load(input)
            self.regWs = pickle.load(input)
            self.regL = pickle.load(input)
            self.regB = pickle.load(input)
            self.L = pickle.load(input)
            self.vocab = pickle.load(input)

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
            n.X = self.L[self.vocab[n.word]]  # Met a jour le mot avec le Lexicon courant
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
            pT.ypred = self.y(pT.X)  # Mise a jour du label predit
            pT.ypred += 1e-300*(pT.ypred == 0)
            errorVal += -np.sum(pT.y*np.log(pT.ypred))
        #E = sum([(self.y(n.X) - n.y) for n in X_tree.nodes])
        #print E
        #return self.Ws.dot(pT.X) -> Pas besoin de retourner le lbel, il faut le maj aussi?
        return errorVal

    def backward_pass(self, X_tree, root=False):
        '''
        Retourne le gradient du a l'erreur commise sur l'arbre X_tree
        Attention: suppose la forward_pass faite
        '''
        #Initialise les gradients
        dWs = np.zeros(self.Ws.shape)
        dV = np.zeros(self.V.shape)
        dW = np.zeros(self.W.shape)
        dL = np.zeros(self.L.shape)
        dB = np.zeros(self.B.shape)

        #Initialise les deltas
        for n in X_tree.nodes:
            n.d = self.Ws.T.dot(n.ypred-n.y)

        #Descend dans l arbre
        for p, [a, b] in X_tree.parcours[::-1]:
            aT = X_tree.nodes[a]  #
            bT = X_tree.nodes[b]  # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            pT = X_tree.nodes[p]
            gX = self.grad(pT.X)
            #Propagation des erreurs vers le bas
            if aT.order < bT.order:  # Si aT est le noeud de gauche
                X = np.append(aT.X, bT.X)
                ddown = (self.W.T.dot(pT.d*gX)+2*(pT.d*gX).dot(self.V.dot(X)))
                aT.d += ddown[:self.dim]
                bT.d += ddown[self.dim:]
            else:  # aT est a droite
                X = np.append(bT.X, aT.X)
                ddown = (self.W.T.dot(pT.d*gX)+2*(pT.d*gX).dot(self.V.dot(X)))
                aT.d += ddown[self.dim:]
                bT.d += ddown[:self.dim]
            #Contribution aux gradients du pT
            dWs += np.outer(pT.ypred-pT.y, pT.X)
            dV += np.tensordot(pT.d*gX, np.outer(X, X), axes=0)
            dW += np.outer(pT.d*gX, X)
            dB += pT.d*gX

        #Contribution des feuilles
        for n in X_tree.leaf:
            dL[self.vocab[n.word]] += n.d
            dWs += np.outer(n.ypred-n.y, n.X)

        return dWs, dV, dW, dL, dB

    def train(self, X_trees, learning_rate=0.01, mini_batch_size=27,
              warm_start=True, r=0.0001, max_iter=1000, val_set=[],
              n_check=100, strat='AdaGrad',
              bin=False,reset_freq=-1,save_tmp='tmp.pkl'):
        '''
        Training avec AdaGrad (Dutchi et al.), prends en entrée une liste d'arbres X_trees
        '''
        #Remise à zero du modele
        if not warm_start:
            dim = self.dim
            self.V = 1e-5*np.ones((dim, 2*dim, 2*dim))
            #Initiate W, the linear operator
            self.W = 1e-5*np.ones((dim, 2*dim))
            #Initiate Ws, the linear operator
            self.Ws = 1e-5*np.ones((5-bin*3, 2*dim))
            #Initiate L, the Lexicon representation
            self.L = np.random.uniform(-r, r, size=(len(self.vocab), dim))
            
            self.B = np.random.uniform(-r,r,size=(dim))

        #Liste pour erreurs
        errMB = []
        errVal = []
        #Condition d'arret
        n_iter = 1
        gradNorm = 1.0
        if val_set != []:
            prevError = self.error(val_set)
            prevError += self.regWs*np.sum(self.Ws*self.Ws)/2.0
            prevError += self.regW*np.sum(self.W*self.W)/2.0
            prevError += self.regV*np.sum(self.V*self.V)/2.0
            prevError += self.regL*np.sum(self.L*self.L)/2.0
            prevError += self.regB*np.sum(self.B*self.B)/2.0
            iniError = prevError
            minError = prevError #optimal error so far
            glError = 0
            errVal.append(prevError)

        # Normalisation pour AdaGrad/Rms-prop
        eta = learning_rate
        dWsHist = np.zeros(self.Ws.shape)
        dVHist = np.zeros(self.V.shape)
        dWHist = np.zeros(self.W.shape)
        dLHist = np.zeros(self.L.shape)
        dBHist = np.zeros(self.B.shape)

        #Adaptative LR for RMSprop
        dWsMask = np.ones(self.Ws.shape)
        dVMask = np.ones(self.V.shape)
        dWMask = np.ones(self.W.shape)
        dLMask = np.ones(self.L.shape)
        dBMask = np.ones(self.B.shape)

        dWsPrev = np.zeros(self.Ws.shape)
        dVPrev = np.zeros(self.V.shape)
        dWPrev = np.zeros(self.W.shape)
        dLPrev = np.zeros(self.L.shape)
        dBPrev = np.zeros(self.B.shape)        
        
        early_stop=False
        while (not early_stop) and n_iter < max_iter:  # Critere moins random
            if n_iter%reset_freq==0 and reset_freq>0: #Remise a zero des rates cf Socher
                dWsHist = np.zeros(self.Ws.shape)
                dVHist = np.zeros(self.V.shape)
                dWHist = np.zeros(self.W.shape)
                dLHist = np.zeros(self.L.shape)
                dBHist = np.zeros(self.B.shape)
                
            #Choose mini batch randomly                
            mini_batch_samples = np.random.choice(X_trees, size=mini_batch_size)
            
            #Initialize gradients to 0
            dWsCurrent = np.zeros(self.Ws.shape)
            dVCurrent = np.zeros(self.V.shape)
            dWCurrent = np.zeros(self.W.shape)
            dLCurrent = np.zeros(self.L.shape)
            dBCurrent = np.zeros(self.B.shape)

            #Mini batch pour gradient
            currentMbe = 0.0
            for X_tree in mini_batch_samples:
                currentMbe += self.forward_pass(X_tree)
                dWs, dV, dW, dL, dB = self.backward_pass(X_tree)
                dWsCurrent += dWs
                dVCurrent += dV
                dWCurrent += dW
                dLCurrent += dL
                dBCurrent += dB                

            currentMbe /= mini_batch_size
            currentMbe += self.regWs*np.sum(self.Ws*self.Ws)/2.0
            currentMbe += self.regW*np.sum(self.W*self.W)/2.0
            currentMbe += self.regV*np.sum(self.V*self.V)/2.0
            currentMbe += self.regL*np.sum(self.L*self.L)/2.0
            currentMbe += self.regB*np.sum(self.B*self.B)/2.0
            
            #Division par le nombre de sample + regularisation
            dWsCurrent = dWsCurrent/mini_batch_size+self.regWs*self.Ws
            dVCurrent = dVCurrent/mini_batch_size+self.regV*self.V
            dWCurrent = dWCurrent/mini_batch_size+self.regW*self.W
            dLCurrent = dLCurrent/mini_batch_size+self.regL*self.L
            dBCurrent = dBCurrent/mini_batch_size/self.regB*self.B

            #Mise a jour des poids et calcul des pas
            if strat == 'AdaGrad':
                eps=0.001 #Adagrad >0 ? cf Socher
                dWsHist += dWsCurrent*dWsCurrent
                dVHist += dVCurrent*dVCurrent
                dWHist += dWCurrent*dWCurrent
                dLHist += dLCurrent*dLCurrent
                dBHist += dBCurrent*dBCurrent

                dWsCurrent = eta*dWsCurrent/np.sqrt(dWsHist+eps)
                dWCurrent = eta*dWCurrent/np.sqrt(dWHist+eps)
                dVCurrent = eta*dVCurrent/np.sqrt(dVHist+eps)
                dLCurrent = eta*dLCurrent/np.sqrt(dLHist+eps)
                dBCurrent = eta*dBCurrent/np.sqrt(dBHist+eps)
            else:
                eps=0.001
                dWsHist = 0.9*dWsHist + 0.1*dWsCurrent*dWsCurrent
                dVHist = 0.9*dVHist + 0.1*dVCurrent*dVCurrent
                dWHist = 0.9*dWHist + 0.1*dWCurrent*dWCurrent
                dLHist = 0.9*dLHist + 0.1*dLCurrent*dLCurrent
                dBHist = 0.9*dBHist + 0.1*dBCurrent*dBCurrent

                dWsMask *= .7*(dWsPrev*dWsCurrent >= 0) + .5
                dWMask *= .7*(dWPrev*dWCurrent >= 0) + .5
                dVMask *= .7*(dVPrev*dVCurrent >= 0) + .5
                dLMask *= .7*(dLPrev*dLCurrent >= 0) + .5
                dBMask *= .7*(dBPrev*dBCurrent >= 0) + .5

                dWsCurrent = eta*dWsMask.clip(1e-6, 50, out=dWsMask)*dWsCurrent/np.sqrt(dWsHist+eps)
                dWCurrent = eta*dWMask.clip(1e-6, 50, out=dWMask)*dWCurrent/np.sqrt(dWHist+eps)
                dVCurrent = eta*dVMask.clip(1e-6, 20, out=dVMask)*dVCurrent/np.sqrt(dVHist+eps)
                dLCurrent = eta*dLMask.clip(1e-6, 20, out=dLMask)*dLCurrent/np.sqrt(dLHist+eps)
                dBCurrent = eta*dBMask.clip(1e-6, 20, out=dBMask)*dBCurrent/np.sqrt(dBHist+eps)

            #Calcul de la norme du gradient (critere d'arret)
            gradNorm = np.sum(np.abs(dWsCurrent))
            gradNorm += np.sum(np.abs(dWCurrent))
            gradNorm += np.sum(np.abs(dVCurrent))
            gradNorm += np.sum(np.abs(dLCurrent))
            gradNorm += np.sum(np.abs(dBCurrent))
            #Keep previous gradient
            dWsPrev = dWsCurrent
            dWPrev = dWCurrent
            dVPrev = dVCurrent
            dLPrev = dLCurrent
            dBPrev = dBCurrent

            #Descente
            self.Ws -= dWsCurrent
            self.W -= dWCurrent
            self.V -= dVCurrent
            self.L -= dLCurrent
            self.B -= dBCurrent

            #Maj de la condition d'arret
            if val_set != [] and (n_iter % n_check) == 0:
                currentError = self.error(val_set)
                currentError += self.regWs*np.sum(self.Ws*self.Ws)/2.0
                currentError += self.regW*np.sum(self.W*self.W)/2.0
                currentError += self.regV*np.sum(self.V*self.V)/2.0
                currentError += self.regL*np.sum(self.L*self.L)/2.0
                currentError += self.regB*np.sum(self.B*self.B)/2.0
                
                errVal.append(currentError)
                errMB.append(currentMbe)
                print('Error on validation set at iter {0} : {1} '
                      '(previous : {2})'.format(n_iter, currentError, prevError))
                print('Error on mini batch at iter {0} : {1} '
                      '(Gradient norm : {2})'.format(n_iter, currentMbe, gradNorm))
                prevError = currentError
                with open(save_tmp, 'wb') as output:
                    pickle.dump(errVal, output, -1)
                    pickle.dump(errMB, output, -1)
                
                #Early stopping
                minError = min(minError,currentError)
                glError = 100*((currentError/minError)-1.0)
                early_stop=(glError>2) #GL2 criterion
            else:
                print('Error on mini batch at iter {0} : {1} '
                      '(Gradient norm : {2})'.format(n_iter,currentMbe, gradNorm))
                errMB.append(currentMbe)

            #Maj iter
            n_iter += 1

        if val_set != []:
            print('Error on training set before and after training'
                  '({2} iter) : {0}->{1}\n'.format(iniError, currentError, n_iter))
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
                scAll += (np.argmax(n.ypred) == np.argmax(n.y))
            countRoot += 1
            n = X_tree.nodes[-1]
            scRoot += (np.argmax(n.ypred) == np.argmax(n.y))
        return scAll/countAll, scRoot/countRoot

    def score_binary(self, X_trees):
        '''
        Score sur les prediction MAP pos/neg
        '''
        countAll = 0
        countRoot = 0
        scAll = 0.0
        scRoot = 0.0
        M = np.mat('1,1,0,0,0;0,0,1,0,0;0,0,0,1,1')
        for X_tree in X_trees:
            self.forward_pass(X_tree)
            for n in X_tree.nodes:
                if np.argmax(n.y) != 2:
                    countAll += 1
                    scAll += np.argmax(M.dot(n.y)) == np.argmax(M.dot(n.ypred))
            n = X_tree.nodes[-1]
            if np.argmax(n.y) != 2:
                countRoot += 1
                scRoot += np.argmax(M.dot(n.y)) == np.argmax(M.dot(n.ypred))
        return scAll/countAll, scRoot/countRoot

    def score_binary2(self, X_trees):
        '''
        Score sur les prediction MAP pos/neg
        '''
        countRoot = 0
        scRoot = 0.0
        V = np.mat('0.1,0.3,0.5,0.7,0.9')
        for X_tree in X_trees:
            self.forward_pass(X_tree)
            n = X_tree.nodes[-1]
            countRoot += 1
            scRoot += ((V.dot(n.y) > .5) - (V.dot(n.ypred) > .5)) ** 2
        return scRoot/countRoot
        
    def check_derivative(self,X_tree,eps=1e-6):
        '''
        Fait une compaaison dérivee / differences finies
        '''
        error1=self.forward_pass(X_tree)
        dWs, dV, dW, dL = self.backward_pass(X_tree)
        dirW=np.random.uniform(size=self.W.shape)
        dirWs=np.random.uniform(size=self.Ws.shape)
        dirL=np.random.uniform(size=self.L.shape)
        dirV=np.random.uniform(size=self.V.shape)
        
        self.Ws+=eps*dirWs
        self.W+=eps*dirW
        self.L+=eps*dirL
        self.V+=eps*dirV
    
        
        error2=self.forward_pass(X_tree)
        diff=(error2-error1)/eps
        diff2=np.sum(dW*dirW)+np.sum(dWs*dirWs)+np.sum(dL*dirL)+np.sum(dV*dirV)
        
        return np.abs(diff-diff2)