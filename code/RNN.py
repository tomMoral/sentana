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
    def __init__(self, vocab={},dim=30,r=5):
        self.dim = dim

        #Initiate V, the tensor operator
        self.V = 1e-5*np.ones((dim, 2*dim, 2*dim))

        #Initiate W, the linear operator
        self.W = 1e-5*np.ones((dim, 2*dim))

        #Initiate Ws, the linear operator
        self.Ws = 1e-5*np.ones((5, dim))
        
        #Regularisation
        self.reg=0.1

        #Initiate L, the Lexicon representation
        self.L = np.random.uniform(-r, r, size=(len(vocab), dim))
        self.vocab = {}
        for i, w in enumerate(vocab):
            self.vocab[w] = i

        self.f = lambda X: np.tanh(X.T.dot(self.V).dot(X) + self.W.dot(X))
        self.grad = lambda f: 1-f**2

        self.y = lambda x: np.exp(self.Ws.dot(x)) / sum(np.exp(self.Ws.dot(x)))
    
    def save(self,saveFile):
        with open(saveFile,'wb') as output:
            pickle.dump(self.dim,output,-1)
            pickle.dump(self.V,output,-1)
            pickle.dump(self.W,output,-1)
            pickle.dump(self.Ws,output,-1)
            pickle.dump(self.reg,output,-1)
            pickle.dump(self.L,output,-1)
            pickle.dump(self.vocab,output,-1)
            
    def load(self,loadFile):
        with open(loadFile,'rb') as input:
            self.dim=pickle.load(input)
            self.V=pickle.load(input)
            self.W=pickle.load(input)
            self.Ws=pickle.load(input)
            self.reg=pickle.load(input)
            self.L=pickle.load(input)
            self.vocab=pickle.load(input)

    def error(self,val_set):
        '''
        Calcule l'erreur moyenne sur le set val_set avec les parametres actuel
        '''
        errorVal=0.0
        for X_tree in val_set:
            errorVal+=self.forward_pass(X_tree)
        return errorVal/len(val_set)
        
    def forward_pass(self, X_tree):
        '''
        Effectue la passe forward et calcule l'erreur actuelle pour l'arbre X_tree
        '''
        errorVal=0.0
        for n in X_tree.leaf:
            n.X = self.L[self.vocab[n.word]] # Met a jour le mot avec le Lexicon courant
            n.ypred=self.y(n.X) # Mise a jour du label predit
            errorVal+=-np.sum(n.y*np.log(n.ypred))

        for p, [a, b] in X_tree.parcours:
            aT = X_tree.nodes[a] #
            bT = X_tree.nodes[b] # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            pT = X_tree.nodes[p]
            if aT.order<bT.order:
                X = np.append(aT.X, bT.X)
            else:
                X = np.append(bT.X, aT.X)
            pT.X = self.f(X) # Misea jour du decripteur du parent
            pT.ypred=self.y(pT.X) # Mise a jour du label predit
            errorVal+=-np.sum(pT.y*np.log(pT.ypred))
        #E = sum([(self.y(n.X) - n.y) for n in X_tree.nodes])
        #print E
        #return self.Ws.dot(pT.X) -> Pas besoin de retourner le lbel, il faut le maj aussi?
        return errorVal
            
    def backward_pass(self,X_tree):
        '''
        Retourne le gradient du a l'erreur commise sur l'arbre X_tree
        Attention: suppose la forward_pass faite
        '''
        #Initialise les gradients
        dWs=np.zeros(self.Ws.shape)
        dV=np.zeros(self.V.shape)
        dW=np.zeros(self.W.shape)
        dL=np.zeros(self.L.shape)
        
        #Initialise les deltas
        for n in X_tree.nodes:
            n.d=self.Ws.T.dot(n.ypred-n.y)
            
        #Descend dans l arbre        
        for p,[a,b] in X_tree.parcours[::-1]:
            aT = X_tree.nodes[a] #
            bT = X_tree.nodes[b] # Recupere pour chaque triplet parent,enfant1/2 les noeuds
            pT = X_tree.nodes[p]
            gX=self.grad(pT.X)
            #Propagation des erreurs vers le bas
            if aT.order<bT.order: #Si aT est le noeud de gauche
                X = np.append(aT.X, bT.X)
                ddown=(self.W.T.dot(pT.d*gX)+(pT.d*gX).dot(self.V.dot(X)))
                aT.d+=ddown[:self.dim]
                bT.d+=ddown[self.dim:]
            else: #aT est a droite
                X = np.append(bT.X, aT.X)
                ddown=(self.W.T.dot(pT.d*gX)+(pT.d*gX).dot(self.V.dot(X)))           
                aT.d+=ddown[self.dim:]
                bT.d+=ddown[:self.dim]
            #Contribution aux gradients du pT
            dWs+=np.outer(pT.ypred-pT.y,pT.X)
            dV+=np.tensordot(pT.d*gX,np.outer(X,X),axes=0)
            dW+=np.outer(pT.d*gX,X)

        #Contribution des feuilles        
        for n in X_tree.leaf:
            dL[self.vocab[n.word]]+=n.d
            dWs+=np.outer(n.ypred-n.y,n.X)
            
        return dWs,dV,dW,dL

    def train(self,X_trees,learning_rate=1.0,mini_batch_size=25,warm_start=True,r=5,max_iter=1000,val_set=[],stop_threshold=10**(-10),n_check=100):
        '''
        Training avec AdaGrad (Dutchi et al.), prends en entrée une liste d'arbres X_trees
        '''
        #Remise à zero du modele
        if warm_start==False:
            dim=self.dim
            self.V = 1e-5*np.ones((dim, 2*dim, 2*dim))
            #Initiate W, the linear operator
            self.W = 1e-5*np.ones((dim, 2*dim))
            #Initiate Ws, the linear operator
            self.Ws = 1e-5*np.ones((5, 2*dim))
            #Initiate L, the Lexicon representation
            self.L = np.random.uniform(-r, r, size=(len(self.vocab), dim))
    
        #Liste pour erreurs
        errMB=[]
        errVal=[]
        #Condition d'arret
        n_iter=1
        gradNorm=1.0
        if val_set!=[]:
            prevError=self.error(val_set)
            iniError=prevError
            errVal.append(prevError)

        #Gradients pour AdaGrad
        eta=learning_rate
        dWsHist=np.ones(self.Ws.shape)
        dVHist=np.ones(self.V.shape)
        dWHist=np.ones(self.W.shape)
        dLHist=np.ones(self.L.shape)
        
        while (gradNorm>stop_threshold) and n_iter<max_iter:#Critere moins random
            mini_batch_samples=np.random.choice(X_trees,size=mini_batch_size)
            #Initialize gradients to 0
            dWsCurrent=np.zeros(self.Ws.shape)
            dVCurrent=np.zeros(self.V.shape)
            dWCurrent=np.zeros(self.W.shape)
            dLCurrent=np.zeros(self.L.shape)
            
            #Mini batch pour gradient
            currentMbe=0.0
            for X_tree in mini_batch_samples:
                currentMbe+=self.forward_pass(X_tree)
                dWs,dV,dW,dL=self.backward_pass(X_tree)
                dWsCurrent+=dWs
                dVCurrent+=dV
                dWCurrent+=dW
                dLCurrent+=dL
                
            currentMbe/=mini_batch_size
            #Division par le nombre de sample + regularisation
            dWsCurrent=dWsCurrent/mini_batch_size+self.reg*self.Ws
            dVCurrent=dVCurrent/mini_batch_size+self.reg*self.V
            dWCurrent=dWCurrent/mini_batch_size+self.reg*self.W
            dLCurrent=dLCurrent/mini_batch_size+self.reg*self.L
            
            #Mise a jour des poids pour l'AdaGrad
            dWsHist+=dWsCurrent*dWsCurrent
            dVHist+=dVCurrent*dVCurrent
            dWHist+=dWCurrent*dWCurrent
            dLHist+=dLCurrent*dLCurrent
            
            #Calcul des gradients apres AdaGrad
            dWsCurrent=eta*dWsCurrent/np.sqrt(dWsHist)
            dWCurrent=eta*dWCurrent/np.sqrt(dWHist)
            dVCurrent=eta*dVCurrent/np.sqrt(dVHist)
            dLCurrent=eta*dLCurrent/np.sqrt(dLHist)
            
            #Calcul de la norme du gradient (critere d'arret)
            gradNorm=np.abs(np.sum(dWsCurrent))
            gradNorm+=np.abs(np.sum(dWCurrent))
            gradNorm+=np.abs(np.sum(dVCurrent))
            gradNorm+=np.abs(np.sum(dLCurrent))
            
            #Descente
            self.Ws-=dWsCurrent
            self.W-=dWCurrent
            self.V-=dVCurrent
            self.L-=dLCurrent
            
            #Maj de la condition d'arret
            if val_set!=[] and (n_iter%n_check)==0:
                currentError=self.error(val_set)
                errVal.append(currentError)
                errMB.append(currentMbe)
                print('Error on validation set at iter {0} : {1} (previous : {2})'.format(n_iter,currentError,prevError))
                print('Error on mini batch at iter {0} : {1} (Gradient norm : {2})'.format(n_iter,currentMbe,gradNorm))
                prevError=currentError
            else:
                print('Error on mini batch at iter {0} : {1} (Gradient norm : {2})'.format(n_iter,currentMbe,gradNorm))
                errMB.append(currentMbe)
            
            #Maj iter
            n_iter+=1
        
        if val_set!=[]:
            print('Error on training set before and after training ({2} iter) : {0}->{1}\n'.format(iniError,currentError,n_iter))
        return errMB,errVal
        
    def score_fine(self,X_trees):
        '''
        Score sur les predictions MAP avec 5 label
        '''
        countAll=0
        countRoot=0
        scAll=0.0
        scRoot=0.0
        for X_tree in X_trees:
            self.forward_pass(X_tree)
            for n in X_tree.nodes:
                countAll+=1
                scAll+=(np.argmax(n.ypred)==np.argmax(n.y))
            countRoot+=1
            n=X_tree.nodes[-1]
            scRoot+=(np.argmax(n.ypred)==np.argmax(n.y))
        return scAll/countAll,scRoot/countRoot
        
    def score_binary(self,X_trees):
        '''
        Score sur les prediction MAP pos/neg
        '''
        countAll=0
        countRoot=0
        scAll=0.0
        scRoot=0.0
        M=np.mat('1,1,0,0,0;0,0,1,0,0;0,0,0,1,1')
        for X_tree in X_trees:
            self.forward_pass(X_tree)
            for n in X_tree.nodes:
                if np.argmax(n.y)!=2:
                    countAll+=1
                    scAll+=np.argmax(M.dot(n.y))==np.argmax(M.dot(n.ypred))
            n=X_tree.nodes[-1]
            if np.argmax(n.y)!=2:
                countRoot+=1
                scRoot+=np.argmax(M.dot(n.y))==np.argmax(M.dot(n.ypred))
        return scAll/countAll,scRoot/countRoot       
            