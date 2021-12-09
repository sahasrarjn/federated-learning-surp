import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from parameters import Parameters


class DecentralizedSGD:
    def __init__(self, params:Parameters):
        self.params = params
        self.w_est = None
        self.w = None
        self.MM = self.create_mixing_matrix(params.topology, params.num_nodes)
    
    def create_mixing_matrix(self, topology:str, num_nodes:int):
        MM = None
        if topology == 'ring':
            MM = np.zeros((num_nodes, num_nodes))
            if num_nodes == 1:
                MM[0][0] = 1
            elif num_nodes == 2:
                MM = np.array([[.5, .5], [.5, .5]])
            else:
                weight = 1./3
                np.fill_diagonal(MM, weight)
                np.fill_diagonal(MM[1:], weight, wrap=False)
                np.fill_diagonal(MM[:,1:], weight, wrap=False)
                MM[0, num_nodes-1] = weight
                MM[num_nodes-1, 0] = weight
        elif topology == 'torus':
            assert int(np.sqrt(num_nodes))**2 == num_nodes
            G = nx.generators.lattice.grid_2d_graph(
                int(np.sqrt(num_nodes)), 
                int(np.sqrt(num_nodes)), 
                periodic = True)
            MM = nx.adjacency_matrix(G).toarray()
            np.fill_diagonal(MM, 1)
            MM = MM / 5
        elif topology == 'fully-connected':
            MM = np.ones((num_nodes, num_nodes), dtype=np.float64) / num_nodes
        elif topology == 'disconnected':
            MM = np.eye(num_nodes)
        else:
            raise Exception('DecentralizedSGD: Unknown topology')
        return MM

    def loss(self, X, y):
        w = self.w_est if self.w_est else self.w
        w = w.copy().mean(axis=1)

        if self.params.loss == 'mse':
            loss = np.mean((X@w - y)**2)
        elif self.params.loss == 'hinge':
            pass
        elif self.params.loss == 'logistic':
            pass 
        else:
            raise Exception('DecentralizedSGD: Unknown loss function')
        
    def sigmoid(self, w:float) -> float:
        # modify later for overflow
        return 1./(1+np.exp(-w))

    def predict(self, X, prob=False):
        ''' Predict function for binary classification '''
        w = self.w_est if self.w_est else self.w
        w = w.copy().mean(axis=1)
        logits = X @ w
        if prob:
            return self.sigmoid(logits)
        else:
            return (logits >= 0.).astype(np.int)

    def accuracy(self, X, y):
        pred = 2 * self.predict(X, prob=False) - 1
        return np.mean(pred == y)

    def fit(self, X_train, y_train):
        '''Decentralised training using Choco-SGD'''
        y = np.copy(y_train)
        losses = np.zeros(self.params.num_epochs+1)
        num_samples, num_features = X_train.shape

        
