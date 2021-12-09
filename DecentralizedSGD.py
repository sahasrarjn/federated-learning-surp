import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from parameters import Parameters


class DecentralizedSGD:
    def __init__(self, params:Parameters):
        self.params = params
        self.x_est = None
        self.x = None
        self.W = self.create_mixing_matrix(params.topology, params.num_nodes)
    
    def create_mixing_matrix(self, topology:str, num_nodes:int):
        W = None
        if topology == 'ring':
            W = np.zeros((num_nodes, num_nodes))
            if num_nodes == 1:
                W[0][0] = 1
            elif num_nodes == 2:
                W = np.array([[.5, .5], [.5, .5]])
            else:
                weight = 1./3
                np.fill_diagonal(W, weight)
                np.fill_diagonal(W[1:], weight, wrap=False)
                np.fill_diagonal(W[:,1:], weight, wrap=False)
                W[0, num_nodes-1] = weight
                W[num_nodes-1, 0] = weight
        elif topology == 'torus':
            assert int(np.sqrt(num_nodes))**2 == num_nodes
            G = nx.generators.lattice.grid_2d_graph(
                int(np.sqrt(num_nodes)), 
                int(np.sqrt(num_nodes)), 
                periodic = True)
            W = nx.adjacency_matrix(G).toarray()
            np.fill_diagonal(W, 1)
            W = W / 5
        elif topology == 'fully-connected':
            W = np.ones((num_nodes, num_nodes), dtype=np.float64) / num_nodes
        elif topology == 'disconnected':
            W = np.eye(num_nodes)
        else:
            raise Exception('DecentralizedSGD: Unknown topology')
        return W

    def loss(self, A, y):
        x = self.x_est if self.x_est else self.x
        x = x.copy().mean(axis=1)

        if self.params.loss == 'mse':
            loss = np.mean((A@x - y)**2)
        elif self.params.loss == 'hinge':
            pass
        elif self.params.loss == 'logistic':
            pass 
        else:
            raise Exception('DecentralizedSGD: Unknown loss function')
        
    def sigmoid(self, x:float) -> float:
        # modify later for overflow
        return 1./(1+np.exp(-x))

    def predict(self, A, prob=False):
        ''' Predict function for binary classification '''
        x = self.x_est if self.x_est else self.x
        x = x.copy().mean(axis=1)
        logits = A @ x
        if prob:
            return self.sigmoid(logits)
        else:
            return (logits >= 0.).astype(np.int)

    def accuracy(self, A, y):
        pred = 2 * self.predict(A, prob=False) - 1
        return np.mean(pred == y)

    def fit(self, A, y):
        '''Decentralised training using Choco-SGD'''
        pass
