from re import L
from time import time
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from parameters import Parameters


class DecentralizedSGD:
    def __init__(self, params:Parameters):
        self.params = params
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
        assert MM.shape == (num_nodes, num_nodes)
        return MM

    def loss(self, X, y, log=False):
        w = self.w.copy().mean(axis=1)

        if log:
            # print("Pred:", X@w)
            # print("Actual:", y)
            print(X@w - y)

        if self.params.loss == 'mse':
            loss = np.mean((X@w - y)**2)
        elif self.params.loss == 'hinge':
            pass
        elif self.params.loss == 'logistic':
            pass 
        else:
            raise Exception('DecentralizedSGD: Unknown loss function')
        return loss

    def sigmoid(self, w:float) -> float:
        # modify later for overflow
        return 1./(1+np.exp(-w))

    def quantize(self, x):
        if self.params.quantize_algo == 'sparsification':
            q = np.zeros(x.shape)
            k = self.params.sparse_k
            # get top k dimension from x
            for i in range(x.shape[1]):
                idxs = np.argsort(np.abs(x[:,i]))[-k:]
                q[idxs, i] = x[idxs, i]
            return q.astype(np.float32)
        elif self.params.quantize_algo == 'random-gossip':
            pass
        elif self.params.quantize_algo == 'full':
            return x
        else:
            raise Exception('DecentralizedSGD: Unknown quantization algorithm')

    def predict(self, X, prob=False):
        ''' Predict function for binary classification '''
        w = self.w.copy().mean(axis=1)
        logits = X @ w
        if prob:
            return self.sigmoid(logits)
        else:
            return (logits >= 0.).astype(np.int)

    def accuracy(self, X, y):
        pred = 2 * self.predict(X) - 1
        return np.mean(pred == y)

    def fit(self, X_train, y_train):
        '''Decentralised training using Choco-SGD'''
        y = np.copy(y_train)
        losses = np.zeros(self.params.num_epochs+1)
        num_samples, num_features = X_train.shape

        if self.params.seed is not None:
            np.random.seed(self.params.seed)

        if self.w is None:
            self.w = np.random.normal(0, 0.1, num_features)   # initialize weights
            self.w = np.tile(self.w, (self.params.num_nodes, 1)).T
            self.w_hat = np.copy(self.w)
            assert self.w.shape == (num_features, self.params.num_nodes)

        # split data onto machines
        if self.params.distribute_data:
            num_samples_per_machine = int(num_samples / self.params.num_nodes)
            if self.params.distribute_data_method == 'random':
                idxs = np.array(range(num_samples))
                np.random.shuffle(idxs)
            elif self.params.distribute_data_method == 'sequential':
                idxs = np.arange(num_samples)
            elif self.params.distribute_data_method == 'label-sorted':
                idxs = np.argsort(y)
            
            indices = []
            for node in range(self.params.num_nodes-1):
                indices.append(idxs[node*num_samples_per_machine:(node+1)*num_samples_per_machine])
            indices.append(idxs[(self.params.num_nodes-1)*num_samples_per_machine:])
        else:
            num_samples_per_machine = num_samples
            indices = np.tile(np.arange(num_samples), (self.params.num_nodes, 1))

        print('Initial Loss: {:.4f}'.format(self.loss(X_train, y_train)))        

        train_start_time = time()
        for epoch in range(self.params.num_epochs):
            for iter in range(num_samples_per_machine):
                w_mid = np.zeros(self.w.shape)
                for node in range(self.params.num_nodes):
                    idx = indices[node][iter]
                    X = X_train[idx]
                    w = self.w[:, node]

                    # Gradient
                    if self.params.loss == 'mse':
                        grad = 2 * (X@w - y[idx]) * X
                    elif self.params.loss == 'hinge':
                        pass
                    elif self.params.loss == 'logistic':
                        pass
                    else:
                        raise Exception('DecentralizedSGD: Unknown loss function')
                    w_mid[:, node] = - self.params.lr * grad

                # w_mid = self.w - w_mid # w^{t+1/2} = w^{t} - \eta grad^{t}
                # print("w_mid\n", w_mid)

                # Decentralized Communication
                if self.params.algorithm == 'choco':
                    w_mid = self.w + w_mid                  # w_mid = w^{t+1/2}
                    q = self.quantize(w_mid - self.w_hat)   # q = q(w^{t+1/2} - w^{t})
                    self.w_hat += q                         # w^{t+1} = w^{t} + q

                    # Update step: w^{t+1} = w^{t+1/2} + \gamma q(w^{t+1/2} - w^{t})
                    self.w = w_mid + self.params.choco_gamma * \
                        (self.w_hat).dot(self.MM - np.eye(self.params.num_nodes)) 
                elif self.params.algorithm == 'plain':
                    self.w = (self.w + w_mid).dot(self.MM)
                else:
                    raise Exception('DecentralizedSGD: Unknown algorithm')
                
                # print('Loss: {:.4f}'.format(self.loss(X_train, y_train)))        
                    
            losses[epoch+1] = self.loss(X_train, y_train)
            print('Epoch: %d, Loss: %.4f' % (epoch+1, losses[epoch+1]))

        # print("Final loss: ", self.loss(X_train, y_train, log=True))
        train_end_time = time()
        print('Training time: %.2f seconds' % (train_end_time - train_start_time))

                
        