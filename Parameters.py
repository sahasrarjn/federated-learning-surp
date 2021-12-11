class Parameters:
    """
    Parameters class.
    """
    def __init__(
        self,
        lr=None,
        regularizer=None,
        num_nodes=1,
        topology='ring',
        algorithm='choco',
        data_split=0.8,
        seed=None,
        loss='mse',
        num_epochs=5,
        distribute_data=True,
        distribute_data_method='random',
        choco_gamma=None,
    ):

        assert topology in ['ring', 'torus', 'fully-connected', 'disconnected']
        assert algorithm in ['plain', 'choco']
        assert loss in ['mse', 'logistic', 'hinge']
        assert distribute_data_method in ['random', 'sequential', 'label-sorted']
        assert data_split <= 1.0 and data_split >= 0.0
        assert num_nodes > 0
        assert (choco_gamma is None) or (algorithm != 'choco')

        self.lr = lr
        self.regularizer = regularizer
        self.num_nodes = num_nodes
        self.topology = topology
        self.algorithm = algorithm
        self.data_split = data_split
        self.seed = seed
        self.loss = loss
        self.num_epochs = num_epochs
        self.distribute_data = distribute_data
        self.distribute_data_method = distribute_data_method


    def __str__(self) -> str:
        return  f'Parameters:\n' \
                f'\tLearning rate: {self.lr}\n' \
                f'\tRegularizer: {self.regularizer}\n' \
                f'\tNumber of nodes: {self.num_nodes}\n' \
                f'\tTopology: {self.topology}\n' \
                f'\tAlgorithm: {self.algorithm}\n' \
                f'\tData split: {self.data_split}\n' \
                f'\tSeed: {self.seed}\n' \
                f'\tLoss: {self.loss}\n' \
                f'\tNumber of epochs: {self.num_epochs}\n' \
                f'\tDistribute data: {self.distribute_data}\n' \
                f'\tDistribute data method: {self.distribute_data_method}\n'