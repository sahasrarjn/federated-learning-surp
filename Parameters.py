class Parameters:
    """
    Parameters class.
    """
    def __init__(
        self,
        lr_init=None,
        lr_type=None,
        epoch_decay_lr=None,
        regularizer=None,
        num_nodes=1,
        topology='ring',
        algorithm='choco',
        quantize_algo=None,
        data_split=0.8,
        seed=None,
        loss='mse',
        num_epochs=5,
        distribute_data=True,
        distribute_data_method='random',
        choco_gamma=None,
        sparse_k=None,
        gossip_p=None,
        num_levels=None,
        plot=False,
        dump=False,
    ):

        assert lr_init is not None
        assert lr_type is not None
        if lr_type == 'decay':
            assert epoch_decay_lr is not None

        assert topology in ['ring', 'torus', 'fully-connected', 'disconnected']
        assert algorithm in ['plain', 'choco']
        assert loss in ['mse', 'logistic', 'hinge']
        assert distribute_data_method in ['random', 'sequential', 'label-sorted']
        assert quantize_algo in ['sparsification', 'random-gossip', 'full', 'random-quantization']
        assert data_split <= 1.0 and data_split >= 0.0
        assert num_nodes > 0
        if algorithm == 'choco':
            assert choco_gamma is not None
            assert quantize_algo is not None
        if quantize_algo == 'sparsification':
            assert sparse_k is not None
        if quantize_algo == 'random-gossip':
            assert gossip_p is not None
        if quantize_algo == 'random-quantization':
            assert num_levels is not None
        
        self.lr_init = lr_init
        self.lr_type = lr_type
        self.epoch_decay_lr = epoch_decay_lr
        self.regularizer = regularizer
        self.num_nodes = num_nodes
        self.topology = topology
        self.algorithm = algorithm
        self.quantize_algo = quantize_algo
        self.data_split = data_split
        self.seed = seed
        self.loss = loss
        self.num_epochs = num_epochs
        self.distribute_data = distribute_data
        self.distribute_data_method = distribute_data_method
        self.choco_gamma = choco_gamma
        self.sparse_k = sparse_k
        self.gossip_p = gossip_p
        self.num_levels = num_levels
        self.plot = plot
        self.dump = dump


    def __str__(self) -> str:
        return  f'Parameters:\n' \
                f'\tlr_init: {self.lr_init}\n' \
                f'\tlr_type: {self.lr_type}\n' \
                f'\tepoch_decay_lr: {self.epoch_decay_lr}\n' \
                f'\tRegularizer: {self.regularizer}\n' \
                f'\tNumber of nodes: {self.num_nodes}\n' \
                f'\tTopology: {self.topology}\n' \
                f'\tAlgorithm: {self.algorithm}\n' \
                f'\tQuantize algorithm: {self.quantize_algo}\n' \
                f'\tData split: {self.data_split}\n' \
                f'\tSeed: {self.seed}\n' \
                f'\tLoss: {self.loss}\n' \
                f'\tNumber of epochs: {self.num_epochs}\n' \
                f'\tDistribute data: {self.distribute_data}\n' \
                f'\tDistribute data method: {self.distribute_data_method}\n' \
                f'\tChoco gamma: {self.choco_gamma}\n' \
                f'\tSparse k: {self.sparse_k}\n' \
                f'\tGossip p: {self.gossip_p}\n' \
                f'\tNum levels: {self.num_levels}\n' \
                f'\tPlot: {self.plot}\n' \
                f'\tDump: {self.dump}\n'
