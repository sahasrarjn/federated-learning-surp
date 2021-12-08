from DecentralizedSGD import DecentralizedSGD
from Parameters import Parameters

sgd = DecentralizedSGD(Parameters(
    lr=0.01,
    regularizer=0.1,
    num_nodes=25,
    topology='torus',
    algorithm='choco',
    data_split=0.8,
    seed=None,
))