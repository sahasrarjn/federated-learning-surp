from parameters import Parameters
from dataset import Dataloader
from decentralizedSGD import DecentralizedSGD

model = DecentralizedSGD(Parameters(
    lr=0.01,
    regularizer=0.1,
    num_nodes=25,
    topology='torus',
    algorithm='choco',
    data_split=0.8,
    seed=None,
))

dataset = Dataloader('data/banana_data.csv')
X_train, y_train, X_test, y_test = dataset.get_data()
