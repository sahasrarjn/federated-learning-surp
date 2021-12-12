from sklearn import datasets
from parameters import Parameters
from dataset import Dataloader
from decentralizedSGD import DecentralizedSGD

model = DecentralizedSGD(Parameters(
    lr=0.1,
    regularizer=None,
    num_nodes=9,
    topology='ring',
    algorithm='choco',
    quantize_algo='full',
    data_split=0.8,
    seed=None,
    loss='mse',
    num_epochs=5,
    distribute_data=True,
    distribute_data_method='random',
    choco_gamma=0.1,
))

# DEBUG: Add regularizer

dataset = Dataloader(data_name='diabetes')
X_train, y_train, X_test, y_test = dataset.get_data()

model.fit(X_test, y_test)


