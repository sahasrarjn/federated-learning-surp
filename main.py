from dataset import Dataloader
from Parameters import Parameters
from DecentralizedSGD import DecentralizedSGD

model = DecentralizedSGD(Parameters(
    lr_init=0.01,
    lr_type='inv',
    regularizer=None,
    num_nodes=9,
    topology='ring',
    algorithm='choco',
    quantize_algo='random-quantization',
    data_split=1,
    seed=None,
    loss='mse',
    num_epochs=10,
    distribute_data=True,
    distribute_data_method='random',
    choco_gamma=0.01,
    sparse_k=10,
    gossip_p=0.5,
    num_levels=5,
    plot=False,
    dump=True,
))

# DEBUG: Add regularizer

dataset = Dataloader(data_name='diabetes')
X_train, y_train, X_test, y_test = dataset.get_data()

model.fit(X_test, y_test)
