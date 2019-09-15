import numpy as np
from functions.models import TwoLayerNetV2, FullyConnectedNet
from functions.helper import get_CIFAR10_data
from functions.solver import Solver

if __name__ == "__main__":

    # Load the (preprocessed) CIFAR10 data.
    # Load the raw CIFAR-10 data.
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    model = TwoLayerNetV2()
    solver = None

    # 50% accuracy on the validation set.                                        #

    data = {
        'X_train':  X_train, # training data
        'y_train':  y_train, # training labels
        'X_val':  X_val, # validation data
        'y_val':  y_val # validation labels
    }

    # solver = Solver(model, data,
    #                 update_rule='sgd',
    #                 optim_config={
    #                     'learning_rate': 1e-3,
    #                 },
    #                 lr_decay=0.95,
    #                 num_epochs=10, batch_size=100,
    #                 print_every=1000)
    #
    # solver.train()

    # TODO: Use a three-layer Net to overfit 50 training examples by
    # tweaking just the learning rate and initialization scale.

    num_train = 50
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    weight_scale = 1e-2  # Experiment with this!
    learning_rate = 1e-2  # Experiment with this!
    model = FullyConnectedNet([100, 100], input_dim=3 * 32 * 32, num_classes=10,
                              weight_scale=weight_scale, dtype=np.float64)

    initial_loss, _ = model.loss(small_data['X_train'], small_data['y_train'])
    print("initial loss: {}".format(initial_loss))

    # solver = Solver(model, small_data,
    #                 print_every=10, num_epochs=20, batch_size=25,
    #                 update_rule='sgd',
    #                 optim_config={
    #                     'learning_rate': learning_rate,
    #                 }
    #                 )
    # solver.train()

    # TODO: Use a five-layer Net to overfit 50 training examples by
    # tweaking just the learning rate and initialization scale.

    num_train = 50
    small_data = {
        'X_train': data['X_train'][:num_train],
        'y_train': data['y_train'][:num_train],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }

    learning_rate = 2e-4  # Experiment with this!
    weight_scale = 1e-1  # Experiment with this!
    model = FullyConnectedNet([100, 100, 100, 100, 100], input_dim=3 * 32 * 32,
                              num_classes=10, reg=0.0,
                              weight_scale=weight_scale, dtype=np.float64)

    initial_loss, _ = model.loss(small_data['X_train'], small_data['y_train'])
    print("initial loss: {}".format(initial_loss))
    # depends on the initial weights, the initial loss would not be always around log(num_classes)
    # mainly because the scores would not be small enough to make the initial probabilities be 1/num_classes anymore.

    solver = Solver(model, small_data,
                    print_every=10, num_epochs=20, batch_size=25,
                    update_rule='sgd_momentum',
                    optim_config={
                        'learning_rate': learning_rate,
                    }
                    )
    solver.train()
