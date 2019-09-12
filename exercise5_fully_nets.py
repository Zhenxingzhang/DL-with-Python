from functions.models import TwoLayerNetV2
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

    ##############################################################################
    # TODO: Use a Solver instance to train a TwoLayerNet that achieves at least  #
    # 50% accuracy on the validation set.                                        #
    ##############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    data = {
        'X_train':  X_train, # training data
        'y_train':  y_train, # training labels
        'X_val':  X_val, # validation data
        'y_val':  y_val # validation labels
    }

    solver = Solver(model, data,
                    update_rule='sgd',
                    optim_config={
                        'learning_rate': 1e-3,
                    },
                    lr_decay=0.95,
                    num_epochs=10, batch_size=100,
                    print_every=1000)

    solver.train()
