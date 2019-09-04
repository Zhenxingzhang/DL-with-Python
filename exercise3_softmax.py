import numpy as np
from functions.losses import softmax_loss_naive
from functions.helper import get_CIFAR10_data


if __name__ == "__main__":
    # Load the raw CIFAR-10 data.
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    print('dev data shape: ', X_dev.shape)
    print('dev labels shape: ', y_dev.shape)

    # sanity check: the cross-entropy loss for predictions from random initial weights would be log(n)
    # -log(1/n) = log(n)

    # Generate a random softmax weight matrix and use it to compute the loss.
    W = np.random.randn(3073, 10) * 0.0001
    loss1, grad1 = softmax_loss_naive(W, X_dev, y_dev, 0.0)
    print(loss1)
