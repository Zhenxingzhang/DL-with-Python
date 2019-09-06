import tensorflow_datasets as tfds
import numpy as np
from random import randrange


def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=1000, num_dev=500):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the linear classifier. These are the same steps as we used for the
    SVM, but condensed to a single function.
    """
    # Load the raw CIFAR-10 data
    train_ds, test_ds = tfds.load("cifar10",
                                  split=[tfds.Split.TRAIN, tfds.Split.TEST],
                                  batch_size=-1)

    # generator that yields NumPy array records out of a tf.data.Dataset
    np_train_ds, np_test_ds = tfds.as_numpy(train_ds), tfds.as_numpy(test_ds)

    X_train, y_train = np_train_ds["image"], np_train_ds["label"]
    X_test, y_test = np_test_ds["image"], np_test_ds["label"]

    # subsample the data
    mask = list(range(num_training, num_training + num_validation))
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = list(range(num_training))
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = list(range(num_test))
    X_test = X_test[mask]
    y_test = y_test[mask]
    mask = np.random.choice(num_training, num_dev, replace=False)
    X_dev = X_train[mask]
    y_dev = y_train[mask]

    # Preprocessing: reshape the image data into rows
    X_train = np.reshape(X_train, (X_train.shape[0], -1)).astype(np.float64)
    X_val = np.reshape(X_val, (X_val.shape[0], -1)).astype(np.float64)
    X_test = np.reshape(X_test, (X_test.shape[0], -1)).astype(np.float64)
    X_dev = np.reshape(X_dev, (X_dev.shape[0], -1)).astype(np.float64)

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train = X_train - mean_image
    X_val = X_val - mean_image
    X_test = X_test - mean_image
    X_dev = X_dev - mean_image

    # add bias dimension and transform into columns
    X_train = np.hstack([X_train, np.ones((X_train.shape[0], 1))])
    X_val = np.hstack([X_val, np.ones((X_val.shape[0], 1))])
    X_test = np.hstack([X_test, np.ones((X_test.shape[0], 1))])
    X_dev = np.hstack([X_dev, np.ones((X_dev.shape[0], 1))])

    return X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev


def grad_check_sparse(f, x, analytic_grad, num_checks=10, h=1e-5):
    """
    Sample a few random elements and only return numerical
    gradients in these dimensions.
    """
    for i in range(num_checks):
        ix = tuple([randrange(m) for m in x.shape])

        oldval = x[ix]
        x[ix] = oldval + h  # increment by h
        fxph = f(x)  # evaluate f(x + h)
        x[ix] = oldval - h  # increment by h
        fxmh = f(x)  # evaluate f(x - h)
        x[ix] = oldval  # reset!! Do not forget!

        grad_numerical = (fxph - fxmh) / (2 * h)
        grad_analytic = analytic_grad[ix]
        rel_error = abs(grad_numerical - grad_analytic) / (abs(grad_numerical) + abs(grad_analytic))
        print('numerical: %f analytic: %f, relative error: %e' %
              (grad_numerical, grad_analytic, rel_error))