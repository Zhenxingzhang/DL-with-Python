import numpy as np
import tensorflow_datasets as tfds
from functions.models import KNearestNeighbor


def load_cifar10():
    first_10_percent_train = tfds.Split.TRAIN.subsplit(tfds.percent[:10])
    first_5_percent_test = tfds.Split.TEST.subsplit(tfds.percent[:5])

    train_ds, test_ds = tfds.load("cifar10",
                                  split=[first_10_percent_train, first_5_percent_test],
                                  batch_size=-1)
    # generator that yields NumPy array records out of a tf.data.Dataset
    np_train_ds, np_test_ds = tfds.as_numpy(train_ds), tfds.as_numpy(test_ds)

    x_train, y_train = np_train_ds["image"], np_train_ds["label"]
    x_test, y_test = np_test_ds["image"], np_test_ds["label"]

    return (x_train, y_train), (x_test, y_test)


def get_accuracy(y, y_pred):
    # Now implement the function predict_labels and run the code below:
    # We use k = 1 (which is Nearest Neighbor).

    # Compute and print the fraction of correctly predicted examples
    num_test = y.shape[0]
    num_correct = np.sum(y_pred == y)
    accuracy = float(num_correct) / num_test
    print('Got %d / %d correct => accuracy: %f' % (num_correct, num_test, accuracy))


# Let's compare how fast the implementations are
def time_function(f, *args):
    """
    Call a function f with args and return the time (in seconds) that it took to execute.
    """
    import time
    tic = time.time()
    dists = f(*args)
    toc = time.time()
    return toc - tic, dists


if __name__ == "__main__":
    (X_train, y_train), (X_test, y_test) = load_cifar10()

    # As a sanity check, we print out the size of the training and test data.
    print('Training data shape: ', X_train.shape)
    print('Training labels shape: ', y_train.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    # Create a kNN classifier instance.
    # Remember that training a kNN classifier is a noop:
    # the Classifier simply remembers the data and does no further processing
    classifier = KNearestNeighbor()
    classifier.train(X_train, y_train)

    two_loop_time, dists_two = time_function(classifier.compute_distances_two_loops, X_test)
    print('Two loop version took %f seconds' % two_loop_time)

    one_loop_time, dists_one = time_function(classifier.compute_distances_one_loop, X_test)
    print('One loop version took %f seconds' % one_loop_time)

    no_loop_time, dists_zero = time_function(classifier.compute_distances_no_loops, X_test)
    print('No loop version took %f seconds' % no_loop_time)

    # check that the distance matrix agrees with the one we computed before:
    difference = np.linalg.norm(dists_zero - dists_two, ord='fro')
    print('Difference was: %f' % (difference,))
    if difference < 0.001:
        print('Good! The distance matrices are the same')
    else:
        print('Uh-oh! The distance matrices are different')

    y_preds = classifier.predict(X_test, num_loops=0)

    get_accuracy(y_test, y_preds)

