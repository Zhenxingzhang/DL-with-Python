import numpy as np
import tensorflow_datasets as tfds


class KNearestNeighbor(object):
    """ a kNN classifier with L2 distance """

    def __init__(self):
        self.X_train = None
        self.y_train = None

    def train(self, X, y):
        """
        Train the classifier. For k-nearest neighbors this is just
        memorizing the training data.

        Inputs:
        - X: A numpy array of shape (num_train, D) containing the training data
          consisting of num_train samples each of dimension D.
        - y: A numpy array of shape (N,) containing the training labels, where
             y[i] is the label for X[i].
        """
        self.X_train = X
        self.y_train = y

    def predict(self, X, k=1, num_loops=0):
        """
                Predict labels for test data using this classifier.

                Inputs:
                - X: A numpy array of shape (num_test, D) containing test data consisting
                     of num_test samples each of dimension D.
                - k: The number of nearest neighbors that vote for the predicted labels.
                - num_loops: Determines which implementation to use to compute distances
                  between training points and testing points.

                Returns:
                - y: A numpy array of shape (num_test,) containing predicted labels for the
                  test data, where y[i] is the predicted label for the test point X[i].
                """
        if num_loops == 0:
            dists = self.compute_distances_no_loops(X)
        elif num_loops == 1:
            dists = self.compute_distances_one_loop(X)
        elif num_loops == 2:
            dists = self.compute_distances_two_loops(X)
        else:
            raise ValueError('Invalid value %d for num_loops' % num_loops)

        return self._predict_labels(dists, k=k)

    def _predict_labels(self, dists, k=1):
        """
        Given a matrix of distances between test points and training points,
        predict a label for each test point.

        Inputs:
        - dists: A numpy array of shape (num_test, num_train) where dists[i, j]
          gives the distance betwen the ith test point and the jth training point.

        Returns:
        - y: A numpy array of shape (num_test,) containing predicted labels for the
          test data, where y[i] is the predicted label for the test point X[i].
        """
        num_test = dists.shape[0]
        y_pred = np.zeros(num_test)
        for i in range(num_test):
            # A list of length k storing the labels of the k nearest neighbors to
            # the ith test point.
            closest_y = [self.y_train[idx] for idx in np.argsort(dists[i])[:k]]
            y_pred[i] = self.get_most_common_label(closest_y)
        return y_pred

    def compute_distances_no_loops(self, X):
        x_train = self.X_train.reshape(self.X_train.shape[0], -1).astype(np.float64)
        x_test = X.reshape([X.shape[0], -1]).astype(np.float64)

        return np.sqrt(np.sum(x_test ** 2, axis=1).reshape(-1, 1) + np.sum(x_train ** 2, axis=1).reshape(-1, 1).T - 2 * np.dot(x_test, x_train.T))

    def compute_distances_one_loop(self, X):
        x_train = self.X_train.reshape(self.X_train.shape[0], -1).astype(np.float64)
        x_test = X.reshape([X.shape[0], -1]).astype(np.float64)
        dists = np.zeros([x_test.shape[0], x_train.shape[0]], dtype=np.float64)

        for i in range(x_test.shape[0]):
            dists[i, :] = np.sqrt(np.sum((x_test[i]-x_train)**2, axis=1))
        return dists

    def compute_distances_two_loops(self, X):
        x_train = self.X_train.reshape(self.X_train.shape[0], -1).astype(np.float64)
        x_test = X.reshape([X.shape[0], -1]).astype(np.float64)

        print(x_train.shape, x_test.shape)

        dists = np.zeros([x_test.shape[0], x_train.shape[0]], dtype=np.float64)

        for i in range(x_test.shape[0]):
            # print("current test : {}".format(i))
            for j in range(x_train.shape[0]):
                dists[i][j] = self._euclidean_dist(x_test[i], x_train[j])

        return dists

        # Find the most common label in the list closest_y of labels.
        # Store this label in y_pred[i]. Break ties by choosing the smaller label.
    def get_most_common_label(self, label_list):
        counter = {}
        for label in label_list:
            counter[label] = counter.get(label, 0) + 1
        counter_tuples = [(k, v) for k, v in counter.items()]
        counter_tuples.sort(key=lambda x: (-x[1], x[0]))
        return counter_tuples[0][0]

    def _euclidean_dist(self, v1, v2):
        return np.sqrt(np.sum(((v1 - v2) ** 2)))


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

