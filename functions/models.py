from functions.losses import svm_loss_vectorized, softmax_loss_vectorized
import numpy as np


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


class LinearClassifier(object):
    def __init__(self):
        self.W = None

    def train(self, X, y, learning_rate=1e-3, reg=1e-5, num_iters=100,
              batch_size=200, verbose=False):
        """
                Train this linear classifier using stochastic gradient descent.

                Inputs:
                - X: A numpy array of shape (N, D) containing training data; there are N
                  training samples each of dimension D.
                - y: A numpy array of shape (N,) containing training labels; y[i] = c
                  means that X[i] has label 0 <= c < C for C classes.
                - learning_rate: (float) learning rate for optimization.
                - reg: (float) regularization strength.
                - num_iters: (integer) number of steps to take when optimizing
                - batch_size: (integer) number of training examples to use at each step.
                - verbose: (boolean) If true, print progress during optimization.

                Outputs:
                A list containing the value of the loss function at each training iteration.
                """
        num_train, dim = X.shape
        # assume y takes values 0...K-1 where K is number of classes
        num_classes = np.max(y) + 1
        if self.W is None:
            # lazily initialize W
            self.W = 0.001 * np.random.randn(dim, num_classes)

        # Run stochastic gradient descent to optimize W
        loss_history = []
        for it in range(num_iters):

            selected_batch = np.random.choice(num_train, size=batch_size, replace=True)
            X_batch = X[selected_batch, :]
            y_batch = y[selected_batch]

            # evaluate loss and gradient
            loss, grad = self.loss(X_batch, y_batch, reg)
            loss_history.append(loss)

            self.W -= learning_rate * grad

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

        return loss_history

    def predict(self, X):
        """
                Use the trained weights of this linear classifier to predict labels for
                data points.

                Inputs:
                - X: A numpy array of shape (N, D) containing training data; there are N
                  training samples each of dimension D.

                Returns:
                - y_pred: Predicted labels for the data in X. y_pred is a 1-dimensional
                  array of length N, and each element is an integer giving the predicted
                  class.
                """
        y_pred = np.argmax(X.dot(self.W), axis=1)
        return y_pred

    def loss(self, X_batch, y_batch, reg):
        pass


class LinearSVM(LinearClassifier):
    def loss(self, X_batch, y_batch, reg):
        return svm_loss_vectorized(self.W, X_batch, y_batch, reg)


class Softmax(LinearClassifier):
    """ A subclass that uses the Softmax + Cross-entropy loss function """

    def loss(self, X_batch, y_batch, reg):
        return softmax_loss_vectorized(self.W, X_batch, y_batch, reg)