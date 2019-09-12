from functions.losses import svm_loss_vectorized, softmax_loss_vectorized, softmax_loss
from functions.layers import relu_forward, affine_relu_forward, affine_relu_backward
from functions.helper import eval_numerical_gradient, rel_error
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


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network. The net has an input dimension of
    N, a hidden layer dimension of H, and performs classification over C classes.
    We train the network with a softmax loss function and L2 regularization on the
    weight matrices. The network uses a ReLU nonlinearity after the first fully
    connected layer.

    In other words, the network has the following architecture:

    input - fully connected layer - ReLU - fully connected layer - softmax

    The outputs of the second fully-connected layer are the scores for each class.
    """

    def __init__(self, input_size, hidden_size, output_size, std=1e-4):
        """
        Initialize the model. Weights are initialized to small random values and
        biases are initialized to zero. Weights and biases are stored in the
        variable self.params, which is a dictionary with the following keys:

        W1: First layer weights; has shape (D, H)
        b1: First layer biases; has shape (H,)
        W2: Second layer weights; has shape (H, C)
        b2: Second layer biases; has shape (C,)

        Inputs:
        - input_size: The dimension D of the input data.
        - hidden_size: The number of neurons H in the hidden layer.
        - output_size: The number of classes C.
        """
        self.params = dict()
        self.params['W1'] = std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def loss(self, X, y=None, reg=0.0):
        """
        Compute the loss and gradients for a two layer fully connected neural
        network.

        Inputs:
        - X: Input data of shape (N, D). Each X[i] is a training sample.
        - y: Vector of training labels. y[i] is the label for X[i], and each y[i] is
          an integer in the range 0 <= y[i] < C. This parameter is optional; if it
          is not passed then we only return scores, and if it is passed then we
          instead return the loss and gradients.
        - reg: Regularization strength.

        Returns:
        If y is None, return a matrix scores of shape (N, C) where scores[i, c] is
        the score for class c on input X[i].

        If y is not None, instead return a tuple of:
        - loss: Loss (data loss and regularization loss) for this batch of training
          samples.
        - grads: Dictionary mapping parameter names to gradients of those parameters
          with respect to the loss function; has the same keys as self.params.
        """
        # Unpack variables from the params dictionary
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        N, D = X.shape
        H = W2.shape[0]
        C = W2.shape[1]

        softmax_loss = 0.0
        # Compute the forward pass
        grads = dict()

        layer1_scores = np.hstack((X, np.ones([N, 1]))).dot(np.vstack((W1, b1)))
        layer1_acts = relu_forward(layer1_scores)
        layer2_scores = np.hstack((layer1_acts, np.ones([N, 1]))).dot(np.vstack((W2, b2)))

        # If the targets are not given then jump out, we're done
        if y is None:
            return layer2_scores

        layer2_exp = np.exp(layer2_scores)
        layer2_sum = np.sum(layer2_exp, axis=1)
        yi_scores = layer2_scores[np.arange(N), y]

        softmax_loss += np.log(layer2_sum).sum()
        softmax_loss -= yi_scores.sum()
        softmax_loss /= N

        softmax_loss += reg * np.sum(np.vstack((W1, b1)) * np.vstack((W1, b1)))
        softmax_loss += reg * np.sum(np.vstack((W2, b2)) * np.vstack((W2, b2)))

        grads_log = 1 / layer2_sum
        grads_exp = grads_log.reshape(-1, 1) * layer2_exp

        grads_2 = np.hstack((layer1_acts, np.ones([N, 1]))).T.dot(grads_exp)
        binary = np.zeros((N, C))
        binary[np.arange(N), y] = 1
        grads_2 -= np.dot(np.hstack((layer1_acts, np.ones([N, 1]))).T, binary)

        grads_2 *= 1/N
        grads_2 += 2 * reg * np.vstack((W2, b2))

        grads['W2'] = grads_2[:-1, :]
        grads['b2'] = grads_2[-1, :]

        grad_1_scores = grads_exp.dot(W2.T)
        relu_derivative = np.heaviside(layer1_acts, 0)
        grads_1_acts = grad_1_scores * relu_derivative  # (N, H)
        grads_1 = np.hstack((X, np.ones([N,1]))).T.dot(grads_1_acts) # (D+1, H)

        binary = np.zeros((N, C))
        binary[np.arange(N), y] = 1
        grad_hidden_scores = np.dot(binary, W2.T)  # (N,C) * (H, C).T == (N, H)
        grad_act_scores = grad_hidden_scores * relu_derivative  # (N, H)
        grads_1 -= np.hstack((X, np.ones([N, 1]))).T.dot(grad_act_scores)

        grads_1 *= 1/N
        grads_1 += 2 * reg * np.vstack((W1, b1))

        grads['W1'] = grads_1[:-1, :]
        grads['b1'] = grads_1[-1, :]

        return softmax_loss, grads

    def train(self,
              X,
              y,
              X_val,
              y_val,
              learning_rate=1e-3,
              learning_rate_decay=0.95,
              reg=5e-6,
              num_iters=100,
              batch_size=200,
              verbose=False):
        """
        Train this neural network using stochastic gradient descent.

        Inputs:
        - X: A numpy array of shape (N, D) giving training data.
        - y: A numpy array f shape (N,) giving training labels; y[i] = c means that
          X[i] has label c, where 0 <= c < C.
        - X_val: A numpy array of shape (N_val, D) giving validation data.
        - y_val: A numpy array of shape (N_val,) giving validation labels.
        - learning_rate: Scalar giving learning rate for optimization.
        - learning_rate_decay: Scalar giving factor used to decay the learning rate
          after each epoch.
        - reg: Scalar giving regularization strength.
        - num_iters: Number of steps to take when optimizing.
        - batch_size: Number of training examples to use per step.
        - verbose: boolean; if true print progress during optimization.
        """
        num_train = X.shape[0]
        iterations_per_epoch = max(num_train / batch_size, 1)

        # Use SGD to optimize the parameters in self.model
        loss_history = []
        train_acc_history = []
        val_acc_history = []

        for it in range(num_iters):

            selected_batch = np.random.choice(num_train, size=batch_size, replace=True)
            X_batch = X[selected_batch, :]
            y_batch = y[selected_batch]

            # Compute loss and gradients using the current minibatch
            loss, grads = self.loss(X_batch, y=y_batch, reg=reg)
            loss_history.append(loss)

            self.params["W2"] -= learning_rate * grads["W2"]
            self.params["b2"] -= learning_rate * grads["b2"]
            self.params["W1"] -= learning_rate * grads["W1"]
            self.params["b1"] -= learning_rate * grads["b1"]

            if verbose and it % 100 == 0:
                print('iteration %d / %d: loss %f' % (it, num_iters, loss))

            # Every epoch, check train and val accuracy and decay learning rate.
            if it % iterations_per_epoch == 0:
                # Check accuracy
                train_acc = (self.predict(X_batch) == y_batch).mean()
                val_acc = (self.predict(X_val) == y_val).mean()
                train_acc_history.append(train_acc)
                val_acc_history.append(val_acc)

                # Decay learning rate
                learning_rate *= learning_rate_decay

        return {
            'loss_history': loss_history,
            'train_acc_history': train_acc_history,
            'val_acc_history': val_acc_history,
        }

    def predict(self, X):
        """
        Use the trained weights of this two-layer network to predict labels for
        data points. For each data point we predict scores for each of the C
        classes, and assign each data point to the class with the highest score.

        Inputs:
        - X: A numpy array of shape (N, D) giving N D-dimensional data points to
          classify.

        Returns:
        - y_pred: A numpy array of shape (N,) giving predicted labels for each of
          the elements of X. For all i, y_pred[i] = c means that X[i] is predicted
          to have class c, where 0 <= c < C.
        """
        scores = self.loss(X)
        y_hat = np.argmax(scores, axis=1)

        return y_hat


class TwoLayerNetV2(object):
    def __init__(self, input_dim=3 * 32 * 32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg

        self.params['W1'] = np.random.normal(0, weight_scale, (input_dim, hidden_dim))
        self.params['b1'] = np.zeros((hidden_dim,))

        self.params['W2'] = np.random.normal(0, weight_scale, (hidden_dim, num_classes))
        self.params['b2'] = np.zeros((num_classes,))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """

        l1_out, l1_cache = affine_relu_forward(X, self.params['W1'], self.params['b1'])
        l2_out, l2_cache = affine_relu_forward(l1_out, self.params['W2'], self.params['b2'])

        if y is None:
            return l2_out

        loss, grads = 0, {}

        loss, dout = softmax_loss(l2_out, y)

        loss += 0.5 * self.reg * np.sum(self.params['W2'] * self.params['W2'])

        loss += 0.5 * self.reg * np.sum(self.params['W1'] * self.params['W1'])

        dout_l2, grads['W2'], grads['b2'] = affine_relu_backward(dout, l2_cache)

        grads['W2'] += self.reg * self.params['W2']

        dout_l1, grads['W1'], grads['b1'] = affine_relu_backward(dout_l2, l1_cache)

        grads['W1'] += self.reg * self.params['W1']

        return loss, grads


if __name__ == "__main__":
    np.random.seed(231)
    N, D, H, C = 3, 5, 50, 7
    X = np.random.randn(N, D)
    y = np.random.randint(C, size=N)

    std = 1e-3
    model = TwoLayerNetV2(input_dim=D, hidden_dim=H, num_classes=C, weight_scale=std)

    print('Testing initialization ... ')
    W1_std = abs(model.params['W1'].std() - std)
    b1 = model.params['b1']
    W2_std = abs(model.params['W2'].std() - std)
    b2 = model.params['b2']
    assert W1_std < std / 10, 'First layer weights do not seem right'
    assert np.all(b1 == 0), 'First layer biases do not seem right'
    assert W2_std < std / 10, 'Second layer weights do not seem right'
    assert np.all(b2 == 0), 'Second layer biases do not seem right'

    print('Testing test-time forward pass ... ')
    model.params['W1'] = np.linspace(-0.7, 0.3, num=D * H).reshape(D, H)
    model.params['b1'] = np.linspace(-0.1, 0.9, num=H)
    model.params['W2'] = np.linspace(-0.3, 0.4, num=H * C).reshape(H, C)
    model.params['b2'] = np.linspace(-0.9, 0.1, num=C)
    X = np.linspace(-5.5, 4.5, num=N * D).reshape(D, N).T
    scores = model.loss(X)
    correct_scores = np.asarray(
        [[11.53165108, 12.2917344, 13.05181771, 13.81190102, 14.57198434, 15.33206765, 16.09215096],
         [12.05769098, 12.74614105, 13.43459113, 14.1230412, 14.81149128, 15.49994135, 16.18839143],
         [12.58373087, 13.20054771, 13.81736455, 14.43418138, 15.05099822, 15.66781506, 16.2846319]])
    scores_diff = np.abs(scores - correct_scores).sum()
    assert scores_diff < 1e-6, 'Problem with test-time forward pass'

    print('Testing training loss (no regularization)')
    y = np.asarray([0, 5, 1])
    loss, grads = model.loss(X, y)
    correct_loss = 3.4702243556
    assert abs(loss - correct_loss) < 1e-10, 'Problem with training-time loss'

    model.reg = 1.0
    loss, grads = model.loss(X, y)
    correct_loss = 26.5948426952
    assert abs(loss - correct_loss) < 1e-10, 'Problem with regularization loss'

    # Errors should be around e-7 or less
    for reg in [0.0, 0.7]:
        print('Running numeric gradient check with reg = ', reg)
        model.reg = reg
        loss, grads = model.loss(X, y)

        for name in sorted(grads):
            f = lambda _: model.loss(X, y)[0]
            grad_num = eval_numerical_gradient(f, model.params[name], verbose=False)
            print('%s relative error: %.2e' % (name, rel_error(grad_num, grads[name])))