import numpy as np


def svm_loss_naive(W, X, y, reg):
    """
    Structured SVM loss function, naive implementation (with loops).

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    loss = 0.0
    dW = np.zeros(W.shape)
    # why the margin is 1? answer here: http://cs231n.github.io/linear-classify/
    delta = 1
    batch_imgs = X

    num_sample = batch_imgs.shape[0]
    num_classes = W.shape[1]

    for i in range(num_sample):
        scores = batch_imgs[i].dot(W)
        y_label = y[i]
        target_label_score = scores[y_label]
        for j in range(num_classes):
            if y_label == j:
                continue
            margin = scores[j] - target_label_score + delta
            if margin > 0:
                loss += margin
                dW[:, j] += batch_imgs[i].T
                dW[:, y_label] -= batch_imgs[i].T

    loss /= num_sample
    dW *= 1.0/num_sample
    # Add regularization to the loss.
    loss += reg * np.sum(W ** 2)
    dW += reg * 2 * W
    return loss, dW


def svm_loss_vectorized(W, X, y, reg):
    """
    Structured SVM loss function, vectorized implementation.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Inputs and outputs are the same as svm_loss_naive.
    """
    loss = 0
    delta = 1

    num_sample = X.shape[0]
    num_classes = W.shape[1]

    scores = X.dot(W)
    yi_scores = scores[np.arange(num_sample), y].reshape(-1, 1)
    margins = np.maximum(0, scores - yi_scores + delta)
    margins[np.arange(num_sample), y] = 0

    loss += np.sum(margins)

    loss /= num_sample
    loss += reg * np.sum(W ** 2)

    masks = np.zeros(margins.shape)
    masks[margins > 0] = 1
    masks[np.arange(num_sample), y] = - (np.sum(masks, axis=1))

    dW = X.T.dot(masks)

    dW *= 1.0/num_sample
    dW += reg * 2 * W

    return loss, dW


def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    num_sample = X.shape[0]
    num_classes = W.shape[1]

    for i in np.arange(num_sample):
        scores = X[i].dot(W)
        scores -= scores.max()
        exp_scores = np.exp(scores)
        exp_sum = np.sum(exp_scores)
        probs = exp_scores / exp_sum
        loss += -np.log(probs[y[i]])

        for j in np.arange(num_classes):
            dLdp = - 1 / probs[j]
            dpdsum = - exp_scores[j] / exp_sum ** 2
            dSdy = exp_scores[j]
            dydW = X[i]

            dW[:, j] += dLdp * dpdsum * dSdy * dydW
        dW[:, y[i]] -= X[i]

    loss /= num_sample

    loss += reg * np.sum(W ** 2)

    dW /= num_sample
    dW += reg * 2 * W

    return loss, dW



def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    num_sample = X.shape[0]
    num_classes = W.shape[1]

    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    z = X.dot(W)
    z -= z.max()
    scores = np.exp(z)
    scores_sum = np.sum(scores, axis=1)
    probs = scores / scores_sum.reshape(-1, 1)

    yi_probs = probs[np.arange(num_sample), y]

    loss += np.sum(-np.log(yi_probs))

    loss /= num_sample

    loss += reg * np.sum(W ** 2)

    dW += X.T.dot(probs)

    binary = np.zeros((num_sample, num_classes))
    binary[np.arange(num_sample), y] = 1
    dW -= np.dot(X.T, binary)

    dW /= num_sample

    dW += reg * 2 * W

    return loss, dW

