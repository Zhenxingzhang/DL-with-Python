import numpy as np
from functions.helper import eval_numerical_gradient, rel_error


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


def softmax_loss(scores, y):
    """
    Computes the loss and gradient for softmax classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    shifted_logits = scores - np.max(scores, axis=1, keepdims=True)
    Z = np.sum(np.exp(shifted_logits), axis=1, keepdims=True)
    log_probs = shifted_logits - np.log(Z)
    probs = np.exp(log_probs)
    N = scores.shape[0]
    loss = -np.sum(log_probs[np.arange(N), y]) / N
    dx = probs.copy()
    dx[np.arange(N), y] -= 1
    dx /= N
    return loss, dx


def svm_loss(x, y):
    """
    Computes the loss and gradient using for multiclass SVM classification.

    Inputs:
    - x: Input data, of shape (N, C) where x[i, j] is the score for the jth
      class for the ith input.
    - y: Vector of labels, of shape (N,) where y[i] is the label for x[i] and
      0 <= y[i] < C

    Returns a tuple of:
    - loss: Scalar giving the loss
    - dx: Gradient of the loss with respect to x
    """
    N = x.shape[0]
    correct_class_scores = x[np.arange(N), y]
    margins = np.maximum(0, x - correct_class_scores[:, np.newaxis] + 1.0)
    margins[np.arange(N), y] = 0
    loss = np.sum(margins) / N
    num_pos = np.sum(margins > 0, axis=1)
    dx = np.zeros_like(x)
    dx[margins > 0] = 1
    dx[np.arange(N), y] -= num_pos
    dx /= N
    return loss, dx


if __name__ == "__main__":
    np.random.seed(231)
    num_classes, num_inputs = 10, 50
    x = 0.001 * np.random.randn(num_inputs, num_classes)
    y = np.random.randint(num_classes, size=num_inputs)

    dx_num = eval_numerical_gradient(lambda x: svm_loss(x, y)[0], x, verbose=False)
    loss, dx = svm_loss(x, y)

    # Test svm_loss function. Loss should be around 9 and dx error should be around the order of e-9
    print('Testing svm_loss:')
    print('loss: ', loss)
    print('dx error: ', rel_error(dx_num, dx))

    dx_num = eval_numerical_gradient(lambda x: softmax_loss(x, y)[0], x, verbose=False)
    loss, dx = softmax_loss(x, y)

    # Test softmax_loss function. Loss should be close to 2.3 and dx error should be around e-8
    print('\nTesting softmax_loss:')
    print('loss: ', loss)
    print('dx error: ', rel_error(dx_num, dx))