import tensorflow_datasets as tfds
from functions.losses import svm_loss_naive, svm_loss_vectorized
from functions.models import LinearSVM
from functions.helper import get_CIFAR10_data
from random import randrange
import matplotlib.pyplot as plt
import numpy as np
import time


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

    # Evaluate the naive implementation of the loss we provided for you:

    # generate a random SVM weight matrix of small numbers
    W = np.random.randn(3073, 10) * 0.0001

    loss, grad = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    print('loss: %f' % (loss,))

    # Once you've implemented the gradient, recompute it with the code below
    # and gradient check it via the grad_check_sparse function above

    # Compute the loss and its gradient at W.

    # Numerically compute the gradient along several randomly chosen dimensions, and
    # compare them with your analytically computed gradient. The numbers should match
    # almost exactly along all dimensions.
    loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.0)
    f = lambda w: svm_loss_naive(w, X_dev, y_dev, 0.0)[0]
    grad_check_sparse(f, W, grad_naive)

    # do the gradient check once again with regularization turned on
    # you didn't forget the regularization gradient did you?
    loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 5e3)
    f = lambda w: svm_loss_naive(w, X_dev, y_dev, 5e3)[0]
    grad_check_sparse(f, W, grad_naive)

    # Next implement the function svm_loss_vectorized; for now only compute the loss;
    # we will implement the gradient in a moment.
    tic = time.time()
    loss_naive, grad_naive = svm_loss_naive(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print('Naive loss and gradient: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, grad_vectorized = svm_loss_vectorized(W, X_dev, y_dev, 0.000005)
    toc = time.time()
    print('Vectorized loss and gradient: %e computed in %fs' % (loss_vectorized, toc - tic))

    # The losses should match but your vectorized implementation should be much faster.
    print('loss difference: %f' % (loss_naive - loss_vectorized))

    # The loss is a single number, so it is easy to compare the values computed
    # by the two implementations. The gradient on the other hand is a matrix, so
    # we use the Frobenius norm to compare them.
    difference = np.linalg.norm(grad_naive - grad_vectorized, ord='fro')
    print('difference: %f' % difference)

    # In the class LinearClassifier, implement SGD in the function
    # LinearClassifier.train() and then run it with the code below.
    svm = LinearSVM()
    tic = time.time()

    loss_hist = svm.train(
        X_train, y_train, learning_rate=1e-7, reg=2.5e4,
        num_iters=1500, verbose=True)
    toc = time.time()
    print('That training took %fs' % (toc - tic))

    # A useful debugging strategy is to plot the loss as a function of
    # iteration number:
    plt.plot(loss_hist)
    plt.xlabel('Iteration number')
    plt.ylabel('Loss value')
    plt.savefig('results/loss_history.png')

    # Write the LinearSVM.predict function and evaluate the performance on both the
    # training and validation set
    y_train_pred = svm.predict(X_train)
    print('training accuracy: %f' % (np.mean(y_train == y_train_pred),))
    y_val_pred = svm.predict(X_val)
    print('validation accuracy: %f' % (np.mean(y_val == y_val_pred),))

    learning_rates = [5e-8, 5e-7, 5e-6]
    regularization_strengths = [2.5e4, 5e4]

    # results is dictionary mapping tuples of the form
    # (learning_rate, regularization_strength) to tuples of the form
    # (training_accuracy, validation_accuracy). The accuracy is simply the fraction
    # of data points that are correctly classified.
    results = {}
    best_val = -1  # The highest validation accuracy that we have seen so far.
    best_svm = None  # The LinearSVM object that achieved the highest validation rate.

    # Grid search for best hyper parameters (learning rate and regularization )
    paras_comb = [(lr, rs) for lr in learning_rates for rs in regularization_strengths]
    niters = 1000
    for (lr, rs) in paras_comb:
        svm = LinearSVM()
        loss_hist = svm.train(X_train, y_train, learning_rate=lr, reg=rs,
                              num_iters=niters, verbose=False)
        y_train_pred = svm.predict(X_train)
        y_val_pred = svm.predict(X_val)

        train_acc = np.mean(y_train == y_train_pred)
        test_acc = np.mean(y_val == y_val_pred)
        if test_acc > best_val:
            best_val = test_acc
            best_svm = svm
        results[(lr, rs)] = (train_acc, test_acc)

    # Print out results.
    for lr, reg in sorted(results):
        train_accuracy, val_accuracy = results[(lr, reg)]
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
            lr, reg, train_accuracy, val_accuracy))
    print('best validation accuracy achieved during cross-validation: %f' % best_val)

    # Visualize the learned weights for each class.
    # Depending on your choice of learning rate and regularization strength, these may
    # or may not be nice to look at.
    w = best_svm.W[:-1, :]  # strip out the bias
    w = w.reshape(32, 32, 3, 10)
    w_min, w_max = np.min(w), np.max(w)

    info = tfds.builder('cifar10').info
    classes = info.features['label'].names

    for i in range(10):
        plt.subplot(2, 5, i + 1)

        # Rescale the weights to be between 0 and 255
        wimg = 255.0 * (w[:, :, :, i].squeeze() - w_min) / (w_max - w_min)
        plt.imshow(wimg.astype('uint8'))
        plt.axis('off')
        plt.title(classes[i])
    plt.savefig("results/svm_weights_visual.png")
