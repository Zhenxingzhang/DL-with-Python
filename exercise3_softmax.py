import numpy as np
import time
from functions.losses import softmax_loss_naive, softmax_loss_vectorized
from functions.helper import get_CIFAR10_data
from functions.helper import grad_check_sparse
from functions.models import Softmax
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt

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
    tic = time.time()
    loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.0)
    toc = time.time()
    print('naive loss: %e computed in %fs' % (loss_naive, toc - tic))

    tic = time.time()
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.0)
    toc = time.time()
    print('vectorized loss: %e computed in %fs' % (loss_vectorized, toc - tic))

    print('sanity check: %f' % (-np.log(1.0 / 10)))

    # As we did for the SVM, use numeric gradient checking as a debugging tool.
    # The numeric gradient should be close to the analytic gradient.
    loss_naive, grad_naive = softmax_loss_naive(W, X_dev, y_dev, 0.000005)
    f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.000005)[0]
    grad_numerical = grad_check_sparse(f, W, grad_naive, 10)

    # similar to SVM case, do another gradient check with regularization
    loss_vectorized, grad_vectorized = softmax_loss_vectorized(W, X_dev, y_dev, 0.000005)
    f = lambda w: softmax_loss_naive(w, X_dev, y_dev, 0.000005)[0]
    grad_numerical = grad_check_sparse(f, W, grad_vectorized, 10)

    # Use the validation set to tune hyperparameters (regularization strength and
    # learning rate). You should experiment with different ranges for the learning
    # rates and regularization strengths; if you are careful you should be able to
    # get a classification accuracy of over 0.35 on the validation set.
    results = {}
    best_val = -1
    best_softmax = None

    learning_rates = [ 5e-7]
    regularization_strengths = [2.5e4]

    paras_grid = [(lr, rs) for lr in learning_rates for rs in regularization_strengths]
    niters = 1000
    for (lr, reg) in paras_grid:
        softmax = Softmax()
        loss_hist = softmax.train(X_train, y_train, learning_rate=lr, reg=reg,
                                  num_iters=niters, verbose=False)
        y_train_pred = softmax.predict(X_train)
        y_val_pred = softmax.predict(X_val)

        train_acc = np.mean(y_train == y_train_pred)
        test_acc = np.mean(y_val == y_val_pred)
        print('lr %e reg %e train accuracy: %f val accuracy: %f' % (
            lr, reg, train_acc, test_acc))
        if test_acc > best_val:
            best_val = test_acc
            best_softmax = softmax
        results[(lr, reg)] = (train_acc, test_acc)

    # evaluate on test set
    # Evaluate the best softmax on test set
    y_test_pred = best_softmax.predict(X_test)
    test_accuracy = np.mean(y_test == y_test_pred)
    print('softmax on raw pixels final test set accuracy: %f' % (test_accuracy,))

    # Visualize the learned weights for each class
    w = best_softmax.W[:-1, :]  # strip out the bias
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
    plt.savefig("results/softmax_weights_visual.png")
