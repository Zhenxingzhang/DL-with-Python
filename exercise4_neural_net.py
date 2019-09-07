import numpy as np
from functions.models import TwoLayerNet
from functions.helper import eval_numerical_gradient, rel_error, get_CIFAR10_data

if __name__ == "__main__":

    # Create a small net and some toy data to check your implementations.
    # Note that we set the random seed for repeatable experiments.

    input_size = 4
    hidden_size = 10
    num_classes = 3
    num_inputs = 5


    def init_toy_model():
        np.random.seed(0)
        return TwoLayerNet(input_size, hidden_size, num_classes, std=1e-1)


    def init_toy_data():
        np.random.seed(1)
        X = 10 * np.random.randn(num_inputs, input_size)
        y = np.array([0, 1, 2, 2, 1])
        return X, y


    net = init_toy_model()
    X, y = init_toy_data()

    # Forward pass: compute scores
    scores = net.loss(X)
    print('Your scores:')
    print(scores)
    print()
    print('correct scores:')
    correct_scores = np.asarray([
        [-0.81233741, -1.27654624, -0.70335995],
        [-0.17129677, -1.18803311, -0.47310444],
        [-0.51590475, -1.01354314, -0.8504215],
        [-0.15419291, -0.48629638, -0.52901952],
        [-0.00618733, -0.12435261, -0.15226949]])
    print(correct_scores)
    print()

    # Forward pass: compute loss
    # The difference should be very small. We get < 1e-7
    print('Difference between your scores and correct scores:')
    print(np.sum(np.abs(scores - correct_scores)))

    # computes the data and regularization loss.
    loss, _ = net.loss(X, y, reg=0.05)
    correct_loss = 1.30378789133

    # should be very small, we get < 1e-12
    print('Difference between your loss and correct loss:')
    print(np.sum(np.abs(loss - correct_loss)))

    loss, grads = net.loss(X, y, reg=0.05)

    # these should all be less than 1e-8 or so
    for param_name in grads:
        f = lambda W: net.loss(X, y, reg=0.05)[0]
        param_grad_num = eval_numerical_gradient(f, net.params[param_name], verbose=False)
        print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grads[param_name])))

    net = init_toy_model()
    stats = net.train(X, y, X, y,
                      learning_rate=1e-1, reg=5e-6,
                      num_iters=100, verbose=True)

    print('Final training loss: ', stats['loss_history'][-1])

    # # plot the loss history
    # plt.plot(stats['loss_history'])
    # plt.xlabel('iteration')
    # plt.ylabel('training loss')
    # plt.title('Training Loss history')
    # plt.show()

    # Load the raw CIFAR-10 data.
    X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)

    input_size = 32 * 32 * 3
    hidden_size = 100
    num_classes = 10
    net = TwoLayerNet(input_size, hidden_size, num_classes)

    # Train the network
    stats = net.train(X_train, y_train, X_val, y_val,
                      num_iters=2000, batch_size=200,
                      learning_rate=1e-3, learning_rate_decay=0.95,
                      reg=0.25, verbose=True)

    # Predict on the validation set
    val_acc = (net.predict(X_val) == y_val).mean()
    print('Validation accuracy: ', val_acc)
