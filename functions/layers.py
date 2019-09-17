import numpy as np
from functions.helper import rel_error, eval_numerical_gradient_array, print_mean_std


def relu_forward(inputs):

    # The ReLU unit simply passes its argument forward if positive,
    # but replaces it with zero if negative.

    # The maximum function selects the bigger of two inputs, element-wise:
    return np.maximum(inputs, 0), inputs


def relu_backward(dout, cache):
    """
    Computes the backward pass for a layer of rectified linear units (ReLUs).

    Input:
    - dout: Upstream derivatives, of any shape
    - cache: Input x, of same shape as dout

    Returns:
    - dx: Gradient with respect to x
    """
    x = cache

    dx = np.heaviside(x, 0) * dout

    return dx


def affine_forward(x, w, b):
    """
    Computes the forward pass for an affine (fully-connected) layer.

    The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
    examples, where each example x[i] has shape (d_1, ..., d_k). We will
    reshape each input into a vector of dimension D = d_1 * ... * d_k, and
    then transform it to an output vector of dimension M.

    Inputs:
    - x: A numpy array containing input data, of shape (N, d_1, ..., d_k)
    - w: A numpy array of weights, of shape (D, M)
    - b: A numpy array of biases, of shape (M,)

    Returns a tuple of:
    - out: output, of shape (N, M)
    - cache: (x, w, b)
    """
    rows = x.reshape(x.shape[0], -1)
    out = rows.dot(w) + b.reshape(1, -1)
    cache = (x, w, b)
    return out, cache


def affine_backward(dout, cache):
    """
    Computes the backward pass for an affine layer.

    Inputs:
    - dout: Upstream derivative, of shape (N, M)
    - cache: Tuple of:
      - x: Input data, of shape (N, d_1, ... d_k)
      - w: Weights, of shape (D, M)
      - b: Biases, of shape (M,)

    Returns a tuple of:
    - dx: Gradient with respect to x, of shape (N, d1, ..., d_k)
    - dw: Gradient with respect to w, of shape (D, M)
    - db: Gradient with respect to b, of shape (M,)
    """
    x, w, b = cache

    x_rows = x.reshape(x.shape[0], -1)

    # (N, D)
    dx = dout.dot(w.T).reshape(x.shape)
    # (D, M)
    dw = x_rows.T.dot(dout)
    # (M,)
    db = dout.sum(axis=0)

    return dx, dw, db


def affine_relu_forward(x, w, b):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    a, fc_cache = affine_forward(x, w, b)
    # print("activation mean: {} and std: {}".format(np.mean(a), np.std(a)))
    out, relu_cache = relu_forward(a)
    cache = (fc_cache, relu_cache)
    return out, cache


def affine_relu_backward(dout, cache):
    """
    Backward pass for the affine-relu convenience layer
    """
    fc_cache, relu_cache = cache
    da = relu_backward(dout, relu_cache)
    dx, dw, db = affine_backward(da, fc_cache)
    return dx, dw, db


def batchnorm_forward(x, gamma, beta, bn_param):
    """
        Forward pass for batch normalization.

        During training the sample mean and (uncorrected) sample variance are
        computed from minibatch statistics and used to normalize the incoming data.
        During training we also keep an exponentially decaying running mean of the
        mean and variance of each feature, and these averages are used to normalize
        data at test-time.

        At each timestep we update the running averages for mean and variance using
        an exponential decay based on the momentum parameter:

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_var

        Note that the batch normalization paper suggests a different test-time
        behavior: they compute sample mean and variance for each feature using a
        large number of training images rather than using a running average. For
        this implementation we have chosen to use running averages instead since
        they do not require an additional estimation step; the torch7
        implementation of batch normalization also uses running averages.

        Input:
        - x: Data of shape (N, D)
        - gamma: Scale parameter of shape (D,)
        - beta: Shift paremeter of shape (D,)
        - bn_param: Dictionary with the following keys:
          - mode: 'train' or 'test'; required
          - eps: Constant for numeric stability
          - momentum: Constant for running mean / variance.
          - running_mean: Array of shape (D,) giving running mean of features
          - running_var Array of shape (D,) giving running variance of features

        Returns a tuple of:
        - out: of shape (N, D)
        - cache: A tuple of values needed in the backward pass
        """
    mode = bn_param["mode"]
    eps = bn_param.get("eps", 1e-5)
    momentum = bn_param.get("momentum", 0.9)

    N, D = x.shape
    running_mean = bn_param.get("running_mean", np.zeros(D, dtype=x.dtype))
    running_var = bn_param.get("running_var", np.zeros(D, dtype=x.dtype))

    scores, cache = None, None

    if mode == "train":
        N = x.shape[0]
        sample_mean = np.sum(x, axis=0) / N
        sample_variance = np.sum((x - sample_mean) ** 2, axis=0) / N
        sample_stddev = np.sqrt(sample_variance + eps)

        x_hat = (x - sample_mean) / sample_stddev

        out = gamma * x_hat + beta

        running_mean = momentum * running_mean + (1 - momentum) * sample_mean
        running_var = momentum * running_var + (1 - momentum) * sample_variance

        cache = (x, gamma, beta, eps, sample_mean, sample_variance, x_hat)

    elif mode == "test":
        running_std = np.sqrt(running_var + eps)
        out = gamma * (x-running_mean)/running_std + beta
    else:
        raise ValueError('Invalid forward batchnorm mode "%s"' % mode)

    # Store the updated running means back into bn_param
    bn_param["running_mean"] = running_mean
    bn_param["running_var"] = running_var

    return out, cache


def batchnorm_backward(dout, cache):
    """
    Backward pass for batch normalization.

    For this implementation, you should write out a computation graph for
    batch normalization on paper and propagate gradients backward through
    intermediate nodes.

    Inputs:
    - dout: Upstream derivatives, of shape (N, D)
    - cache: Variable of intermediates from batchnorm_forward.

    Returns a tuple of:
    - dx: Gradient with respect to inputs x, of shape (N, D)
    - dgamma: Gradient with respect to scale parameter gamma, of shape (D,)
    - dbeta: Gradient with respect to shift parameter beta, of shape (D,)
    """
    x, gamma, beta, eps, sample_mean, sample_variance, x_hat = cache

    dgamma = np.sum(dout * x_hat, axis=0)
    dbeta = np.sum(dout, axis=0)

    N, D = x.shape

    inv_var = 1. / np.sqrt(sample_variance + eps)

    dxhat = dout * gamma
    dsigma = np.sum(-0.5 * dxhat * (x - sample_mean) * np.power(inv_var, 3), axis=0)
    dmu = np.sum(-dxhat * inv_var, axis=0) + dsigma * np.mean(-2 * (x - sample_mean), axis=0)

    dx = dxhat * inv_var + dsigma * 2 * (x - sample_mean) / N + dmu / N

    return dx, dgamma, dbeta


def affine_bn_relu_forward(x, w, b, gamma=None, beta=None, bn_param=None):
    """
    Convenience layer that perorms an affine transform followed by a ReLU

    Inputs:
    - x: Input to the affine layer
    - w, b: Weights for the affine layer

    Returns a tuple of:
    - out: Output from the ReLU
    - cache: Object to give to the backward pass
    """
    # print("activation mean: {} and std: {}".format(np.mean(x), np.std(x)))

    a, fc_cache = affine_forward(x, w, b)
    # print(bn_param)

    if bn_param is None:
        out, relu_cache = relu_forward(a)
        cache = (fc_cache, relu_cache, None)
    else:
        a_hat, bn_cache = batchnorm_forward(a, gamma, beta, bn_param)
        out, relu_cache = relu_forward(a_hat)
        cache = (fc_cache, relu_cache, bn_cache)

    return out, cache


def affine_bn_relu_backward(dout, cache):
    fc_cache, relu_cache, bn_cache = cache

    if bn_cache is None:
        da = relu_backward(dout, relu_cache)
        dx, dw, db = affine_backward(da, fc_cache)
        return dx, dw, db
    else:
        da = relu_backward(dout, relu_cache)
        dbn, dgamma, dbeta = batchnorm_backward(da, bn_cache)
        dx, dw, db = affine_backward(dbn, fc_cache)
        return dx, dw, db, dgamma, dbeta


if __name__ == "__main__":
    # Test the affine_forward function
    num_inputs = 2
    input_shape = (4, 5, 6)
    output_dim = 3

    input_size = num_inputs * np.prod(input_shape)
    weight_size = output_dim * np.prod(input_shape)

    x = np.linspace(-0.1, 0.5, num=input_size).reshape(num_inputs, *input_shape)
    w = np.linspace(-0.2, 0.3, num=weight_size).reshape(np.prod(input_shape), output_dim)
    b = np.linspace(-0.3, 0.1, num=output_dim)

    out, _ = affine_forward(x, w, b)
    correct_out = np.array([[1.49834967, 1.70660132, 1.91485297],
                            [3.25553199, 3.5141327, 3.77273342]])

    # Compare your output with ours. The error should be around e-9 or less.
    print('Testing affine_forward function:')
    print('difference: ', rel_error(out, correct_out))

    # Test the affine_backward function
    np.random.seed(231)
    x = np.random.randn(10, 2, 3)
    w = np.random.randn(6, 5)
    b = np.random.randn(5)
    dout = np.random.randn(10, 5)

    dx_num = eval_numerical_gradient_array(lambda x: affine_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_forward(x, w, b)[0], b, dout)

    _, cache = affine_forward(x, w, b)
    dx, dw, db = affine_backward(dout, cache)

    # The error should be around e-10 or less
    print('Testing affine_backward function:')
    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))

    # Test the relu_forward function
    x = np.linspace(-0.5, 0.5, num=12).reshape(3, 4)

    out, _ = relu_forward(x)
    correct_out = np.array([[0., 0., 0., 0., ],
                            [0., 0., 0.04545455, 0.13636364, ],
                            [0.22727273, 0.31818182, 0.40909091, 0.5, ]])

    # Compare your output with ours. The error should be on the order of e-8
    print('Testing relu_forward function:')
    print('difference: ', rel_error(out, correct_out))

    # Test the relu_backward function
    np.random.seed(231)
    x = np.random.randn(10, 10)
    dout = np.random.randn(*x.shape)

    dx_num = eval_numerical_gradient_array(lambda x: relu_forward(x)[0], x, dout)

    _, cache = relu_forward(x)
    dx = relu_backward(dout, cache)

    # Testing affine_relu_forward and affine_relu_backward:
    # The error should be on the order of e-12
    print('Testing relu_backward function:')
    print('dx error: ', rel_error(dx_num, dx))

    np.random.seed(231)
    x = np.random.randn(2, 3, 4)
    w = np.random.randn(12, 10)
    b = np.random.randn(10)
    dout = np.random.randn(2, 10)

    out, cache = affine_relu_forward(x, w, b)
    dx, dw, db = affine_relu_backward(dout, cache)

    dx_num = eval_numerical_gradient_array(lambda x: affine_relu_forward(x, w, b)[0], x, dout)
    dw_num = eval_numerical_gradient_array(lambda w: affine_relu_forward(x, w, b)[0], w, dout)
    db_num = eval_numerical_gradient_array(lambda b: affine_relu_forward(x, w, b)[0], b, dout)

    # Relative error should be around e-10 or less
    print('Testing affine_relu_forward and affine_relu_backward:')
    print('dx error: ', rel_error(dx_num, dx))
    print('dw error: ', rel_error(dw_num, dw))
    print('db error: ', rel_error(db_num, db))

    # Check the training-time forward pass by checking means and variances
    # of features both before and after batch normalization

    # Simulate the forward pass for a two-layer network
    np.random.seed(231)
    N, D1, D2, D3 = 200, 50, 60, 3
    X = np.random.randn(N, D1)
    W1 = np.random.randn(D1, D2)
    W2 = np.random.randn(D2, D3)
    a = np.maximum(0, X.dot(W1)).dot(W2)

    print('Before batch normalization:')
    print_mean_std(a, axis=0)

    gamma = np.ones((D3,))
    beta = np.zeros((D3,))
    # Means should be close to zero and stds close to one
    print('After batch normalization (gamma=1, beta=0)')
    a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
    print_mean_std(a_norm, axis=0)

    gamma = np.asarray([1.0, 2.0, 3.0])
    beta = np.asarray([11.0, 12.0, 13.0])
    # Now means should be close to beta and stds close to gamma
    print('After batch normalization (gamma=', gamma, ', beta=', beta, ')')
    a_norm, _ = batchnorm_forward(a, gamma, beta, {'mode': 'train'})
    print_mean_std(a_norm, axis=0)