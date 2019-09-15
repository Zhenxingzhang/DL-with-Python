import numpy as np
from functions.helper import rel_error


def sgd(w, dw, config=None):
    """
    Performs vanilla stochastic gradient descent.

    config format:
    - learning_rate: Scalar learning rate.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)

    w -= config['learning_rate'] * dw
    return w, config


def sgd_momentum(w, dw, config=None):
    """
    Performs stochastic gradient descent with momentum.

    config format:
    - learning_rate: Scalar learning rate.
    - momentum: Scalar between 0 and 1 giving the momentum value.
      Setting momentum = 0 reduces to sgd.
    - velocity: A numpy array of the same shape as w and dw used to store a
      moving average of the gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('momentum', 0.9)
    v = config.get('velocity', np.zeros_like(w))

    # the next_w variable. You should also use and update the velocity v.     #

    v = config['momentum'] * v - config['learning_rate'] * dw

    next_w = w + v

    config['velocity'] = v

    return next_w, config


def rmsprop(w, dw, config=None):
    """
    Uses the RMSProp update rule, which uses a moving average of squared
    gradient values to set adaptive per-parameter learning rates.

    config format:
    - learning_rate: Scalar learning rate.
    - decay_rate: Scalar between 0 and 1 giving the decay rate for the squared
      gradient cache.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - cache: Moving average of second moments of gradients.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-2)
    config.setdefault('decay_rate', 0.99)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('cache', np.zeros_like(w))

    next_w = None

    # in the next_w variable. Don't forget to update cache value stored in    #
    # config['cache'].                                                        #

    cache = config['decay_rate'] * config['cache'] + (1 - config['decay_rate']) * dw ** 2

    next_w = w - config['learning_rate'] * dw / np.sqrt(cache + 1e-07)

    config['cache'] = cache

    return next_w, config


def adam(w, dw, config=None):
    """
    Uses the Adam update rule, which incorporates moving averages of both the
    gradient and its square and a bias correction term.

    config format:
    - learning_rate: Scalar learning rate.
    - beta1: Decay rate for moving average of first moment of gradient.
    - beta2: Decay rate for moving average of second moment of gradient.
    - epsilon: Small scalar used for smoothing to avoid dividing by zero.
    - m: Moving average of gradient.
    - v: Moving average of squared gradient.
    - t: Iteration number.
    """
    if config is None: config = {}
    config.setdefault('learning_rate', 1e-3)
    config.setdefault('beta1', 0.9)
    config.setdefault('beta2', 0.999)
    config.setdefault('epsilon', 1e-8)
    config.setdefault('m', np.zeros_like(w))
    config.setdefault('v', np.zeros_like(w))
    config.setdefault('t', 0)

    next_w = None

    # the next_w variable. Don't forget to update the m, v, and t variables   #
    # stored in config.                                                       #
    #                                                                         #
    # NOTE: In order to match the reference output, please modify t _before_  #
    # using it in any calculations.                                           #

    t = config['t'] + 1
    m = config['m']
    v = config['v']
    beta1 = config['beta1']
    beta2 = config['beta2']

    # Update biased moments estimate
    m = beta1 * m + (1 - beta1) * dw
    v = beta2 * v + (1 - beta2) * (dw ** 2)

    # (Compute bias-corrected moments estimate)
    m_bias = m / (1 - beta1 ** t)
    v_bias = v / (1 - beta2 ** t)

    # Apply Adam update
    next_w = w - config['learning_rate'] * m_bias / (np.sqrt(v_bias) + config['epsilon'])

    # update config
    config['v'] = v
    config['t'] = t
    config['m'] = m

    return next_w, config


if __name__ == "__main__":
    N, D = 4, 5
    w = np.linspace(-0.4, 0.6, num=N * D).reshape(N, D)
    dw = np.linspace(-0.6, 0.4, num=N * D).reshape(N, D)
    v = np.linspace(0.6, 0.9, num=N * D).reshape(N, D)

    config = {'learning_rate': 1e-3, 'velocity': v}
    next_w, _ = sgd_momentum(w, dw, config=config)

    expected_next_w = np.asarray([
        [0.1406, 0.20738947, 0.27417895, 0.34096842, 0.40775789],
        [0.47454737, 0.54133684, 0.60812632, 0.67491579, 0.74170526],
        [0.80849474, 0.87528421, 0.94207368, 1.00886316, 1.07565263],
        [1.14244211, 1.20923158, 1.27602105, 1.34281053, 1.4096]])
    expected_velocity = np.asarray([
        [0.5406, 0.55475789, 0.56891579, 0.58307368, 0.59723158],
        [0.61138947, 0.62554737, 0.63970526, 0.65386316, 0.66802105],
        [0.68217895, 0.69633684, 0.71049474, 0.72465263, 0.73881053],
        [0.75296842, 0.76712632, 0.78128421, 0.79544211, 0.8096]])

    # Should see relative errors around e-8 or less
    print('next_w error: ', rel_error(next_w, expected_next_w))
    print('velocity error: ', rel_error(expected_velocity, config['velocity']))
