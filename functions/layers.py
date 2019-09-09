import numpy as np


def ReLU(inputs, vectorise = True):

    # The ReLU unit simply passes its argument forward if positive,
    # but replaces it with zero if negative.

    if vectorise:

        # The maximum function selects the bigger of two inputs, element-wise:
        return np.maximum(inputs, 0)

    else:

        # Find shapes:
        N = inputs.shape[0]
        J = inputs.shape[1]

        # Set up a clean template for the outputs
        Z = np.zeros((N, J), dtype = np.float32)

        # Loop-and-select
        for i in range(N):
            for j in range(J):

                Z[i, j] = max(inputs[i, j], 0)

        return Z


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

