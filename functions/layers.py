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