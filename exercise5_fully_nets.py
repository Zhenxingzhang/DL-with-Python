import numpy as np

from functions.helper import get_CIFAR10_data, rel_error
from functions.layers import affine_forward

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

    # # Load the (preprocessed) CIFAR10 data.
    # # Load the raw CIFAR-10 data.
    # X_train, y_train, X_val, y_val, X_test, y_test, X_dev, y_dev = get_CIFAR10_data()
    # print('Train data shape: ', X_train.shape)
    # print('Train labels shape: ', y_train.shape)
    # print('Validation data shape: ', X_val.shape)
    # print('Validation labels shape: ', y_val.shape)
    # print('Test data shape: ', X_test.shape)
    # print('Test labels shape: ', y_test.shape)
