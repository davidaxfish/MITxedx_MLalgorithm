import numpy as np

### Functions for you to fill in ###



def polynomial_kernel(X, Y, c, p):
    """
        Compute the polynomial kernel between two matrices X and Y::
            K(x, y) = (<x, y> + c)^p
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            c - a coefficient to trade off high-order and low-order terms (scalar)
            p - the degree of the polynomial kernel

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    #kernel_matrix = []
    # for x in X:
    #     kernel_matrix_row = []
    #     for y in Y:
    #         kernel_matrix_row.append([x+c*np.ones(np.shape(x)), y+c*np.ones(np.shape(y))])
    #         # print('x', x, np.shape(x))
    #         # print('y', y, np.shape(y))
    #     kernel_matrix.append(kernel_matrix_row)
    # return np.power(np.array(kernel_matrix), p)
    return np.power(np.dot(X, np.transpose(Y)) + c * np.ones([np.shape(X)[0], np.shape(Y)[0]]) , p)
    # YOUR CODE HERE
    raise NotImplementedError



def rbf_kernel(X, Y, gamma):
    """
        Compute the Gaussian RBF kernel between two matrices X and Y::
            K(x, y) = exp(-gamma ||x-y||^2)
        for each pair of rows x in X and y in Y.

        Args:
            X - (n, d) NumPy array (n datapoints each with d features)
            Y - (m, d) NumPy array (m datapoints each with d features)
            gamma - the gamma parameter of gaussian function (scalar)

        Returns:
            kernel_matrix - (n, m) Numpy array containing the kernel matrix
    """
    # YOUR CODE HERE
    return np.array([[ np.exp(-gamma * np.square(np.linalg.norm(x-y))) for y in Y ] for x in X])
    raise NotImplementedError
