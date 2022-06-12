import numpy as np


def f1(X1, coef1, X2, coef2, seed1, seed2, seed3, shape1, shape2):
    # note: shape is of the forst (x1,x2)
    # return W1 x (X1 ** coef1) + W2 x (X2 ** coef2) +b
    # where W1 is random matrix of shape shape1 with seed1
    # where W2 is random matrix of shape shape2 with seed2
    # where B is a random matrix of comaptible shape with seed3
    # if dimension mismatch occur return -1
    ans = None
    m1 = X1 ** coef1
    print(X1, m1)
    m2 = X2 ** coef2
    # m1 = np.power(X1, coef1)
    # m2 = np.power(X2, coef2)
    # m1 = np.linalg.matrix_power(X1, coef1)
    # m2 = np.linalg.matrix_power(X2, coef2)
    np.random.seed(seed1)
    W1 = np.random.rand(*(shape1))
    np.random.seed(seed2)
    W2 = np.random.rand(*(shape2))

    if m1.shape[0] != shape1[-1] or m2.shape[0] != shape2[-1]:
        return -1

    m1 = np.matmul(W1, m1)
    m2 = np.matmul(W2, m2)

    if m1.shape != m2.shape:
        return -1

    np.random.seed(seed3)
    b = np.random.rand(*(m1.shape))
    ans = m1 + m2 + b

    return ans


print(f1(np.array([[1, 2], [3, 4]]), 3, np.array(
    [[1, 2], [3, 4]]), 2, 1, 2, 3, (3, 2), (3, 2)))
