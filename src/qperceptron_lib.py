import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
import sys


def plot_boundary(X, y, model, method='EV', cl_or_qm='qm', title=0, colormap=None):
    if colormap is not None:
        cmap = colormap
    else:
        cmap = plt.cm.get_cmap('coolwarm')
    plt.rc('font', size=15)

    blue = cmap(0.0)
    red = cmap(1.0)
    h = .05
    max_grid = 5
    x_min, x_max = 0 - max_grid, 0 + max_grid
    y_min, y_max = 0 - max_grid, 0 + max_grid
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    if cl_or_qm == 'qm':
        if method == 'EV':
            Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        elif method == 'mz':
            if model.bias:
                X_bias = np.c_[xx.ravel(), yy.ravel(), np.ones_like(yy.ravel())]
            else:
                X_bias = np.c_[xx.ravel(), yy.ravel()]
            h = np.dot(model.w.T, X_bias.T)
            h_x = np.sqrt(np.sum(np.square(h), axis=0))
            Z = np.tanh(h_x) * h[2, :] / h_x
        else:
            sys.exit('Wrong method, choose "EV" or "mz"')
    elif cl_or_qm == 'cl':
        # if model.bias:
        #     X_bias = np.c_[xx.ravel(), yy.ravel(), np.ones_like(yy.ravel())]
        # else:
        #     X_bias = np.c_[xx.ravel(), yy.ravel()]
        X_bias = np.c_[xx.ravel(), yy.ravel()]

        Z = model.predict(X_bias)
    else:
        sys.exit('Wrong method, choose "cl" or "qm"')

    Z = Z.reshape(xx.shape)
    fig, ax = plt.subplots()
    ax.contour(xx, yy, Z, cmap=cmap)
    # Plot also the training points
    y = y.flatten()
    np.random.seed(123)
    spread = 0.3
    ax.scatter(X[(y == -1), 0] + np.random.uniform(-spread, spread, np.sum((y == -1))),
               X[(y == -1), 1] + np.random.uniform(-spread, spread, np.sum((y == -1))),
               color=blue, label='-1', s=5)
    ax.scatter(X[(y == 1), 0] + np.random.uniform(-spread, spread, np.sum((y == 1))),
               X[(y == 1), 1] + np.random.uniform(-spread, spread, np.sum((y == 1))),
               color=red, label='+1', s=5)

    if title:
        ax.set_title(title)
    else:
        ax.set_title('Perceptron {}'.format(cl_or_qm))
    m = plt.cm.ScalarMappable(cmap=cmap)
    m.set_array(Z)

    plt.colorbar(m, boundaries=np.linspace(-1, 1, 11))
    ax.set_xlabel('$x_0$')
    ax.set_ylabel('$x_1$')
    ax.legend()


def _apply_func_array_of_matrices(arr, func, axis):
    """
    Apply function along axis for array of matrices.
    Not that if the matrix has dim=NxN, func should return a
    matrix with dim=NxN
    :param arr: 3 dimensional array
    :param func: function that works on matrix and returns matrix
    :param axis: axis where the matrices are stored
    :return: 3 dimensional array
    """
    dim_len = arr.shape[axis]
    slc = [slice(None)] * len(arr.shape)
    for i in range(dim_len):
        slc[axis] = i
        arr[slc] = func(arr[slc])
    return arr


def calculate_shannon_entropy(classical_perceptron, X):
    mz_cl = np.zeros(X.shape[0])

    for i in range(X.shape[0]):
        h = np.dot(classical_perceptron.w.T, X[i])
        h_x = np.sqrt(np.sum(np.square(h)))
        mz_cl[i] = np.tanh(h_x) * h / h_x

    Shannon = np.zeros_like(mz_cl)

    for i in range(mz_cl.shape[0]):
        rho = np.array([[1 + mz_cl[i], 0],
                        [0, 1 - mz_cl[i]]], dtype=np.complex)
        lam, v_x = la.eig(rho)
        lam /= np.sqrt(np.sum(np.square(lam)))
        Shannon[i] -= np.sum(lam * np.log2(lam))
    return Shannon, mz_cl


def calculate_von_neumann_entropy(quantum_perceptron, X):
    mx = np.zeros(X.shape[0])
    mz = np.zeros(X.shape[0])
    for i in range(X.shape[0]):
        h = np.dot(quantum_perceptron.w.T, X[i])
        h_x = np.sqrt(np.sum(np.square(h)))
        mx[i] = np.tanh(h_x) * h[0] / h_x
        mz[i] = np.tanh(h_x) * h[2] / h_x

    Neumann = np.zeros_like(mz)

    for i in range(mz.shape[0]):
        rho = np.array([[1 + mz[i], mx[i]],
                        [mx[i], 1 - mz[i]]], dtype=np.complex)
        lam, _ = la.eig(rho)
        lam /= np.sqrt(np.sum(np.square(lam)))
        Neumann[i] -= np.sum(lam * np.log2(lam))
    return Neumann, mx, mz


# manual lookup generation if all samples are unique
def manual_lookup(perceptron):
    perceptron.bx_lookup = {}
    perceptron.qx_lookup = {}

    for i in range(perceptron.n_samples):
        perceptron.bx_lookup[i] = perceptron.y[i]
        perceptron.qx_lookup[i] = 1.0 / perceptron.n_samples


def get_qp_EV_statistics(model, X, y, q, return_stats=False, verbose=True):
    p = model.predict(X, False)[0]
    mse = MSE(0.5 * (y.flatten() + 1), p)
    mae = MAE(0.5 * (y.flatten() + 1), p)
    likelihood = -np.sum(q * 0.5 * (y.flatten() + 1) * np.log2(p))
    if verbose:
        print('EV MSE = {}'.format(mse))
        print('EV MAE = {}'.format(mae))
        print('EV lh = {}'.format(likelihood))
    expectation_val = y.flatten() * p
    if return_stats:
        return mse, mae, likelihood, p, expectation_val
    return p, expectation_val


def get_qp_mz_statistics(model, X, y, q, return_stats=False, verbose=True):
    if model.bias:
        X = np.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
    h = np.dot(model.w.T, X.T)
    h_x = np.sqrt(np.sum(np.square(h), axis=0))

    p = 0.5 * (1 + np.tanh(h_x) * h[2, :] / h_x)
    mse = MSE(0.5 * (y.flatten() + 1), p)
    mae = MAE(0.5 * (y.flatten() + 1), p)
    # clip log to not get log(0)
    likelihood = -np.sum(q * 0.5 * (y.flatten() + 1) * np.log2(np.clip(p, 10e-16, np.inf)))
    if verbose:
        print('mz MSE = {}'.format(mse))
        print('mz MAE = {}'.format(mae))
        print('mz lh = {}'.format(likelihood))
    expectation_val = y.flatten() * p
    if return_stats:
        return mse, mae, likelihood, p, expectation_val
    return p, expectation_val


def get_cl_statistics(model, X, y, q, return_stats=False, verbose=True):
    if model.bias:
        X = np.hstack([X, np.ones(X.shape[0]).reshape(-1, 1)])
    h = np.dot(model.w.T, X.T)
    p = 0.5*(np.tanh(h) +1)
    mse = MSE(0.5 * (y.flatten() + 1), p)
    mae = MAE(0.5 * (y.flatten() + 1), p)
    # clip log to not get log(0)
    likelihood = -np.sum(q * 0.5 * (y.flatten() + 1) * np.log2(np.clip(p, 10e-16, np.inf)))
    if verbose:
        print('cl MSE = {}'.format(mse))
        print('cl MAE = {}'.format(mae))
        print('cl lh = {}'.format(likelihood))
    expectation_val = y.flatten() * p
    if return_stats:
        return mse, mae, likelihood, p, expectation_val

    return p, expectation_val


def MSE(p, q):
    # print(p)
    return np.mean(np.square(p - q))


def MAE(p, q):
    return np.mean(np.abs(p - q))


def plot_weights(w, k):
    x = np.array(w)[:, :, k]
    plt.plot(x[:, 0], label='x')
    plt.plot(x[:, 1], label='y')
    plt.plot(x[:, 2], label='z')
    plt.yscale('log')
    plt.legend()
