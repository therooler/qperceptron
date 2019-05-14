import numpy as np
import scipy.sparse as sp
import time
import scipy.linalg as la


class CLPerceptron():
    S_X = np.array([[0, 1], [1, 0]], dtype=complex)
    S_Y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]], dtype=complex)
    S_Z = np.array([[1, 0], [0, -1]], dtype=complex)
    S = np.array([S_X, S_Y, S_Z])

    def __init__(self, D, y, bias=False, manual_lookup=False):
        self.D = D
        self.bias = bias
        self.y = y
        self.n_samples = D.shape[0]
        if bias:
            self.add_bias()
        self.dim = self.D.shape[1]
        if not manual_lookup:
            self._create_statistics_lookup_table()

    def _create_statistics_lookup_table(self):

        self.bx_lookup = np.zeros(self.n_samples)
        self.qx_lookup = np.zeros(self.n_samples)
        # gather statistics for each sample, store these based on index
        for i in range(self.n_samples):
            self.bx_lookup[i] = self._bx(self.D, self.D[i, :], self.y)
            self.qx_lookup[i] = self._qx(self.D, self.D[i, :])

    def train(self, max_iter, eta, calculate_loss=False, tol=10e-8, verbose=True):

        _w = np.random.uniform(low=-1, high=1, size=self.dim)
        _loss = []
        _lh = []

        _lh.append(self.likelihood(_w))

        for i in range(max_iter):

            h = np.dot(self.D, _w)

            h_x = np.sqrt(np.square(h))

            _delta_z = self.qx_lookup * (self.bx_lookup - np.tanh(h_x) * (h / h_x))

            _w += eta * np.einsum(_delta_z, [0, ], self.D, [0, 1], [1, ])  # reg - 10e-10 * np.sum(_w)
            _lh.append(self.likelihood(_w))

            if abs(_lh[i] - _lh[i - 1]) < tol:
                if verbose:
                    print("Convergence reached after {} steps".format(i))
                self.w = _w
                self.lh = _lh
                self.loss = _loss

                return
        if verbose:
            print("No convergence after {} steps!".format(max_iter))
        self.w = _w
        self.lh = _lh
        self.loss = _loss
        return

    def predict(self, samples, ev=True):
        def get_evalue(sample):
            h = np.dot(self.w.T, sample)
            p_one = 0.5 * (np.tanh(h) + 1)

            return p_one, 1-p_one

        # add bias if our training was done with bias
        if self.bias:
            samples = np.hstack([samples, np.ones(samples.shape[0]).reshape(-1, 1)])
        # works similarly as calculate loss, but now returns the expectation value
        p = np.apply_along_axis(get_evalue, axis=1, arr=samples)
        if ev:
            return p[:,0] - p[:,1]
        return p[:,0], p[:,1]

    def get_loss(self):
        y_pred = self.predict(self.D)
        loss = 0.5 * np.sum(np.absolute(y_pred - self.y))
        return loss / self.n_samples

    # def predict(self, _samples):
    #     return np.sign(np.dot(self.w, _samples.T))

    def predict_sigm(self, _samples):
        return self._sigmoid(np.dot(self.w, _samples.T))

    def _H_x(self, _x,):
        # calculate parameterised hamiltonian, in pauli basis.
        _h = np.dot(self.w.T, _x)
        _H = _h * CLQPerceptron.S[2]

        return _H
    @staticmethod
    def _bx(X, sample, y):
        _idx = np.where((X == tuple(sample)).all(axis=1))[0]
        return np.sum(y[_idx]) / len(_idx)

    @staticmethod
    def _qx(X, sample):
        _idx = np.where((X == tuple(sample)).all(axis=1))[0]
        return len(_idx) / X.shape[0]

    def likelihood(self, _w):
        h = np.dot(_w.T, self.D.T)
        h_x = np.sqrt(np.square(h))
        L = np.sum(self.qx_lookup * (h * self.bx_lookup - np.logaddexp(h_x, -h_x)))
        return L

    def _delta_w(self, idx):
        h = np.dot(self.w.T, self.D[idx, :])
        return self.qx_lookup[idx] * (self.bx_lookup[idx] - np.tanh(h))

    @staticmethod
    def _sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def add_bias(self):
        self.D = np.hstack([self.D, np.ones(self.n_samples).reshape(-1, 1)])
