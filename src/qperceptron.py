import numpy as np
import scipy.linalg as la
from scipy.misc import logsumexp
import time

class QPerceptron():
    # Pauli matrices
    S_X = np.array([[0, 1], [1, 0]], dtype=complex)
    S_Y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]], dtype=complex)
    S_Z = np.array([[1, 0], [0, -1]], dtype=complex)
    S = np.array([S_X, S_Y, S_Z])

    def __init__(self, D, y, bias=False, manual_lookup=False):
        # Set class variables
        self.D = D
        self.bias = bias
        self.y = y
        self.n_samples = D.shape[0]
        # add bias to data
        if bias:
            self.add_bias()
        self.dim = self.D.shape[1]
        # gather statistics bx and qx
        if not manual_lookup:
            self._create_statistics_lookup_table()

    def _create_statistics_lookup_table(self):
        # Initialize dicts
        self.bx_lookup = np.zeros(self.n_samples)
        self.qx_lookup = np.zeros(self.n_samples)
        # gather statistics for each sample, store these based on index
        for i in range(self.n_samples):
            self.bx_lookup[i] = self._bx(self.D, self.D[i,:], self.y)
            self.qx_lookup[i] = self._qx(self.D, self.D[i,:])


    def train(self, max_iter, eta, calculate_loss=False, tol=10e-8, verbose=True):

        # initialize weights

        _w = np.random.uniform(low=-1, high=1, size=(self.dim, 3)).astype(np.float)
        _loss = []
        _lh = []
        new_w = np.zeros_like(_w)

        _lh.append(self.likelihood(_w))
        self.h_storage = []
        samples = [5*x for x in range(5)]

        for i in range(max_iter):

            # calculate fields for all 3 inputs
            h = np.dot(_w.T, self.D.T)
            # print("elapsed time dot {}".format(time.time() - start))
            self.h_storage.append(h[:,samples])
            # calculate norm
            h_x = np.sqrt(np.sum(np.square(h), axis=0))
            # calculate gradients, vectorized completely
            _delta_x = self.qx_lookup * (np.sqrt(1 - np.square(self.bx_lookup))
                                         - np.tanh(h_x) * (h[0, :] / h_x))
            _delta_y = -1 * self.qx_lookup * np.tanh(h_x) * (h[1,:] / h_x)
            _delta_z = self.qx_lookup * (self.bx_lookup - np.tanh(h_x) * h[2,:] / h_x)
            # print("elapsed time delta {}".format(time.time() - start))
            # add sum of gradients to weights, multiplying by the learning rate
            new_w[:, 0] = np.einsum(_delta_x, [0,], self.D, [0,1], [1,])
            new_w[:, 1] = np.einsum(_delta_y, [0,], self.D, [0,1], [1,])
            new_w[:, 2] = np.einsum(_delta_z, [0,], self.D, [0,1], [1,])
            # print("elapsed time w prop {}".format(time.time() - start))
            # momentum
            _w += eta * new_w
            # calculate the likelhood and add it to the list.
            _lh.append(self.likelihood(_w))
            # calculate the loss if nescessary
            if calculate_loss:
                self.w = _w
                _loss.append(self.get_loss())

            # if the likelihood has converged, we break.
            if abs(_lh[i] - _lh[i - 1]) < tol:
                if verbose:
                    print("Convergence reached after {} steps".format(i))
                self.w = _w
                self.lh = _lh
                self.loss = _loss
                return
            # print("elapsed time {}".format(time.time() - start))

        # break if we have reached the maximum number of iterations
        print("No convergence after {} steps!".format(max_iter))
        self.w = _w
        self.lh = _lh
        self.loss = _loss
        return


    def get_loss(self):
        # calculate the classification loss
        loss = 0
        for k in range(self.n_samples):
            # calculate the hamiltonian
            H = self._H_x(self.D[k, :])
            # the the eigenvectors
            lam, v_x = la.eig(H)
            # take exponent of eigenvalues, so we have the density matrix eigenvalues
            # lam = np.exp(lam)
            # find the largerst eigenvalue
            # id = np.argmax(np.absolute(lam))
            id = np.argmax(lam)
            # get the corresponding eigenvector
            v = v_x[:, id]
            # calculate class probabilities
            p_one = np.absolute(v[0]) ** 2
            p_minus_one = np.absolute(v[1]) ** 2
            # threshold the probabilities and see if classes match.
            loss += 0.5 * abs(self.y[k] - np.sign(p_one - p_minus_one))

        return loss / self.n_samples


    def predict(self, samples, ev=True):
        def get_evalue(sample):
            H = self._H_x(sample)
            lam, v_x = la.eig(H)
            lam = np.exp(lam)
            id = np.argmax(np.absolute(lam))
            v = v_x[:, id]
            p_one = np.absolute(v[0]) ** 2
            p_minus_one = np.absolute(v[1]) ** 2
            return p_one, p_minus_one

        # add bias if our training was done with bias
        if self.bias:
            samples = np.hstack([samples, np.ones(samples.shape[0]).reshape(-1, 1)])
        # works similarly as calculate loss, but now returns the expectation value
        p = np.apply_along_axis(get_evalue, axis=1, arr=samples)
        if ev:
            return p[:,0] - p[:,1]
        return p[:,0], p[:,1]


    @staticmethod
    def _bx(X, sample, y):
        # find copies of our sample
        _idx = np.where((X == tuple(sample)).all(axis=1))[0]
        # return expected y for our samples, divided by number of copies
        return np.sum(y[_idx]) / len(_idx)


    @staticmethod
    def _qx(X, sample):
        # find copies of our sample
        _idx = np.where((X == tuple(sample)).all(axis=1))[0]
        # calcualte emperical probability of sample
        return len(_idx) / X.shape[0]


    def _H_x(self, _x,):
        # calculate parameterised hamiltonian, in pauli basis.
        _h = np.dot(self.w.T, _x)
        _H = _h[0] * QPerceptron.S[0]
        _H += _h[1] * QPerceptron.S[1]
        _H += _h[2] * QPerceptron.S[2]

        return _H


    def likelihood(self, _w):
        # calculate generalised quantum likelihood
        h = np.dot(_w.T, self.D.T)
        h_x = np.sqrt(np.sum(np.square(h), axis=0))
        # # completely vectorized.
        # L = np.sum(_qx_loopup_vec * (h[0,:] * np.sqrt(1 - np.square(_bx_loopup_vec)) +
        #                             h[2,:] * _bx_loopup_vec - np.log(2 * np.cosh(h_x))))

        # completely vectorized. protected against overflowing
        L = np.sum(self.qx_lookup * (h[0, :] * np.sqrt(1 - np.square(self.bx_lookup)) +
                                     h[2, :] * self.bx_lookup -
                                     np.logaddexp(h_x, - h_x)))

        return L

    def add_bias(self):
        self.D = np.hstack([self.D, np.ones(self.n_samples).reshape(-1, 1)])
