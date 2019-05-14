import numpy as np
import tensorflow as tf
import cmath as cm
import matplotlib.pyplot as plt
from tensorflow.python.client import timeline
import os

tf.logging.set_verbosity(tf.logging.ERROR)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


class Field():
    def __init__(self, n_spins, h, learnable=True):
        assert isinstance(h, (dict, list, np.ndarray)), 'Pass field as dict, list or ndarray'
        self.h = h
        if learnable and isinstance(h, list):
            self.h = {}
            for i in h:
                self.h[i] = np.random.rand(1)
        elif learnable and isinstance(h, np.ndarray):
            self.h = {}
            assert len(h) == max(h.shape), 'Expected 1 dimensional array, received array with shape {}'.format(h.shape)
            for i in h:
                self.h[i] = np.random.rand(1)
        assert all([(isinstance(x, int)) & ((x >= 0) & (x <= n_spins - 1)) for x in
                    self.h.keys()]), 'Keys must be non-negative ints < {}'.format(
            n_spins - 1)

        self.n_spins = n_spins
        self.learnable = learnable


class Coupling():
    def __init__(self, n_spins, j, learnable=True):
        assert isinstance(j, (dict, list)), 'Pass coupling as dict or list of tuples'
        self.j = j
        if learnable and isinstance(j, list):
            assert all([(isinstance(x, tuple)) & (len(x) == 2) for x in j]), "Expected list of tuples of length 2"
            self.j = {}
            for tup in j:
                self.j[tup] = np.random.rand(1)

        assert all([(isinstance(x, tuple)) & (len(x) == 2) for x in j]), "Keys to be a list of tuples of length 2"
        assert all(all((i >= 0) & (i <= n_spins - 1) for i in x) for x in
                   j.keys()), 'Keys must be non-negative ints < {}'.format(
            n_spins - 1)

        self.n_spins = n_spins
        self.learnable = learnable


class AbstractHamiltonian():
    S_X = np.array([[0, 1], [1, 0]], dtype=complex)
    S_Y = np.array([[0, complex(0, -1)], [complex(0, 1), 0]], dtype=complex)
    S_Z = np.array([[1, 0], [0, -1]], dtype=complex)
    PAULI_MATRICES = {'x': S_X, 'y': S_Y, 'z': S_Z}
    UNIT_MATRIX = np.array([[1, 0], [0, 1]], dtype=complex)

    def __init__(self, n_spins, field, coupling):
        assert all(k in ['hx', 'hz'] for k in field.keys()), 'Dict required with keys "hx" or "hz"'
        assert all(k in ['jxx', 'jzz'] for k in coupling.keys()), 'Dict required with keys "jxx" or "jzz"'
        assert all(isinstance(k, Field) for k in field.values()), 'Use the Field class to describe the fields'
        assert all(
            isinstance(k, Coupling) for k in coupling.values()), 'Use the Coupling class to describe the couplings'
        self.field = field
        self.coupling = coupling
        self.n_spins = n_spins

    @staticmethod
    def _kronPower(n, arr):
        # Kronecker exponent
        if n == 0:
            return 1
        else:
            return (np.kron(arr, AbstractHamiltonian._kronPower((n - 1), arr)))

    def _tensorchain(self, index, pauli_matrix):
        # let (x) denote the kronecker tensor product and N the number of spins:
        # S_xi = from j=1 to index-1[((x)_j UNIT_MATRIX)]
        #        (x) pauli_matrix (x)
        #        from l=1 to N[((x)_l UNIT_MATRIX)]
        # See also equation (9)-(10) in Kadowaki(2008)

        return np.kron(
            np.kron(self._kronPower(index - 1, AbstractHamiltonian.UNIT_MATRIX), pauli_matrix),
            self._kronPower(self.n_spins - index, AbstractHamiltonian.UNIT_MATRIX)
        )

    def get_field_hamiltonian(self, location, value, name):
        # interactions = [[[1,1],J_11],[[1,2],J_12],...[n,n],J_nn]
        # where n = number_of_spins

        return value * self._tensorchain(location + 1, AbstractHamiltonian.PAULI_MATRICES[name[1]])

    def get_coupling_hamiltonian(self, location, value, name):
        # field for x,y,z direction

        return value * np.dot(self._tensorchain(location[0] + 1, AbstractHamiltonian.PAULI_MATRICES[name[1]]),
                              self._tensorchain(location[1] + 1, AbstractHamiltonian.PAULI_MATRICES[name[2]]))

    def field_hamiltonian_iterator(self):
        for name, f in self.field.items():
            for loc, val in f.h.items():
                yield f.learnable, name + '_' + str(loc), self.get_field_hamiltonian(loc, val, name)

    def coupling_hamiltonian_iterator(self):
        for name, f in self.coupling.items():
            for loc, val in f.j.items():
                yield f.learnable, name + '_' + str(loc[0]) + str(loc[1]), self.get_coupling_hamiltonian(loc, val, name)

    def construct_hamiltonian(self):
        H = np.zeros((2 ** self.n_spins, 2 ** self.n_spins), dtype=np.complex128)
        for _, _, h in self.field_hamiltonian_iterator():
            H += h
        for _, _, h in self.coupling_hamiltonian_iterator():
            H += h
        return H


class QuantumLearning:
    ALPHABET = ['a', 'b', 'c', 'd', 'e', 'f', 'g',
                'h', 'i', 'j', 'k', 'l', 'm', 'n',
                'o', 'p', 'q', 'r', 's', 't', 'u',
                'v', 'w', 'x', 'y', 'z']

    def __init__(self, bias=True, n_classes=2, d_or_c='discr', **params):
        '''
        Model for quantum learning.

        :param bias:
        :param n_classes:
        :param d_or_c:
        :param params:
        '''

        assert n_classes >= 2, 'n_classes is {} but must be >= 2'.format(params['n_classes'])
        self.c = n_classes
        assert d_or_c in ['discr', 'cont'], '\'{}\' not in \"[\'discr\', \'cont\']\"'.format(d_or_c)
        self.d_or_c = d_or_c
        self.bias = bias
        self.FLAG_STATISTICS = False
        self.fetch_names = None
        self.fetchable_vars = []
        self.learn_hamiltonian = None

        if 'dim' in params.keys():
            self.dim = params['dim']
        else:
            self.dim = None
        if 'rank_one_approx' in params.keys():
            self.rank_one_approx = params['rank_one_approx']
        else:
            self.rank_one_approx = False

        if 'n_samples' in params.keys():
            self.n_samples = params['n_samples']
        else:
            self.n_samples = None
        if 'learn_hamiltonian' in params.keys():
            self.learn_hamiltonian = True
            self.hamiltonian = params['learn_hamiltonian']
            assert (isinstance(self.hamiltonian, AbstractHamiltonian)), 'Use the abstract Hamiltonian class'

    def get_statistics(self, X, y, **params):

        if self.bias:
            X = self.add_bias(X)
        if self.n_samples == None:
            self.n_samples = X.shape[0]
        if self.dim == None:
            self.dim = X.shape[1]

        if self.d_or_c == 'discr':
            self._get_discr_statistics(X, y)
        elif self.d_or_c == 'cont':
            self._get_cont_statistics(X, y, **params)
        self.Q = np.nan_to_num(self.Q)

    def _get_cont_statistics(self, X, y, sigma=1):
        classes = np.unique(y)
        assert len(
            classes) == self.c, 'number of classes found is {}, but the model is specified for {} classes'.format(
            len(classes), self.c)

        def _q_x_y(c, sample):
            _idy = (y == c)
            return np.sum(1 / np.sqrt(np.power(2 * cm.pi * sigma, self.dim)) * \
                          np.exp(
                              -0.5 * np.einsum('ij,ij->i', (X[_idy, :] - sample),
                                               (X[_idy, :] - sample)) / sigma))

        # Initialize arrays
        self.q_x_y = np.zeros((self.n_samples, self.c))
        self.q_y = np.zeros(self.c)
        self.q_x = np.zeros(self.n_samples)
        # Get statistics per class
        for j, c in enumerate(classes):
            for i in range(self.n_samples):
                self.q_x_y[i, j] = _q_x_y(c, X[i, :])
            self.q_y[j] = np.sum((y == c)) / self.n_samples
        # Normalize
        self.q_x_y /= np.sum(self.q_x_y, axis=1, keepdims=True)
        self.q_x = np.sum(self.q_x_y * self.q_y, axis=1)
        self.q_y = self.q_y.reshape(-1, 1)
        self.Q = np.einsum('ni,nj->nij', np.sqrt(self.q_x_y), np.sqrt(self.q_x_y))
        self.Q *= np.outer(np.sqrt(self.q_y), np.sqrt(self.q_y)).reshape(1, self.c, self.c)
        # Flag
        self.FLAG_STATISTICS = True

    def _get_discr_statistics(self, X, y):
        classes = np.unique(y)
        assert len(
            classes) == self.c, 'number of classes found is {}, but the model is specified for {} classes'.format(
            len(classes), self.c)

        def _q_y_x(c, sample):
            # find copies of our sample
            _idx = np.where((X == tuple(sample)).all(axis=1))[0]
            # return probablity q(y=1|x)
            return np.sum(y[_idx] == c) / len(_idx)

        def _q_x(sample):
            # find copies of our sample
            _idx = np.where((X == tuple(sample)).all(axis=1))[0]
            # calcualte emperical probability of sample
            return len(_idx) / X.shape[0]

        # Initialize arrays
        self.q_y_x = np.zeros((self.n_samples, self.c))
        self.q_y = np.zeros(self.c)
        self.q_x = np.zeros((self.n_samples, 1))
        # Get statistics per class
        for i in range(self.n_samples):
            for j, c in enumerate(classes):
                self.q_y_x[i, j] = _q_y_x(c, X[i, :])
            self.q_x[i] = _q_x(X[i, :])
        for j, c in enumerate(classes):
            self.q_y[j] = np.sum((y == c)) / self.n_samples
        # Normalize
        self.Q = np.einsum('ni,nj->nij', np.sqrt(self.q_y_x), np.sqrt(self.q_y_x))
        self.Q *= self.q_x.reshape(-1, 1, 1)
        # Flag
        self.FLAG_STATISTICS = True

    def build_graph(self, hilbert_space_composition, epsilon=0.1, complex_weights=True, device='CPU'):
        # Assert first that the model is compiled properly
        assert self.dim != None, 'Data dimension unknown, specify as model argument or run QubitLearning.get_statistics'
        assert self.n_samples != None, 'Number of samples unknown, specify as model argument or run QubitLearning.get_statistics'
        assert isinstance(hilbert_space_composition,
                          (list, tuple, np.ndarray)), 'Pass list, tuple or ndarray of Hilbert space dimensions'

        assert all(isinstance(d, int) for d in
                   hilbert_space_composition), 'Expected list of integers for the dimensions of the composite Hilbert spaces'
        assert all(d >= 2 for d in hilbert_space_composition), 'Composite Hilbert spaces must have dim(H_i) >= 2'
        assert (hilbert_space_composition[
                    0] == self.c), 'dim(H_A) must be equal to the number of classes = {}, but dim(H_A) = {}'.format(
            self.c, hilbert_space_composition[0])

        # Build the model
        with tf.name_scope('model'):
            # TODO: X now has shape 1 but needs shape -1, rest of functions needs to be adapted
            dims = np.array([d for d in hilbert_space_composition])
            # self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.dim))
            # The data density matrix is C x C where C is the number of classes
            self.eta = tf.placeholder(dtype=tf.complex128, shape=(None, self.c, self.c))

            if len(hilbert_space_composition) == 1:
                self.MODEL_TYPE = 'single'
                generators = self.SU_generators(dims[0])
                self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.dim))
                # For single system we just take the SU(C) matrices with C^2 - 1 free parameters
                weights = tf.get_variable("weights", (self.dim, dims[0] ** 2 - 1), dtype=tf.float32)
                # Multiply the weights with the inputs to get a field phi for each vector parameter.
                phi = tf.matmul(self.x, weights)
                phi_norm = tf.sqrt(tf.reshape(tf.einsum('nj,nj->n', phi, tf.conj(phi)), (-1, 1)))
                phi = tf.tanh(phi_norm) * phi / phi_norm
                # Construct the model density matrix from the SU(C) generators.
                rho_red = 0.5 * (tf.eye(dims[0], dims[0], dtype=tf.complex128) + tf.einsum('ni,jki->njk',
                                                                                           tf.to_complex128(phi),
                                                                                           generators))
            else:
                if len(hilbert_space_composition) == 2:
                    self.MODEL_TYPE = 'bipartite'
                elif len(hilbert_space_composition) > 2:
                    self.MODEL_TYPE = 'multipartite'
                # In the bi- or multipartite case the number of parameters is equal to the size the combined Hilbert space
                if self.learn_hamiltonian is None:

                    # weights = tf.get_variable("weights", (self.dim, (np.product(dims, keepdims=True))), dtype=tf.float32)

                    if complex_weights:
                        self.x = tf.placeholder(dtype=tf.complex128, shape=(None, self.dim))
                        weights_r = tf.get_variable("weights_r", (self.dim, (np.product(dims, keepdims=True))),
                                                    dtype=tf.float32)
                        weights_c = tf.get_variable("weights_c", (self.dim, (np.product(dims, keepdims=True))),
                                                    dtype=tf.float32)
                        weights = tf.to_complex128(tf.complex(weights_r, weights_c))
                    else:
                        self.x = tf.placeholder(dtype=tf.float32, shape=(None, self.dim))
                        weights = tf.get_variable("weights", (self.dim, (np.product(dims, keepdims=True))),
                                                  dtype=tf.float32)
                    phi = tf.matmul(self.x, weights)
                    phi_norm = tf.sqrt(tf.reshape(tf.einsum('nj,nj->n', phi, tf.conj(phi)), (-1, 1)))
                    phi /= phi_norm
                    rho = tf.einsum('ni,nj->nij', phi, tf.conj(phi))
                    # Trace out the entangled systems to get a C x C density matrix
                    rho_red = self.partial_trace(rho, dims)
                else:
                    self.x = tf.placeholder(dtype=tf.complex128, shape=(None, self.dim))
                    I = tf.ones((tf.shape(self.x)[0], 1), dtype=tf.complex128)
                    H = tf.zeros(shape=(np.product(dims), np.product(dims)), dtype=tf.complex128)
                    num_weights = 0
                    for flag, name, f in self.hamiltonian.field_hamiltonian_iterator():
                        if flag:
                            weight = tf.get_variable("weight_{}".format(name), (self.dim, 1),
                                                     dtype=tf.float32)
                            h = tf.matmul(self.x, tf.to_complex128(weight),
                                          name="field_{}".format(name))
                            H += tf.einsum('ni,jk->njk', h, tf.constant(f))
                            num_weights += 1
                        else:
                            H += tf.einsum('ni,jk->njk', I, tf.constant(f))

                    for flag, name, f in self.hamiltonian.coupling_hamiltonian_iterator():
                        if flag:
                            weight = tf.get_variable("weight_{}".format(name), (self.dim, 1),
                                                     dtype=tf.float32)
                            h = tf.matmul(self.x, tf.to_complex128(weight),
                                          name="coupling_{}".format(name))
                            H += tf.einsum('ni,jk->njk', h, tf.constant(f))
                            num_weights += 1
                        else:
                            H += tf.einsum('ni,jk->njk', I, tf.constant(f))

                    energies, phi = tf.linalg.eigh(H)
                    energies = tf.identity(energies, name='energies')

                    if self.rank_one_approx:
                        rho = tf.einsum('ni,nj->nij', phi[:, 0], tf.conj(phi[:, 0]))
                    else:
                        rho = self.matrix_exp(H)
                        rho /= tf.reshape(tf.trace(rho), (-1, 1, 1))
                    rho_red = self.partial_trace(rho, dims)

                    with tf.name_scope('quantities'):
                        tf.identity(energies[:,0], name='gs_energy')
                        tf.identity(energies[:,0] - energies[:,1], name='width')
                        lams, _ =  tf.linalg.eigh(rho_red)
                        lams = tf.identity(lams, name='lams')
                        tf.reduce_sum(-lams * tf.log(lams), axis=1, name='vn_entropy')

        with tf.name_scope('negative_quantum_log_likelihood'):
            if self.c == 2:
                self.negqll = -tf.reduce_sum(tf.real(tf.trace(tf.matmul(self.eta, self.matrix_log_2x2(rho_red)))))
            else:
                self.negqll = -tf.reduce_sum(tf.real(tf.trace(tf.matmul(self.eta, self.matrix_log(rho_red)))))

            tf.summary.scalar('negative_quantum_log_likelihood', self.negqll)

        with tf.name_scope('optimizer'):
            # Get the optimizer
            opt = tf.train.GradientDescentOptimizer(learning_rate=epsilon)
            # opt = tf.train.AdagradOptimizer(learning_rate=epsilon)
            # self.train_step = opt.minimize(self.negqll, var_list=[weights])
            if self.MODEL_TYPE == 'bipartite' or self.MODEL_TYPE == 'multipartite':
                if self.learn_hamiltonian:
                    self.train_step = opt.minimize(self.negqll, var_list=[var for var in tf.trainable_variables()])
                else:
                    if complex_weights:
                        self.train_step = opt.minimize(self.negqll, var_list=[weights_r, weights_c])
                    else:
                        self.train_step = opt.minimize(self.negqll, var_list=[weights])
            else:
                self.train_step = opt.minimize(self.negqll, var_list=[weights])
            self._initialize_session(device)

        with tf.name_scope('predictor'):
            # self.probabilities = tf.placeholder(dtype=tf.float32, shape=(None, self.c))
            self.probabilities = tf.stack([rho_red[:, 0, 0], rho_red[:, 1, 1]], axis=1)
            # _, v = tf.linalg.eigh(rho)
            # self.probabilities = tf.reshape(v[:, :-1] * tf.conj(v[:, :-1]), (-1, self.c))
            # print(self.probabilities)

    def fetch_vars_train(self, fetch_names):
        assert isinstance(fetch_names, (list, tuple)), 'Expected list or tuple of variable names'
        assert self.learn_hamiltonian, 'not a learnable hamiltonian'
        tensorlist = [n.name for n in tf.get_default_graph().as_graph_def().node]
        assert all(w in tensorlist for w in fetch_names), '{} not found in list of trainable variables'.format(
            list(w for w in fetch_names if w not in tensorlist)
        )
        self.fetchable_vars = []
        for name in fetch_names:
            self.fetchable_vars.append(self._sess.graph.get_tensor_by_name(name + ':0'))
        self.fetch_names = fetch_names

    def _initialize_session(self, device):
        """
        Initialize session, variables, saver
        """
        assert device in ['CPU', 'GPU'], "device must be on of '['CPU', 'GPU'], received {}".format_map(device)


        if device == 'GPU':
            config = tf.ConfigProto()
        elif device == 'CPU':
            config = tf.ConfigProto(
                device_count={'GPU': 0}
            )

        config.gpu_options.allow_growth = True
        self._sess = tf.Session(config=config)
        self.merged = tf.summary.merge_all()
        self.file_writer = tf.summary.FileWriter('./tensorboard', self._sess.graph)
        self._sess.run(tf.global_variables_initializer())

    def predict(self, data):

        with tf.name_scope('predict'):
            if self.bias:
                data = self.add_bias(data)

            feed_dict = {self.x: data}
            probabilities = self._sess.run([self.probabilities], feed_dict=feed_dict)[0]
        return probabilities

    def fetcher(self, data, name):
        with tf.name_scope('fetcher'):
            if self.bias:
                data = self.add_bias(data)

            feed_dict = {self.x: data}
            quantity = self._sess.run([self._sess.graph.get_tensor_by_name(name + ':0')], feed_dict=feed_dict)[0]
        return quantity

    def train(self, data, verbose=True, tol=1e-4, maxiter=100):

        with tf.name_scope('train'):

            if self.bias:
                data = self.add_bias(data)
            self.lh = []
            fetch_dict = {}
            if self.fetch_names is not None:
                for name in self.fetch_names:
                    fetch_dict[name] = []
            else:
                self.fetch_names = []
            # Get the feed_dict
            feed_dict = {self.x: data,
                         self.eta: self.Q}
            for iteration in range(maxiter):
                # add additional options to trace the session execution
                options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
                run_metadata = tf.RunMetadata()

                if self.learn_hamiltonian is not None:
                    _, l, merged, fetches = self._sess.run(
                        [self.train_step, self.negqll, self.merged, self.fetchable_vars],
                        feed_dict=feed_dict,
                        options=options, run_metadata=run_metadata)
                    for i, name in enumerate(self.fetch_names):
                        fetch_dict[name].append(fetches[i])

                else:
                    _, l, merged = self._sess.run([self.train_step, self.negqll, self.merged], feed_dict=feed_dict,
                                                  options=options, run_metadata=run_metadata)
                self.lh.append(l)
                self.file_writer.add_run_metadata(run_metadata, 'step {}'.format(iteration))
                self.file_writer.add_summary(merged, iteration)
                if (iteration > 0) and (abs(self.lh[iteration] - self.lh[iteration - 1]) < tol):
                    if verbose:
                        print("Convergence reached after {} steps".format(iteration))
                    return fetch_dict
            # Create the Timeline object, and write it to a json file
            fetched_timeline = timeline.Timeline(run_metadata.step_stats)
            chrome_trace = fetched_timeline.generate_chrome_trace_format()
            with open('timeline_01.json', 'w') as f:
                f.write(chrome_trace)
            # break if we have reached the maximum number of iterations
            print("No convergence after {} steps!".format(maxiter))
        return fetch_dict

    @staticmethod
    def add_bias(X):
        n_samples, _ = X.shape
        return np.hstack([X, np.ones((n_samples, 1))])

    @staticmethod
    def partial_trace(tensor, dims_subsystems):
        '''
        Take the partial trace, leaving H_A, where H = H_A \otimes H_B \otimes H_N
        # Einsum does not allow you to do traces so we need this hacky method.
        # reorder the axes and take the trace of the final two.
        # These loops does the following (for 2 subsystems):
        # first loop
        # permute axes 0,a,b,c,a,b,c -> 0,a,a,c,b,b,c
        # permute axes 0,a,a,c,b,b,c -> 0,a,a,b,b,c,c
        # perform permutation
        # second loop
        # trace 0,a,a,b,b,c,c -> 0,a,a,b,b
        # trace 0,a,a,b,b -> 0,a,a

        :param tensor: NxN density matrix
        :param dims_subsystems: Dimensions of subsystems
        :return: Reduced density matrix
        '''
        dims = np.tile(dims_subsystems, 2)
        # insert batch dimension
        dims = np.insert(dims, [0], [-1])
        tensor = tf.reshape(tensor, dims)
        idx = list(range(len(dims)))
        n = len(dims_subsystems)
        for i in range(n - 1):
            idx[n + i + 1], idx[2 + i] = idx[2 + i], idx[n + i + 1]
        tensor = tf.transpose(tensor, perm=idx)
        for _ in range(n - 1):
            tensor = tf.trace(tensor)
        return tf.to_complex128(tensor)

    @staticmethod
    def SU_generators(dim):
        '''
        Generate the matrices for the complex N x N space of matrices, or SU(N) reprentations

        :param dim: Dimension of C^N x N
        :return: tensor of generators
        '''

        def direct_sum(a, b):
            '''
            Direct sum of two matrices

            :param a: n x p  matrix
            :param b: m x k matrix
            :return: (n + m )x (p+k) matrix
            '''
            dsum = np.zeros(np.add(a.shape, b.shape))
            dsum[:a.shape[0], :a.shape[1]] = a
            dsum[a.shape[0]:, a.shape[1]:] = b
            return dsum

        f = np.zeros((dim, dim, dim ** 2 - 1), dtype=complex)
        c = 0

        for j in range(dim):
            for k in range(dim):
                if k < j:
                    f[j, k, c] = 1
                    f[k, j, c] = 1
                    c += 1
                if k < j:
                    f[j, k, c] = 1j
                    f[k, j, c] = -1j
                    c += 1
        d = dim

        for k in reversed(range(1, dim + 1)):
            if k == d:
                h_d = np.sqrt(2 / (d * (d - 1))) * (
                    direct_sum(np.identity(d - 1), np.array([[1 - d]])))
                f[:, :, c] = h_d
                c += 1
            if (1 < k) & (k < dim):
                d -= 1
                h_d = np.sqrt(2 / (d * (d - 1))) * (
                    direct_sum(np.identity(d - 1), np.array([[1 - d]])))
                f[:, :, c] = direct_sum(h_d, np.array([[0]]))
                c += 1
        return tf.constant(f, dtype=tf.complex128)
        # return tf.constant(f[:, :, 2].reshape(2, 2, 1), dtype=tf.complex128)

    def matrix_log(self, M):
        '''
        Calculate matrix log2 through diagonalization of Hermitian matrix  M = U^-1 D U

        :param M: n x n matrix
        :return: n x n matrix
        '''
        rx, Ux = tf.linalg.eigh(M)
        Ux_inv = tf.linalg.adjoint(Ux)
        rx = tf.cast(tf.log(tf.clip_by_value(tf.real(rx), 1e-13, 1e13)), rx.dtype)
        tx = tf.linalg.LinearOperatorDiag(rx).to_dense()
        return tf.matmul(Ux, tf.matmul(tx, Ux_inv))

    def matrix_exp(self, M):
        '''
        Calculate matrix log2 through diagonalization of Hermitian matrix  M = U^-1 D U

        :param M: n x n matrix
        :return: n x n matrix
        '''
        rx, Ux = tf.linalg.eigh(M)
        Ux_inv = tf.linalg.adjoint(Ux)
        rx = tf.cast(tf.exp(tf.real(rx)), rx.dtype)
        tx = tf.linalg.LinearOperatorDiag(rx).to_dense()
        return tf.matmul(Ux, tf.matmul(tx, Ux_inv))

    def matrix_log_2x2(self, M):
        '''
        Calculate matrix log2 through diagonalization of Hermitian matrix  M = U^-1 D U

        :param M: n x n matrix
        :return: n x n matrix
        '''
        rx = 0.5 * tf.stack(
            [(M[:, 0, 0] + M[:, 1, 1]) - tf.sqrt((M[:, 0, 0] - M[:, 1, 1]) ** 2 + 4 * M[:, 0, 1] * M[:, 1, 0]),
             (M[:, 0, 0] + M[:, 1, 1]) + tf.sqrt((M[:, 0, 0] - M[:, 1, 1]) ** 2 + 4 * M[:, 0, 1] * M[:, 1, 0])], axis=1)
        I = tf.ones_like(M[:, 0, 0])
        lam_m = tf.stack([(rx[:, 1] - M[:, 1, 1]) / M[:, 0, 1], I], axis=1)
        lam_m /= tf.reshape(tf.linalg.norm(lam_m, axis=1), (-1, 1))
        lam_p = tf.stack([(rx[:, 0] - M[:, 1, 1]) / M[:, 0, 1], I], axis=1)
        lam_p /= tf.reshape(tf.linalg.norm(lam_p, axis=1), (-1, 1))
        Ux = tf.stack([lam_m, lam_p], axis=2)
        Ux_inv = tf.linalg.adjoint(Ux)

        rx = tf.cast(tf.log(tf.clip_by_value(tf.real(rx), 1e-13, 1e13)), rx.dtype)
        tx = tf.linalg.LinearOperatorDiag(rx).to_dense()
        return tf.matmul(Ux, tf.matmul(tx, Ux_inv))
