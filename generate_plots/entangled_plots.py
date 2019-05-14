from qperceptron import tf_qperceptron
import numpy as np
import matplotlib.pyplot as plt

epsilon = 0.01
nsteps = 1000
BIAS = True
number_of_copies = 40
# noise = 0.3
tol = 10e-5
# Problem
X_1 = np.tile([1, 1], (number_of_copies, 1))
X_2 = np.tile([-1, -1], (number_of_copies, 1))
X_3 = np.tile([1, -1], (number_of_copies, 1))
X_4 = np.tile([-1, 1], (number_of_copies, 1))
# X_5 = np.tile([0, 0], (number_of_copies, 1))
# X_6 = np.tile([1, -1], (number_of_copies, 1))

Y_1 = np.tile([-1], (number_of_copies, 1))
Y_2 = np.tile([-1], (number_of_copies, 1))
Y_3 = np.tile([1], (number_of_copies, 1))
Y_4 = np.tile([1], (number_of_copies, 1))
# Y_5 = np.tile(-1], (number_of_copies, 1))
# Y_6 = np.tile([-1], (number_of_copies, 1))

# Y_1[:int(noise * number_of_copies)] *= -1
# Y_2[:int(noise * number_of_copies)] *= -1
# Y_3[:int(noise * number_of_copies)] *= -1
# Y_4[:int(noise * number_of_copies)] *= -1
# Y_5[:int(noise * number_of_copies)] *= -1
# Y_6[:int(noise * number_of_copies)] *= -1

X = np.vstack((X_1, X_2, X_3, X_4))
y = np.vstack((Y_1, Y_2, Y_3, Y_4)).flatten()

# TRAINING
qp = tf_qperceptron.QuantumLearning(bias=BIAS, n_classes=2, d_or_c='discr')
qp.get_statistics(np.copy(X), np.copy(y))

qp.build_graph([2, 2], epsilon=epsilon, complex_weights=True, device='GPU')
qp.train(X, X.shape[0], maxiter=nsteps, tol=tol)

plt.figure()
plt.plot(qp.lh)
plt.xlabel('iteration')
plt.ylabel('likelihood')  # plt.show()
# p = m.predict(X)
cmap = plt.cm.get_cmap('viridis')
plt.rc('font', size=15)

blue = cmap(0.0)
red = cmap(1.0)
h = .05
max_grid = 5
x_min, x_max = 0 - max_grid, 0 + max_grid
y_min, y_max = 0 - max_grid, 0 + max_grid
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
Z = qp.predict(np.c_[xx.ravel(), yy.ravel()])

Z = np.real(Z[:, 0] - Z[:, 1])

Z = Z.reshape(xx.shape)
fig, ax = plt.subplots(1, 1)
ax.contour(xx, yy, Z, cmap=cmap)
# Plot also the training points
y = y.flatten()
np.random.seed(123)
spread = 0.3
ax.scatter(X[(y == -1), 0] + np.random.uniform(-spread, spread, np.sum((y == -1))),
           X[(y == -1), 1] + np.random.uniform(-spread, spread, np.sum((y == -1))),
           c=blue, label='-1', s=5)
ax.scatter(X[(y == 1), 0] + np.random.uniform(-spread, spread, np.sum((y == 1))),
           X[(y == 1), 1] + np.random.uniform(-spread, spread, np.sum((y == 1))),
           c=red, label='+1', s=5)
ax.set_title('Entangled Perceptron')
m = plt.cm.ScalarMappable(cmap=cmap)
m.set_array(Z)

plt.colorbar(m, boundaries=np.linspace(-1, 1, 11))
ax.set_xlabel('$x_0$')
ax.set_ylabel('$x_1$')
ax.legend()

plt.show()
