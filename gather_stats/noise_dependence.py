from sklearn.model_selection import train_test_split

import numpy as np
import pandas as pd

from src.qperceptron import QPerceptron
from src.cl_perceptron import CLQPerceptron
from src.qperceptron_lib import get_cl_statistics, get_qp_mz_statistics


def qx(_X, _sample):
    # find copies of our sample
    _idx = np.where((_X == tuple(_sample)).all(axis=1))[0]
    # calcualte emperical probability of sample
    return len(_idx) / _X.shape[0]


number_of_duplicates = 5
number_of_samples = 600
number_of_samples = int(number_of_samples / number_of_duplicates)
dim = 8
n_runs = 100
stats = np.zeros((n_runs, 6))

eta = 0.01
nsteps = 100000
BIAS = True
tol=10e-5
val_split = 0.2

noise_list = [x * 0.05 for x in range(0,11)]

for noise in noise_list:
    np.random.seed(600)
    print(noise)
    for run in range(n_runs):

        samples = 2 * np.random.randint(0, 2, (number_of_samples, dim)) - 1
        samples = np.repeat(samples, repeats=number_of_duplicates, axis=0)

        labels = np.sign(np.dot(samples, np.random.randn(dim)))
        labels = np.repeat(labels, repeats=number_of_duplicates, axis=0)
        samples = np.repeat(samples, repeats=number_of_duplicates, axis=0)

        samples_train, samples_test, labels_train, labels_test = train_test_split(samples, labels, test_size=val_split)

        noise_idx = np.random.permutation(list(range(samples_train.shape[0])))[:int(noise * samples_train.shape[0])]
        labels_train[noise_idx] *= -1

        qp = QPerceptron(np.copy(samples_train), labels_train, bias=BIAS)
        qp.train(nsteps, eta, calculate_loss=False, tol=tol, verbose=True)

        cl = CLQPerceptron(np.copy(samples_train), labels_train, bias=BIAS)
        cl.train(nsteps, eta, calculate_loss=False, tol=tol, verbose=False)
        q = np.zeros(len(labels_test))
        # gather statistics for each sample, store these based on index
        for i in range(len(labels_test)):
            q[i] = qx(samples_test, samples_test[i, :])

        stats[run, 0], stats[run, 1], stats[run, 2], _, _ = get_qp_mz_statistics(qp, np.copy(samples_test), labels_test,
                                                                                 q, return_stats=True, verbose=True)
        stats[run, 3], stats[run, 4], stats[run, 5], _, _ = get_cl_statistics(cl, np.copy(samples_test), labels_test,
                                                                              q, return_stats=True, verbose=True)

    df = pd.DataFrame(stats,
                      columns=['MSE_qm_test', 'MAE_qm_test', 'lh_qm_test', 'MSE_cl_test', 'MAE_cl_test', 'lh_cl_test'])
    df.to_csv('../data/noise_dependence/' + 'NL_statistics_noise_mz_{:1.2f}'.format(noise) + '.csv')
