import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

path = 'data/noise_dependence/'
cmap = plt.cm.get_cmap('viridis')
plt.rc('font', size=15)

blue = cmap(0.0)
red = cmap(1.0)
type_of_stat = 'MSE' # choose 'MSE' or ' lh

dir_list = os.listdir(path)
noise_files = []
noise_files_langevin = []
noise_amount_list = []
noise_amount_list_langevin = []
for f in dir_list:
    name = f.split('_')

    if name[3] == 'mz':
        noise_files.append(f)
        noise_amount = name[4]
        print(noise_amount)
        noise_amount_list.append(float(noise_amount))

idx = np.argsort(noise_amount_list)
noise_amount_list = [noise_amount_list[i] for i in idx]
noise_files = [noise_files[i] for i in idx]
idx = np.argsort(noise_amount_list_langevin)
noise_amount_list_langevin = [noise_amount_list_langevin[i] for i in idx]
noise_files_langevin = [noise_files_langevin[i] for i in idx]
plt.figure()
plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
plt.rc('font', size=15)

means = []
vars = []
mean_lh = []
var_lh = []
for i, file in enumerate(noise_files):
    df = pd.read_csv(path + file)
    mean_lh.append((df[type_of_stat + '_qm_test'] + df[type_of_stat+ '_cl_test']).mean())
    var_lh.append((df[type_of_stat + '_qm_test'] + df[type_of_stat+ '_cl_test']).std())
    means.append((-df[type_of_stat + '_qm_test'] + df[type_of_stat + '_cl_test']).mean())
    vars.append((-df[type_of_stat + '_qm_test'] + df[type_of_stat + '_cl_test']).std())
plt.errorbar(noise_amount_list, means, yerr=vars, ecolor='gray', color=blue)
plt.plot(noise_amount_list, means, 'o', markersize=8, color=blue)
plt.plot(noise_amount_list, [0 for _ in range(len(noise_amount_list))], linestyle='--', markersize=8, color='black')

plt.xlabel('Noise %')
# plt.yscale('log')
# plt.legend()
plt.ylabel(r'$\Delta${}'.format(type_of_stat))
plt.title('Teacher Student')


plt.show()
