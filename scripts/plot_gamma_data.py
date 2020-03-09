import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 45})

gamma_data = pickle.load(open(
    '/home/avemula/workspaces/odium_ws/src/odium/save/pr2_experiments_broken_joint.pkl', 'rb'))

gammas = np.array([0.01, 0.1, 1.0, 10.0, 100.0])
gammas = 1 / gammas


gamma_data = np.array(gamma_data).squeeze()
gamma_data_mean = np.mean(gamma_data, axis=0)
gamma_data_std = np.std(gamma_data, axis=0) / np.sqrt(gamma_data.shape[0])


plt.plot(gammas, gamma_data_mean, 'r-')
plt.fill_between(gammas, gamma_data_mean - gamma_data_std,
                 gamma_data_mean + gamma_data_std, color='red', alpha=0.2)

# plt.plot(gammas, [100 for _ in range(len(gammas))],
#          'k--', label='Max timesteps allowed')
plt.xscale('log')
plt.xlabel('Length scale $\gamma$ of RBF Kernel')
plt.ylabel('Number of timesteps to reach the goal')
plt.xticks(gammas)
# plt.legend()
plt.title(
    'Performance with varying length scale \nin 7D arm planning')

plt.gcf().set_size_inches([12, 12])
plt.savefig(
    '/home/avemula/Documents/Drafts/RSS-2020-paper/fig/gamma.pdf', format='pdf')
plt.savefig('plot/gamma.png', format='png')

plt.show()
