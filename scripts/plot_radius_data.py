import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
plt.rcParams.update({'font.size': 45})

radius_data = pickle.load(open(
    '/home/avemula/workspaces/odium_ws/src/odium/save/fetch_experiments_radius.pkl', 'rb'))

radiuss = np.array([0.01, 0.02, 0.04, 0.06, 0.08])


radius_data = np.array(radius_data).squeeze()
radius_data_mean = np.mean(radius_data, axis=0)
radius_data_std = np.std(radius_data, axis=0) / np.sqrt(radius_data.shape[0])


plt.plot(radiuss, radius_data_mean, 'b-')
plt.fill_between(radiuss, radius_data_mean - radius_data_std,
                 radius_data_mean + radius_data_std, color='blue', alpha=0.2)
plt.xticks(radiuss)

# plt.plot(radiuss, [100 for _ in range(len(radiuss))],
#          'k--', label='Max timesteps allowed')
# plt.xscale('log')
plt.xlabel('Radius of the hypersphere $\delta$')
plt.ylabel('Number of timesteps to reach the goal')
# plt.legend()
plt.title(
    'Performance with varying radius \nin 4D planar pushing')

plt.gcf().set_size_inches([12, 12])
plt.savefig(
    '/home/avemula/Documents/Drafts/RSS-2020-paper/fig/radius.pdf', format='pdf')
plt.savefig('plot/radius.png', format='png')

plt.show()
