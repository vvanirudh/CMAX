import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import os.path as osp
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = True
# plt.rcParams.update({'font.size': 20})
plt.style.use('seaborn-whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument('--her_env', type=str, default='FetchPush-v1')
parser.add_argument('--residual_her_env', type=str,
                    default='ResidualFetchPush-v1')
parser.add_argument('--psdp_her_env', type=str, default=None)

args = parser.parse_args()


def read_csv_data(file_name):
    data = []
    with open(file_name, 'r') as csvfile:
        file_reader = csv.DictReader(csvfile)
        for row in file_reader:
            data.append(
                [float(row['n_steps']), float(row['success_rate'])])

    data = np.array(data)
    return data


# Read HER CSV file
her_file_name = osp.join('logs', 'her', args.her_env, 'progress.csv')
her_data = read_csv_data(her_file_name)

# Read Residual HER CSV File
residual_her_file_name = osp.join(
    'logs', 'residual_her', args.residual_her_env, 'progress.csv')
residual_her_data = read_csv_data(residual_her_file_name)

if args.psdp_her_env is not None:
    # Read Residual PSDP HER CSV file
    psdp_her_file_name = osp.join(
        'logs', 'residual_psdp_her', args.psdp_her_env, 'progress.csv')
    psdp_her_data = read_csv_data(psdp_her_file_name)

# Plot data
plt.plot(her_data[:, 0], her_data[:, 1], color='red', label='HER + DDPG')
plt.plot(residual_her_data[:, 0], residual_her_data[:,
                                                    1], color='blue', label='Residual HER + DDPG')
if args.psdp_her_env is not None:
    plt.plot(psdp_her_data[:, 0], psdp_her_data[:, 1],
             color='green', label='Last step policy')

plt.xlabel('Number of simulator steps')
plt.ylabel('Success rate')
plt.title('Performance on '+args.her_env)
plt.legend()

plt.show()
