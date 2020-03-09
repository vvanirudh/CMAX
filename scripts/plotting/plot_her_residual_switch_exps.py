import matplotlib.pyplot as plt
import numpy as np
import csv
import argparse
import os.path as osp
import matplotlib

matplotlib.rcParams['ps.useafm'] = True
matplotlib.rcParams['pdf.use14corefonts'] = True
matplotlib.rcParams['text.usetex'] = False
# plt.rcParams.update({'font.size': 20})
plt.style.use('seaborn-whitegrid')

parser = argparse.ArgumentParser()
parser.add_argument('--env_name', type=str, default='FetchPush-v1')
parser.add_argument('--residual', action='store_true')
parser.add_argument('--switch', action='store_true')
parser.add_argument('--switch_residual', action='store_true')
parser.add_argument('--hardcoded', action='store_true')
parser.add_argument('--zoom', action='store_true')
parser.add_argument('--save', action='store_true')


args = parser.parse_args()

cmap = plt.get_cmap('tab10')
colors = cmap(np.linspace(0, 1, 10))


def read_csv_data(file_name, prop=False):
    data = []
    with open(file_name, 'r') as csvfile:
        file_reader = csv.DictReader(csvfile)
        for row in file_reader:
            if not prop:
                data.append(
                    [float(row['n_steps']), float(row['success_rate'])])
            else:
                data.append([float(row['n_steps']), float(
                    row['success_rate']), float(row['prop_hardcoded'])])

    data = np.array(data)
    return data


# Read Hardcoded data
if args.hardcoded:
    hardcoded_file_name = osp.join(
        'logs', 'hardcoded', 'Residual'+args.env_name, 'progress.csv')
    with open(hardcoded_file_name, 'r') as csvfile:
        file_reader = csv.DictReader(csvfile)
        for row in file_reader:
            hardcoded_success_rate = np.round(float(row['success_rate']), 2)

# Read HER CSV file
her_file_name = osp.join('logs', 'her', args.env_name, 'progress.csv')
her_data = read_csv_data(her_file_name)

# Read Residual HER CSV File
if args.residual:
    residual_her_file_name = osp.join(
        'logs', 'residual_her', 'Residual'+args.env_name, 'progress.csv')
    residual_her_data = read_csv_data(residual_her_file_name)

# Read Switch HER CSV File
if args.switch:
    switch_her_file_name = osp.join(
        'logs', 'switch_her', args.env_name, 'progress.csv')
    switch_her_data = read_csv_data(switch_her_file_name, prop=True)

# Read Switch Residual HER CSV File
if args.switch_residual:
    switch_residual_her_file_name = osp.join(
        'logs', 'switch_residual_her', 'Residual'+args.env_name, 'progress.csv')
    switch_residual_her_data = read_csv_data(
        switch_residual_her_file_name, prop=True)
# Plot data
if args.switch or args.switch_residual:
    plt.subplot(1, 2, 1)
if args.hardcoded:
    plt.plot(her_data[:, 0], [
        hardcoded_success_rate for _ in range(her_data.shape[0])], color='black', linestyle=':', label='Hardcoded')

plt.plot(her_data[:, 0], her_data[:, 1], color=colors[0],
         label='HER + DDPG', marker='.')
if args.residual:
    plt.plot(residual_her_data[:, 0], residual_her_data[:,
                                                        1], color=colors[1], label='Residual HER + DDPG', marker='.')
if args.switch:
    plt.plot(switch_her_data[:, 0], switch_her_data[:, 1],
             color=colors[2], label='Switch HER + DDPG', marker='.')
if args.switch_residual:
    plt.plot(switch_residual_her_data[:, 0], switch_residual_her_data[:,
                                                                      1], color=colors[3], label='Switch Residual HER + DDPG', marker='.')

plt.xlabel('Number of simulator steps')
plt.ylabel('Success rate')
if args.zoom:
    plt.xlim([0, 50000])
plt.legend()

if args.switch or args.switch_residual:
    plt.subplot(1, 2, 2)
    if args.switch:
        plt.plot(switch_her_data[:, 0], switch_her_data[:, 2],
                 color=colors[2], label='Switch HER + DDPG')
    if args.switch_residual:
        plt.plot(switch_residual_her_data[:, 0], switch_residual_her_data[:, 2],
                 color=colors[3], label='Switch Residual HER + DDPG')
    plt.xlabel('Number of simulator steps')
    plt.ylabel('Proportion of steps hardcoded controller was used')
    if args.zoom:
        plt.xlim([0, 50000])

plt.suptitle('Performance on '+args.env_name)

plt.gcf().set_size_inches([10, 6])
if args.save:
    filename = 'plots/'+args.env_name
    filename = filename + '_hardcoded' if args.hardcoded else filename
    filename = filename + '_residual' if args.residual else filename
    filename = filename + '_switch' if args.switch else filename
    filename = filename + '_switchresidual' if args.switch_residual else filename
    filename = filename + '.png'
    plt.savefig(filename, format='png')

plt.show()
