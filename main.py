__author__ = 'cipriancorneanu'

from betti import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from sklearn import linear_model
import seaborn as sns
import pandas as pd

root = "/Users/cipriancorneanu/Research/data/new_results/obj_rec/"
n_nodes = {'mlp_300_100': 400, 'mlp_300_200_100': 600, 'conv_2': 650, 'conv_4': 906, 'alexnet': 1162, 'conv_6': 1418,
           'resnet18': 1736, 'vgg16': 1930, 'resnet34': 1736, 'resnet50': 6152}

epcs = [290]
TRIALS = [40, 45, 46, 47, 48, 49, 50, 56, 57, 58, 59, 64, 65, 66, 67, 68, 70, 71, 72]
NETS = ['conv_2', 'conv_4', 'alexnet', 'conv_6', 'vgg16', 'resnet18']
DATASETS = ['mnist', 'svhn', 'cifar10', 'cifar100', 'imagenet']

''' Plot AU rec'''
def get_data(nets, trials):
    pts = []
    for i_net, net in enumerate(nets):
        for i_dataset, dataset in enumerate(DATASETS):
            directory = root + net + '_' + dataset + '/'
            if os.path.exists(directory):
                for i, epc in enumerate([290]):
                    # If file exists, read and plot
                    for trial in trials:
                        if os.path.exists(directory + 'adj_epc' + str(epc) + '_trl{}_0.4.bin.out'.format(trial)):
                            # read data
                            data = np.asarray([read_results(directory, epcs, trl=trl, dim=1, persistence=0.02
                                                            ) for trl in [trial]][0])
                            if len(data) > 0:
                                x_ref = np.linspace(0.05, 0.4, 200)
                                # read generalization gap
                                with open(root + 'losses/' + net + '_' + dataset + '/stats_trial_' + str(trial) + '.pkl',
                                          'rb') as f:
                                    loss = pkl.load(f)
                                    print(net, dataset, trial, loss[epc]['acc_tr'] - loss[epc]['acc_te'])
                                    if loss[epc]['acc_tr'] - loss[epc]['acc_te'] > 1 and loss[epc]['acc_tr'] - loss[epc]['acc_te'] < 100:

                                        ggap = (loss[epc]['acc_tr'] - loss[epc]['acc_te'])
                                        betti = betti_max(x_ref, data[i]) / n_nodes[net]
                                        epsilon = epsilon_max(x_ref, data[i])

                                        pts.append({'net':net, 'dataset':dataset, 'ggap':ggap, 'trial':trial, 'betti':betti, 'epsilon':epsilon})



                                        # axes[i_net, i_dataset].scatter(gap[net+'_'+dataset][str(trial)]/100, betti_max(x_ref, data[i]))
                        else:
                            # print("No such file {}".format(directory+'adj_epc'+str(epc)+'_trl'+str(trial)+'_0.4.bin.out'))
                            pass

    return pts


data = get_data(NETS, TRIALS)


''' Filter per nets '''
data = [[item for item in data if item['net']==net] for net in NETS]



fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(21, 9), sharex=True, sharey=False)
axes.set_xlabel('$\Delta_{acc}[\%]$')
axes.set_ylabel('$betti_1^{max}[cavs/node]$')
axes.grid(which='major', linestyle='--')
axes.set_ylim([0,0.5])
axes.set_xlim([0,1])


for net in data:
    x = [np.asarray(item['ggap'])/100 for item in net]
    y = [item['epsilon'] for item in net]

    axes.scatter(x, y, label=net[0]['net'])

    ''' Regression '''
    sns.set(color_codes=True)
    tips = pd.DataFrame(list(zip(x, y)),
               columns =['ggap', 'epsilon'])

    sns.regplot(x="ggap", y="epsilon", data=tips, order=1)

fig.legend()
fig.savefig(root + 'cvpr2020/all.png')



for net in data:
    fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(21, 9), sharex=True, sharey=False)
    axes.set_xlabel('$\Delta_{acc}[\%]$')
    axes.set_ylabel('$betti_1^{max}[cavs/node]$')
    axes.grid(which='major', linestyle='--')
    axes.set_ylim([0,0.5])
    axes.set_xlim([0,1])

    x = [np.asarray(item['ggap'])/100 for item in net]
    y = [item['epsilon'] for item in net]

    ''' Regression '''
    sns.set(color_codes=True)
    tips = pd.DataFrame(list(zip(x, y)),
               columns =['ggap', 'epsilon'])

    sns.regplot(x="ggap", y="epsilon", data=tips, order=1)


    for dataset in DATASETS:
        a = [item for item in net if item['dataset']==dataset]

        x = [np.asarray(item['ggap'])/100 for item in a]
        y = [item['epsilon'] for item in a]

        axes.scatter(x, y, label=dataset)


    fig.legend()
    fig.savefig(root + 'cvpr2020/'+net[0]['net']+'.png')
