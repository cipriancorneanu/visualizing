__author__ = 'cipriancorneanu'

from betti import *
import numpy as np
import matplotlib.pyplot as plt
import os
import pickle as pkl
from sklearn import linear_model
import seaborn as sns
import pandas as pd

root = "/Users/cipriancorneanu/Desktop/new_results/obj_rec/"
n_nodes = {'mlp_300_100': 400, 'mlp_300_200_100': 600, 'conv_2': 650, 'conv_4': 906, 'alexnet': 1162, 'conv_6': 1418,
           'resnet18': 1736, 'vgg16': 1930, 'resnet34': 1736, 'resnet50': 6152}

epcs = [290]
TRIALS = [40, 50, 59, 68, 70, 71, 72]
NETS = ['conv_2', 'conv_4', 'conv_6', 'alexnet', 'resnet18', 'vgg16']
DATASETS = ['imagenet']

''' Plot AU rec'''
def get_data(nets, trials):
    pts = {}
    for i_net, net in enumerate(nets):
        pts[net] = {'ggap': [], 'betti': [], 'epsilon': [], 'trial': [], 'dataset':[]}
        for i_dataset, dataset in enumerate(DATASETS):
            pts[net]['dataset'].append(dataset)
            directory = root + net + '_' + dataset + '/'

            if os.path.exists(directory):
                for i, epc in enumerate([290]):
                    # If file exists, read and plot
                    for trial in trials:
                        pts[net]['trial'].append(trial)

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
                                        pts[net]['ggap'].append((loss[epc]['acc_tr'] - loss[epc]['acc_te']))
                                        pts[net]['betti'].append(betti_max(x_ref, data[i]) / n_nodes[net])
                                        pts[net]['epsilon'].append(epsilon_max(x_ref, data[i]))



                                        # axes[i_net, i_dataset].scatter(gap[net+'_'+dataset][str(trial)]/100, betti_max(x_ref, data[i]))
                        else:
                            # print("No such file {}".format(directory+'adj_epc'+str(epc)+'_trl'+str(trial)+'_0.4.bin.out'))
                            pass

    return pts


data = get_data(NETS, TRIALS)
fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(21, 9), sharex=True, sharey=False)
axes.set_xlabel('$\Delta_{acc}[\%]$')
axes.set_ylabel('$betti_1^{max}[cavs/node]$')
axes.grid(which='major', linestyle='--')
axes.set_ylim([0,0.5])
axes.set_xlim([0,1])

for model in data:
    x = np.asarray(data[model]['ggap'])/100
    y = data[model]['epsilon']
    #x = [np.power(item, 0.25) for item in x]

    axes.scatter(x, y, label=model)

    ''' Regression '''
    '''
    sns.set(color_codes=True)
    tips = pd.DataFrame(list(zip(x, y)),
               columns =['ggap', 'epsilon'])

    ax = sns.regplot(x="ggap", y="epsilon", data=tips, order=1)
    '''


fig.legend()
fig.savefig(root + 'cvpr2020/all_p_0.02_lin_imagenet.png')