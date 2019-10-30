__author__ = 'cipriancorneanu'

import numpy as np
import array
import matplotlib.pyplot as plt
import time

def read_bin(fname):
    header = array.array("L")
    values = array.array("d")

    with open(fname, mode='rb') as file: # b is important -> binary
        header.fromfile(file, 3)
        values.fromfile(file, int(header[2]*header[2]))
        values = list(values)

    values = np.asarray([float("{0:.2f}".format(1-x)) for x in np.asarray(values)])
    values = np.reshape(values, (header[2], header[2]))

    return values


def read_bin_out(fname):
    '''
    :param fname: Binary file name as writtne by DIPHA
    :return: contents of binary file name (dimensions, birth_values, death_values)
    '''
    header = array.array("L")
    dims = array.array("l")
    values = array.array("d")

    with open(fname, mode='rb') as file: # b is important -> binary
        header.fromfile(file, 3)
        dims.fromfile(file, 3*header[2])
        dims = list(dims[::3])
        file.seek(24)
        values.fromfile(file, 3*(header[2]))
        values = list(values)
        birth_values = values[1::3]
        death_values = values[2::3]

    return dims, birth_values, death_values


def read_betti(fname, dimension, persistence):
    '''
    Read binar file name from DIPHA and transform to betti with normed x_axis
    :param fname:
    :param dimension:
    :param persistence:
    :return:
    '''
    dims, b_values, d_values =  read_bin_out(fname)
    x, betti = persistpairs2betti(b_values, d_values, dims, dimension, persistence)
    return norm_x_axis(x, betti, np.linspace(0.05, 0.4, 200))


def persistpairs2betti(birth, death, dims, dimension, persistence):
    d = [i for i,x in enumerate(dims) if x==dimension]
    x_birth = [float("{0:.4f}".format(x)) for x in np.asarray(birth)[d]]
    x_death = [float("{0:.4f}".format(x)) for x in np.asarray(death)[d]]

    ''' Ignore low persistence '''
    filter = [i for i, (xb, xd) in enumerate(zip(x_birth, x_death)) if (xd-xb) > persistence]
    x_birth = [x_birth[x] for x in filter]
    x_death = [x_death[x] for x in filter]
    x = sorted(x_birth + x_death)

    b_x = [x.index(item) for item  in x_birth]
    b_y = [x.index(item) for item  in x_death]

    delta_birth = np.zeros_like(x, dtype=np.int)
    delta_death = np.zeros_like(x, dtype=np.int)

    acc = 0
    for item in b_x:
        acc = acc+1
        delta_birth[item:] = acc

    acc = 0
    for item in b_y:
        acc = acc+1
        delta_death[item:] = acc


    return x, delta_birth - delta_death


def betti_max(x, curve):
    return np.max(curve)

def epsilon_max(x, curve):
    return x[np.argmax(curve)]

def compute_integral(curve):
    '''
    :param curve: list of real values
    :return: sum of curve
    '''
    return np.sum(curve)


def compute_auc(curve):
    '''
    :param curve: list of values
    :return: AUC for curve
    '''
    auc = [np.sum(curve[:end])/np.sum(curve) for end in range(len(curve))]
    return auc


def plot(axes, data, epochs, i_epochs, N):
    for i, (ax, i_epcs) in enumerate(zip(axes, i_epochs)):
        for i_epc in i_epcs:
            x, betti = data[i_epc]
            betti = np.array([betti[np.argmin([np.abs(a-b) for b in x])] for a in x_ref])
            ax.semilogx(x_ref, betti/N, label='epc{}'.format(epochs[i_epc]))
            ax.set_ylabel("cavs/node")
            ax.legend()
            ax.grid()


def norm_x_axis(x, curve, x_ref):
    start = time.time()
    x_axis = np.array([curve[np.argmin([np.abs(a-b) for b in x])] for a in x_ref])
    #print("{}s".format(time.time()-start))
    return x_axis


def read_results_part(path, epcs, parts, trl, dim, persistence=0.04):
    return [[read_betti(path+'adj_epc{}_trl{}_part{}_0.4.bin.out'.format(epc, trl, part), dimension=dim, persistence=persistence) for epc in epcs] for part in parts]


def read_results(path, epcs, trl, dim=1, persistence=0.01):
    return [read_betti(path+'adj_epc{}_trl{}_0.4.bin.out'.format(epc, trl), dimension=dim, persistence=persistence) for epc in epcs]


def evaluate_node_importance(adj, epsilon):
    node_importance = np.zeros(adj.shape[0])
    adj[adj<epsilon]=0
    adj[adj>=epsilon]=1
    importance = np.sum(adj, axis=0)
    print(np.sort(importance))
    return  np.argsort(importance)[::-1]


def compute_node_importance(net, dataset, epcs, trl):
    data = read_results(root+'results_15_04/lenet_mnist/', epcs_lenet, trl, dim=1)

    x_ref = np.linspace(0.05, 0.4, 200)
    maxcav= [np.argmax(norm_x_axis(epc[0], epc[1], x_ref)) for epc in part]

    ''' Get epc and eps '''
    epc, eps = 0 , 0

    ''' Evaluate node importance '''
    fname = root + 'adj_epc{}_trl{}.bin'.format(epc, trl)
    adj = read_bin_out(fname, dimension=1, persistence=0.03)
    node_importance = evaluate_node_importance(adj, epsilon=eps)

    return node_importance



