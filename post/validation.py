"""
    Do embedding clustering and generate similar subgraphs.
"""
if __name__ == '__main__':
    import sys

    sys.path.append('../')

import os
import pickle
import math
import random
from random import shuffle

from tqdm import tqdm
import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import jaccard
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import seaborn as sns
import pandas as pd

from learning.rgcn import Model
from post.utils import *


def get_decoys(mode='pdb', annots_dir='../data/annotated/pockets_nx'):
    """
    Build decoys set for validation.
    """
    if mode=='pdb':
        fp_dict = {}
        for g in os.listdir(annots_dir):
            try:
                lig_id = g.split(":")[2]
            except:
                print(f"failed on {g}")
            _,_,_,fp = pickle.load(open(os.path.join(annots_dir, g), 'rb'))
            fp_dict[lig_id] = fp
        decoy_list = list(fp_dict.values())
        decoy_dict = {k:(v, decoy_list) for k,v in fp_dict.items()}
        return decoy_dict
    pass
def distance_rank(active, pred, decoys, dist_func=jaccard):
    """
        Get rank of prediction in `decoys` given a known active ligand.
    """

    pred_dist = dist_func(active, pred)
    rank = 0
    for lig in decoys:
        d = dist_func(active, lig)
        if d < pred_dist:
            rank += 1
    return 1- (rank / (len(decoys) + 1))

def decoy_test(model, decoys, edge_map, embed_dim, test_graphlist=None, test_graph_path="../data/annotated/pockets_nx"):
    """
        Check performance against decoy set.
        decoys --> {'ligand_id', ('expected_FP', [decoy_fps])}
        test_set --> [annot_graph_path,]
        :model trained model
        :test_set inputs for model to test (RNA graphs)
        :decoys dictionary with list of decoys for each input to test.
        :test_graphlist list of graph names to use in the test.

        :return: enrichment score
    """
    ranks = []

    if test_graphlist is None:
        test_graphlist = os.listdir(test_graph_path)
        
    for g_path in test_graphlist:
        g,_,_,_ = pickle.load(open(os.path.join(test_graph_path, g_path), 'rb'))
        try:
            true_id = g_path.split(":")[2]
        except:
            print(f">> failed on {g_path}")
            continue
        nx_graph, dgl_graph = nx_to_dgl(os.path.join(test_graph_path, g_path), edge_map, embed_dim)
        _,fp_pred= model(dgl_graph)
        fp_pred = fp_pred.detach().numpy() > 0.5
        # fp_random = np.random.choice([0, 1], size=(166,), p=[1./2, 1./2])
        active = decoys[true_id][0]
        decs = decoys[true_id][1]
        rank = distance_rank(active, fp_pred, decs)
        ranks.append(rank)
    return np.mean(ranks), ranks

def ablation_results():
    modes = ['', '_bb-only', '_wc-bb', '_wc-bb-nc', '_no-label', '_label-shuffle']
    decoys = get_decoys()
    ranks, methods = [], []
    for m in modes:

        if m == '':
            graph_dir = "../data/annotated/pockets_nx"
            run = 'small_no_rec_2'
        else:
            graph_dir = "../data/annotated/pockets_nx" + m
            run = 'small_no_rec' + m
        edge_map = get_edge_map(graph_dir)
        num_edge_types = len(edge_map)


        dims = [32] * 3
        # dims = [32]*6
        attributor_dims = [32, 166]

        model = Model(dims=dims, attributor_dims=attributor_dims, num_rels=num_edge_types, num_bases=-1)
        model.load_state_dict(torch.load(f'../trained_models/{run}/{run}.pth', map_location='cpu')['model_state_dict'])

        graph_ids = pickle.load(open(f'../results/{run}/splits.p', 'rb'))

        acc, ranks_this  = decoy_test(model, decoys, edge_map, 32, test_graphlist=graph_ids['test'], test_graph_path=graph_dir)
        ranks.extend(ranks_this)
        methods.extend([m]*len(ranks_this))
        print("test", 1-acc)


    df = pd.DataFrame({'rank': ranks, 'method':methods})
    ax = sns.violinplot(x="method", y="rank", data=df, color='0.8')
    for artist in ax.lines:
        artist.set_zorder(10)
    for artist in ax.findobj(PathCollection):
        artist.set_zorder(11)
    sns.stripplot(data=df, x='method', y='rank', jitter=True, alpha=0.6)
    plt.savefig("../tex/Figs/violins_gcn.pdf", format="pdf")
    plt.show()

if __name__ == "__main__":
    ablation_results()
