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
from scipy.spatial.dist import jaccard
import torch

from learning.rgcn import Model

def distance_rank(active, pred, decoys, dist_func=jaccard):
    """
        Get rank of prediction in `decoys` given a known active ligand.
    """

    pred_dist = distance(active, pred)
    rank = 0
    for lig in ligands:
        d = distance(active, lig)
        if d < pred_dist:
            rank += 1
    return rank / (len(decoys) + 1)

def decoy_test(model, test_set, decoys):
    """
        Check performance against decoy set.
        decoys --> {'ligand_id', ('expected_FP', [decoy_fps])}
        test_set --> [(rna_graph_dgl, native_ligand_id)]
        :model trained model
        :test_set inputs for model to test (RNA graphs)
        :decoys dictionary with list of decoys for each input to test.

        :return: enrichment score
    """
    ranks = []
    for g, true_id in test_set:
        out = model(g)
        active = decoys[true_id][0]
        decs = decoys[true_id][1]
        rank = distance_rank(active, out, decs)
        ranks.append(rank)
    return np.mean(ranks)

if __name__ == "__main__":
    graph_dir = "../data/annotated/samples"
    motif_id = 7
    graph_dir = f"../data/annotated/motif_{motif_id}"
    graph_dir = f"../data/annotated/samples"
    graph_dir = "../data/annotated/carnaval_motifs"
    # graph_dir = "../data/annotated/hairpins"
    run = 'carna_5'
    # run = 'hairpins'
    # model = pickle.load(open('model_v1.p', 'rb'))
    # num_edge_types = 19
    k_motif(graph_dir)
    edge_map = get_edge_map(graph_dir)
    num_edge_types = len(edge_map)

    dims = [128] * 6
    dims = [64] * 6
    # dims = [32]*6
    attributor_dims = [128, 64, 32]
    attributor_dims = [64, 32, 16]
    # attributor_dims = [32, 16, 4]

    model = Model(dims=dims, attributor_dims=attributor_dims, num_rels=num_edge_types, motif_norm=False, num_bases=-1)
    model.load_state_dict(torch.load(f'../trained_models/{run}/{run}.pth', map_location='cpu')['model_state_dict'])
