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
import torch

from learning.rgcn import Model

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
    for g, true_id in test_set:
        out = model(g)
        decs = decoys[true_id][0] + decoys[true_id][1]
    pass

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

    random_attr = dummy_attrib(graph_dir, edge_map, dims[0], n_clusters=16)
    same_attr = dummy_attrib(graph_dir, edge_map, dims[0], n_clusters=16, mode='same')
    # Use the attributing model to build an attribution dict
    # attr_dict_e2e = e2e_motif(graph_dir, model, edge_map, dims[0])
    # print(attr_dict_e2e)
    # attr_draw(attr_dict_e2e, 3, num_motifs=16, varna_draw=False)

    # Use the agglomerative clustering to build the attirubtion dict
    attr_dict_agg = node_cluster(graph_dir, model, edge_map, dims[0], n_clusters=16)
    for m in range(16):
        attr_draw(attr_dict_agg, m, num_motifs=16, varna_draw=False)

    carnaval_nodes = pickle.load(open(f'../data/carnaval_motifs_nodes.p', 'rb'))
    scores = defaultdict(list)

    # get entropy scores
    for i, motif_instance_nodes in enumerate(carnaval_nodes):
        print(i)
        motif_instance_attribs_e2e = {k: v for k, v in attr_dict_e2e.items() if int(k.split("_")[0]) == i}
        motif_instance_attribs_agg = {k: v for k, v in attr_dict_agg.items() if int(k.split("_")[0]) == i}
        motif_instance_attribs_random = {k: v for k, v in random_attr.items() if int(k.split("_")[0]) == i}
        motif_instance_attribs_same = {k: v for k, v in same_attr.items() if int(k.split("_")[0]) == i}

        motif_node_dict = {f"{i}_{j}.nxpickle": nodes for j, nodes in enumerate(motif_instance_nodes)}
        score = carnaval_motif_entropy(
            motif_instance_attribs_e2e,
            motif_node_dict,
            N=16)
        scores['e2e'].append(score)
        print(f"e2e {score}")
        score = carnaval_motif_entropy(
            motif_instance_attribs_agg,
            motif_node_dict,
            N=16)
        scores['agg'].append(score)
        print(f"agg {score}")
        score = carnaval_motif_entropy(
            motif_instance_attribs_random,
            motif_node_dict,
            N=16)
        scores['random'].append(score)
        print(f"random {score}")
        score = carnaval_motif_entropy(
            motif_instance_attribs_same,
            motif_node_dict,
            N=16)
        scores['same'].append(score)
        print(f"same {score}")
        # ((graph, attribs, motif_nodes),..)
        # motif_draw(motif_instance_attribs_e2e, motif_node_dict,
        # show=True)
    sys.exit()
    # plot scores
    for i, val in enumerate(['Intra-Motif_Entropy', 'Jensen-Shannon_Divergence']):
        for method, sc in scores.items():
            plt.plot([s[i] for s in sc], label=method)
        plt.legend()
        plt.xlabel("motif")
        plt.ylabel(val)
        # plt.savefig(f"/Users/carlosgonzalezoliver/Projects/gemini/figs/{val.replace('_','-')}.pdf", format="pdf")
        plt.show()
    sys.exit()
