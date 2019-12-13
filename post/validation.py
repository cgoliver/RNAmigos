"""
    Do embedding clustering and generate similar subgraphs.
"""
import sys
if __name__ == '__main__':
    sys.path.append('../')

import os
import pickle
import math
import random
from random import choice, shuffle
from tqdm import tqdm
import networkx as nx
import numpy as np
from numpy.linalg import norm
from scipy.spatial.distance import jaccard,euclidean
import torch
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.collections import PathCollection
import seaborn as sns
import pandas as pd
from Bio.PDB import MMCIFParser, NeighborSearch

from learning.rgcn import Model
from rna_classes import *
from post.utils import *

from learning.attn import get_attention_map
from learning.utils import dgl_to_nx
from post.drawing import rna_draw

def load_model(run, graph_dir):
    dims = [32] * 3
    # dims = [32]*6
    attributor_dims = [32, 166]

    edge_map = get_edge_map(graph_dir)

    model = Model(dims=dims, attributor_dims=attributor_dims, num_rels=len(edge_map), num_bases=-1)
    model.load_state_dict(torch.load(f'../trained_models/{run}/{run}.pth', map_location='cpu')['model_state_dict'])


    return model, edge_map, dims[-1]

def get_decoys(mode='pdb', annots_dir='../data/annotated/pockets_nx_2'):
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
    if mode == 'dude':
        return pickle.load(open('../data/decoys_zinc.p', 'rb'))
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

def decoy_test(model, decoys, edge_map, embed_dim, test_graphlist=None, shuffle=False, test_graph_path="../data/annotated/pockets_nx"):
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
    sims = []

    if test_graphlist is None:
        test_graphlist = os.listdir(test_graph_path)
        
    ligs = list(decoys.keys())
    for g_path in test_graphlist:
        g,_,_,_ = pickle.load(open(os.path.join(test_graph_path, g_path), 'rb'))
        try:
            true_id = g_path.split(":")[2]
        except:
            print(f">> failed on {g_path}")
            continue
        try:
            decoys[true_id]
        except KeyError:
            print("missing fp", true_id)
            continue
        nx_graph, dgl_graph = nx_to_dgl(g, edge_map, embed_dim)
        _,fp_pred= model(dgl_graph)

        if False:
                n_nodes = len(dgl_graph.nodes)
                att= get_attention_map(dgl_graph, src_nodes=dgl_graph.nodes(), dst_nodes=dgl_graph.nodes(), h=1)
                att_g0 = att[0] # get attn weights only for g0
                
                # Select atoms with highest attention weights and plot them 
                tops = np.unique(np.where(att_g0>0.51)) # get top atoms in attention
                print(f"tops {tops}")

                g0 = dgl_to_nx(dgl_graph, edge_map)
                nodelist = list(g0.nodes())
                highlight_edges = list(g0.subgraph([nodelist[t] for t in tops]).edges())
                rna_draw(g0, highlight_edges=highlight_edges)

        fp_pred = fp_pred.detach().numpy() > 0.5
        print(fp_pred)
        # print(fp_pred)
        # fp_pred = np.random.choice([0, 1], size=(166,), p=[1./2, 1./2])
        if shuffle:
            true_id = choice(ligs)
        active = decoys[true_id][0]
        decs = decoys[true_id][1]
        rank = distance_rank(active, fp_pred, decs)
        sim = jaccard(active, fp_pred)
        ranks.append(rank)
        sims.append(sim)
    return ranks, sims

def generic_fp(annot_dir):
    """
        Compute generic fingerprint by majority over dimensions.
        TODO: Finish this
    """
    fps = []
    for g in os.listdir(annot_dir):
        _,_,fp,_ = pickle.load(open(os.path.join(annot_dir, g), 'rb'))
        fps.append(fp)
    consensus = np.unique(fps, axis=0)
    pass
    
def ablation_results():
    modes = ['', '_bb-only', '_wc-bb', '_wc-bb-nc', '_no-label', '_label-shuffle', 'pair-shuffle']
    modes = ['', 'pair-shuffle']
    decoys = get_decoys(mode='pdb')
    ranks, methods = [], []
    graph_dir = '../data/annotated/pockets_nx_2'
    graph_dir = '../data/annotated/pockets_nx'
    run = "pockets_1_noatt_mean"
    for m in modes:

        # if m in ['', 'pair-shuffle']:
            # graph_dir = "../data/annotated/pockets_nx"
            # run = 'small_no_rec_2'
        # else:
            # graph_dir = "../data/annotated/pockets_nx" + m
            # run = 'small_no_rec' + m


        model,edge_map,embed_dim = load_model(run, graph_dir)
        num_edge_types = len(edge_map)

        graph_ids = pickle.load(open(f'../results/{run}/splits.p', 'rb'))

        shuffle = False
        if m == 'pair-shuffle':
            shuffle = True
        ranks_this,sims_this  = decoy_test(model, decoys, edge_map, embed_dim, shuffle=shuffle, test_graphlist=graph_ids['test'], test_graph_path=graph_dir)
        test_ligs = []
        ranks.extend(ranks_this)
        methods.extend([m]*len(ranks_this))

        #decoy distance distribution
        dists = []
        for _,(active, decs) in decoys.items():
            for d in decs:
                dists.append(jaccard(active, d))
        plt.scatter(ranks_this, sims_this)
        plt.xlabel("ranks")
        plt.ylabel("distance")
        plt.show()
        sns.distplot(dists, label='decoy distance')
        sns.distplot(sims_this, label='pred distance')
        plt.xlabel("distance")
        plt.legend()
        plt.show()

        rank_cut = 0.9
        cool = [graph_ids['test'][i] for i,(d,r) in enumerate(zip(sims_this, ranks_this)) if d <0.4 and r > 0.8]
        print(cool, len(ranks_this))
        #4v6q_#0:BB:FME:3001.nx_annot.p
        test_ligs = set([f.split(":")[2] for f in graph_ids['test']])
        train_ligs = set([f.split(":")[2] for f in graph_ids['train']])
        print(test_ligs - train_ligs)
        points = []
        tot = len([x for x in ranks_this if x >= rank_cut])
        for sim_cut in np.arange(0,1.1,0.1):
            pos = 0
            for s,r in zip(sims_this, ranks_this):
                if s < sim_cut and r > rank_cut:
                    pos += 1
            points.append(pos / tot)
        from sklearn.metrics import auc
        plt.title(f"Top 20% Accuracy {auc(np.arange(0, 1.1, 0.1), points)}, {m}")
        plt.plot(points, label=m)
        plt.plot([x for x in np.arange(0,1.1, 0.1)], '--')
        plt.ylabel("Positives")
        plt.xlabel("Distance threshold")
        plt.xticks(np.arange(10), [0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.9, 1.0])
        plt.legend()
        plt.show()

    df = pd.DataFrame({'rank': ranks, 'method':methods})
    ax = sns.violinplot(x="method", y="rank", data=df, color='0.8')
    for artist in ax.lines:
        artist.set_zorder(10)
    for artist in ax.findobj(PathCollection):
        artist.set_zorder(11)
    sns.stripplot(data=df, x='method', y='rank', jitter=True, alpha=0.6)
    # plt.savefig("../tex/Figs/violins_gcn_2.pdf", format="pdf")
    plt.show()

def structure_scanning(pdb, ligname, graph, model, edge_map, embed_dim):
    """
        Given a PDB structure make a prediction for each residue in the structure:
            - chop the structure into candidate sites (for each residue get a sphere..)
            - convert residue neighbourhood into graph
            - get prediction from model for each
            - compare prediction to native ligand.
        :returns: `residue_preds` dictionary with residue id as key and fingerprint prediction as value.
    """
    from data_processor.build_dataset import get_pocket_graph

    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("", pdb)[0]

    residue_preds = {}
    residues = list(structure.get_residues())
    for residue in tqdm(residues):
        if residue.resname in ['A', 'U', 'C', 'G', ligname]:
            res_info = ":".join(["_",residue.get_parent().id, residue.resname, str(residue.id[1])])
            pocket_graph = get_pocket_graph(pdb, res_info, graph)
            _,dgl_graph = nx_to_dgl(pocket_graph, edge_map, embed_dim)
            _,fp_pred= model(dgl_graph)
            fp_pred = fp_pred.detach().numpy() > 0.5
            residue_preds[(residue.get_parent().id, residue.id[1])] = fp_pred
        else:
            continue
    return residue_preds

def scanning_analyze():
    """
        Visualize results of scanning on PDB.
        Color residues by prediction score.
          1fmn_#0.1:A:FMN:36.nx_annot.p
    """
    from data_processor.build_dataset import find_residue,lig_center

    model, edge_map, embed_dim  = load_model('small_no_rec_2', '../data/annotated/pockets_nx')
    for f in os.listdir("../data/annotated/pockets_nx"):
        pdbid = f.split("_")[0]
        _,chain,ligname,pos = f.replace(".nx_annot.p", "").split(":")
        pos = int(pos)
        print(chain,ligname, pos)
        graph = pickle.load(open(f'../data/RNA_Graphs/{pdbid}.pickle', 'rb'))
        if len(graph.nodes()) > 100:
            continue
        try:
            fp_preds = structure_scanning(f'../data/all_rna_prot_lig_2019/{pdbid}.cif', ligname, graph, model, edge_map, embed_dim)
        except Exception as e:
            print(e)
            continue
        parser = MMCIFParser(QUIET=True)
        structure = parser.get_structure("", f"../data/all_rna_prot_lig_2019/{pdbid}.cif")[0]
        lig_res = find_residue(structure[chain], pos)
        lig_c = lig_center(lig_res.get_atoms())

        fp_dict = pickle.load(open("../data/all_ligs_maccs.p", 'rb'))
        true_fp = fp_dict[ligname]
        dists = []
        jaccards = []
        decoys = get_decoys()
        for res, fp in fp_preds.items():
            chain, pos = res
            r = find_residue(structure[chain], pos)
            r_center = lig_center(r.get_atoms())
            dists.append(euclidean(r_center, lig_c))
            jaccards.append(jaccard(true_fp, fp))
        plt.title(f)
        plt.distplot(dists, jaccards)
        plt.xlabel("dist to binding site")
        plt.ylabel("dist to fp")
        plt.show()
    pass

if __name__ == "__main__":
    # scanning_analyze()
    ablation_results()
