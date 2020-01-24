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
from tools.learning_utils import load_model
# from post.drawing import rna_draw

def mse(x,y):
    d = np.sum((x-y)**2) / len(x)
    return d

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
        decoy_dict = {k:(v, [f for lig,f in fp_dict.items() if lig != k]) for k,v in fp_dict.items()}
        return decoy_dict
    if mode == 'dude':
        return pickle.load(open('../data/decoys_zinc.p', 'rb'))
    pass
def distance_rank(active, pred, decoys, dist_func=mse):
    """
        Get rank of prediction in `decoys` given a known active ligand.
    """

    pred_dist = dist_func(active, pred)
    rank = 0
    for decoy in decoys:
        d = dist_func(pred, decoy)
        #if find a decoy closer to prediction, worsen the rank.
        if d < pred_dist:
            rank += 1
    return 1 - (rank / (len(decoys) + 1))

def decoy_test(model, decoys, edge_map, embed_dim,
                        test_graphlist=None,
                        shuffle=False,
                        nucs=False,
                        test_graph_path="../data/annotated/pockets_nx",
                        majority=False):
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
    if majority:
        generic = generic_fp("../data/annotated/pockets_nx_symmetric_orig")
    for g_path in test_graphlist:
        g,_,_,true_fp = pickle.load(open(os.path.join(test_graph_path, g_path), 'rb'))
        try:
            true_id = g_path.split(":")[2]
        except:
            print(f">> failed on {g_path}")
            continue
        nx_graph, dgl_graph = nx_to_dgl(g, edge_map, nucs=nucs)
        with torch.no_grad():
            fp_pred, _ = model(dgl_graph)

        fp_pred = fp_pred.detach().numpy() > 0.5
        fp_pred = fp_pred.astype(int)
        if majority:
            fp_pred = generic
        # fp_pred = fp_pred.detach().numpy()
        active = decoys[true_id][0]
        decs = decoys[true_id][1]
        rank = distance_rank(active, fp_pred, decs, dist_func=mse)
        sim = jaccard(true_fp, fp_pred)
        ranks.append(rank)
        sims.append(sim)
    return ranks, sims

def wilcoxon_all_pairs(df):
    """
        Compute pairwise wilcoxon on all runs.
    """
    from scipy.stats import wilcoxon
    wilcoxons = {'method_1': [], 'method_2':[], 'p-value': []}
    for method_1, df1 in df.groupby('method'):
        for method_2, df2 in df.groupby('method'):
            p_val = wilcoxon(df1['rank'], df2['rank'])

            wilcoxons['method_1'].append(method_1)
            wilcoxons['method_2'].append(method_2)
            wilcoxons['p-value'].append(p_val[1])
            pass
    wil_df = pd.DataFrame(wilcoxons)
    wil_df.fillna(0)
    pvals = wil_df.pivot("method_1", "method_2", "p-value")
    pvals.fillna(0)
    mask = np.zeros_like(pvals)
    mask[np.triu_indices_from(mask)] = True
    g = sns.heatmap(pvals, cmap="Reds_r", annot=True, mask=mask, cbar=True)
    g.set_facecolor('grey')
    plt.show()
    pass
def generic_fp(annot_dir):
    """
        Compute generic fingerprint by majority over dimensions.
        TODO: Finish this
    """
    fps = []
    for g in os.listdir(annot_dir):
        _,_,_,fp = pickle.load(open(os.path.join(annot_dir, g), 'rb'))
        fps.append(fp)
    counts = np.sum(fps, axis=0)
    consensus = np.zeros(166)
    ones = counts > len(fps) / 2
    consensus[ones] = 1
    return consensus
    
def make_violins(df, x='method', y='rank', save=None, show=True):
    ax = sns.violinplot(x=x, y=y, data=df, color='0.8', bw=.1)
    for artist in ax.lines:
        artist.set_zorder(10)
    for artist in ax.findobj(PathCollection):
        artist.set_zorder(11)
    sns.stripplot(data=df, x=x, y=y, jitter=True, alpha=0.6)
    if not save is None:
        plt.savefig(save, format="pdf")
    if show:
        plt.show()

    pass

def make_ridge(df, x='method', y='rank', save=None, show=True):
    # Initialize the FacetGrid object
    sns.set(style="white", rc={"axes.facecolor": (0, 0, 0, 0)})
    pal = sns.cubehelix_palette(10, rot=-.25, light=.7)
    g = sns.FacetGrid(df, row=x, hue=x, aspect=15, height=.5, palette=pal)

    # Draw the densities in a few steps
    g.map(sns.kdeplot, y, clip_on=False, shade=True, alpha=1, lw=1.5, bw=.2)
    g.map(sns.kdeplot, y, clip_on=False, color="w", lw=2, bw=.2)
    g.map(plt.axhline, y=0, lw=2, clip_on=False)


    # Define and use a simple function to label the plot in axes coordinates
    def label(x, color, label):
        ax = plt.gca()
        ax.text(0, .2, label, fontweight="bold", color=color,
                ha="left", va="center", transform=ax.transAxes)
    g.map(label, x)

    # Set the subplots to overlap
    g.fig.subplots_adjust(hspace=-.25)

    # Remove axes details that don't play well with overlap
    g.set_titles("")
    g.set(yticks=[])
    g.despine(bottom=True, left=True)
    plt.show()

def ablation_results():
    # modes = h'', '_bb-only', '_wc-bb', '_wc-bb-nc', '_no-label', '_label-shuffle', 'pair-shuffle']
    # modes = ['raw', 'bb', 'wc-bb', 'pair-shuffle']
    # modes = ['raw', 'warm', 'wc-bb', 'bb', 'majority', 'swap', 'random']
    modes = ['raw', 'wc-bb', 'bb', 'majority', 'swap', 'random']
    decoys = get_decoys(mode='pdb')
    ranks, methods, jaccards  = [], [], []
    graph_dir = '../data/annotated/pockets_nx_symmetric_orig'
    # graph_dir = '../data/annotated/pockets_nx_2'
    run = 'ismb'
    num_folds = 10 
    majority = False
    for m in modes:
        print(m)

        if m in ['raw', 'pair-shuffle']:
            graph_dir = "../data/annotated/pockets_nx_symmetric_orig"
            run = 'ismb-raw'
        elif m == 'swap':
            graph_dir = '../data/annotated/pockets_nx_symmetric_scramble_orig'
            run = 'ismb-' + m
        elif m == 'majority':
            run = 'ismb-raw'
            majority = True
        elif m == 'random':
            graph_dir = '../data/annotated/pockets_nx_symmetric_random_orig'
            run = 'random'
        elif m == 'warm':
            graph_dir = '../data/annotated/pockets_nx_symmetric_orig'
            run  = 'ismb-warm'
        else:
            graph_dir = "../data/annotated/pockets_nx_symmetric_" + m + "_orig"
            run = 'ismb-' + m


        for fold in range(num_folds):
            model, meta = load_model(run +"_" + str(fold))
            # model, meta = load_model(run)
            edge_map = meta['edge_map']
            embed_dim = meta['embedding_dims'][-1]
            num_edge_types = len(edge_map)

            graph_ids = pickle.load(open(f'../results/trained_models/{run}_{fold}/splits_{fold}.p', 'rb'))
            # graph_ids = pickle.load(open(f'../results/trained_models/{run}/splits.p', 'rb'))

            ranks_this,sims_this  = decoy_test(model, decoys, edge_map, embed_dim,
                shuffle=shuffle,
                nucs=meta['nucs'],
                test_graphlist=graph_ids['test'],
                test_graph_path=graph_dir,
                majority=majority)
            test_ligs = []
            ranks.extend(ranks_this)
            jaccards.extend(sims_this)
            methods.extend([m]*len(ranks_this))

            # decoy distance distribution
            # dists = []
            # for _,(active, decs) in decoys.items():
                # for d in decs:
                    # dists.append(jaccard(active, d))
            # plt.scatter(ranks_this, sims_this)
            # plt.xlabel("ranks")
            # plt.ylabel("distance")
            # plt.show()
            # sns.distplot(dists, label='decoy distance')
            # sns.distplot(sims_this, label='pred distance')
            # plt.xlabel("distance")
            # plt.legend()
            # plt.show()

            # # rank_cut = 0.9
            # cool = [graph_ids['test'][i] for i,(d,r) in enumerate(zip(sims_this, ranks_this)) if d <0.4 and r > 0.8]
            # cool = [graph_ids['test'][i] for i,(d,r) in enumerate(zip(sims_this, ranks_this)) if d <0.3]
            # print(cool)
            # print(cool, len(ranks_this))
            #4v6q_#0:BB:FME:3001.nx_annot.p
            # test_ligs = set([f.split(":")[2] for f in graph_ids['test']])
            # train_ligs = set([f.split(":")[2] for f in graph_ids['train']])
            # print("ligands not in train set", test_ligs - train_ligs)
            # points = []
            # tot = len([x for x in ranks_this if x >= rank_cut])
            # for sim_cut in np.arange(0,1.1,0.1):
                # pos = 0
                # for s,r in zip(sims_this, ranks_this):
                    # if s < sim_cut and r > rank_cut:
                        # pos += 1
                # points.append(pos / tot)
            # from sklearn.metrics import auc
            # plt.title(f"Top 20% Accuracy {auc(np.arange(0, 1.1, 0.1), points)}, {m}")
            # plt.plot(points, label=m)
            # plt.plot([x for x in np.arange(0,1.1, 0.1)], '--')
            # plt.ylabel("Positives")
            # plt.xlabel("Distance threshold")
            # plt.xticks(np.arange(10), [0, 0.1, 0.2, 0.3, 0.4, 0.5,0.6, 0.7, 0.9, 1.0])
            # plt.legend()
            # plt.show()
    df = pd.DataFrame({'rank': ranks, 'jaccard': jaccards, 'method':methods})
    wilcoxon_all_pairs(df)
    # make_ridge(df, x='method', y='rank')
    # make_ridge(df, x='method', y='jaccard')
    # make_violins(df, x='method', y='jaccard')
    # make_violins(df, x='method', y='rank')

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
            jaccards.append(mse(true_fp, fp))
        plt.title(f)
        plt.distplot(dists, jaccards)
        plt.xlabel("dist to binding site")
        plt.ylabel("dist to fp")
        plt.show()
    pass

if __name__ == "__main__":
    # scanning_analyze()
    ablation_results()
