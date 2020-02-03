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
from post.tree_grid_vincent import compute_clustering 

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
    true_ids = []
    fp_dict = {}
    for g_path in test_graphlist:
        g,_,_,true_fp = pickle.load(open(os.path.join(test_graph_path, g_path), 'rb'))
        try:
            true_id = g_path.split(":")[2]
            fp_dict[true_id] = true_fp
            decoys[true_id]
        except:
            print(f">> failed on {g_path}")
            continue
        nx_graph, dgl_graph = nx_to_dgl(g, edge_map, nucs=nucs)
        with torch.no_grad():
            fp_pred, _ = model(dgl_graph)

        # fp_pred = fp_pred.detach().numpy()
        fp_pred = fp_pred.detach().numpy() > 0.5
        fp_pred = fp_pred.astype(int)
        if majority:
            fp_pred = generic
        # fp_pred = fp_pred.detach().numpy()
        active = decoys[true_id][0]
        decs = decoys[true_id][1]
        rank = distance_rank(active, fp_pred, decs, dist_func=mse)
        sim = mse(true_fp, fp_pred)
        true_ids.append(true_id)
        ranks.append(rank)
        sims.append(sim)
    return ranks, sims, true_ids, fp_dict

def wilcoxon_all_pairs(df, methods):
    """
        Compute pairwise wilcoxon on all runs.
    """
    from scipy.stats import wilcoxon
    wilcoxons = {'method_1': [], 'method_2':[], 'p-value': []}
    for method_1 in methods:
        for method_2 in methods:
            vals1 = df.loc[df['method'] == method_1]
            vals2 = df.loc[df['method'] == method_2]
            p_val = wilcoxon(vals1['rank'], vals2['rank'], correction=True)

            wilcoxons['method_1'].append(method_1)
            wilcoxons['method_2'].append(method_2)
            wilcoxons['p-value'].append(p_val[1])
            pass
    wil_df = pd.DataFrame(wilcoxons)
    wil_df.fillna(0)
    pvals = wil_df.pivot("method_1", "method_2", "p-value")
    pvals.fillna(0)
    print(pvals.to_latex())
    # mask = np.zeros_like(pvals)
    # mask[np.triu_indices_from(mask)] = True
    # g = sns.heatmap(pvals, cmap="Reds_r", annot=True, mask=mask, cbar=True)
    # g.set_facecolor('grey')
    # plt.show()
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
    if save:
        plt.savefig(save)
    if show:
        plt.show()


def make_tree_grid(df, fp_dict, method='htune'):
    lig_dict = {}
    df_tree = df.loc[df['method'] == method]
    means = df_tree.groupby('lig').mean()
    for row in means.itertuples():
        lig_dict[row.Index] = (fp_dict[row.Index], row.rank)
    compute_clustering(lig_dict)
    pass
def ablation_results(run, graph_dir, mode, decoy_mode='pdb', folds=10):
    """
        Compute decoy and distances for a given run and ablation mode

        Returns:
            DataFrame: decoy results dataframe
    """
    ranks, methods, jaccards, ligs  = [], [], [], []
    graph_dir = '../data/annotated/pockets_nx_symmetric_orig'
    decoys = get_decoys(mode=decoy_mode, annots_dir=graph_dir)
    majority = mode == 'majority'
    fp_dict = {}
    for fold in range(num_folds):
        model, meta = load_model(run +"_" + str(fold))
        # model, meta = load_model(run)
        edge_map = meta['edge_map']
        embed_dim = meta['embedding_dims'][-1]
        num_edge_types = len(edge_map)

        graph_ids = pickle.load(open(f'../results/trained_models/{run}_{fold}/splits_{fold}.p', 'rb'))

        ranks_this,sims_this, lig_ids, fp_dict_this  = decoy_test(model, decoys, edge_map, embed_dim,
            nucs=meta['nucs'],
            test_graphlist=graph_ids['test'],
            test_graph_path=graph_dir,
            majority=majority)
        fp_dict.update(fp_dict_this)
        ranks.extend(ranks_this)
        jaccards.extend(sims_this)
        ligs.extend(lig_ids)
        methods.extend([m]*len(ranks_this))


    df = pd.DataFrame({'rank': ranks, 'jaccard': jaccards, 'method':methods, 'lig': ligs})
    return df

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
