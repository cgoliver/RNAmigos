"""
    Build graph dataset for learning.
"""

import sys
if __name__ == "__main__":
    sys.path.append("..")

import pickle
import itertools

import numpy as np
import networkx as nx
from Bio.PDB import *

from tools.rna_draw import *
from rna_classes import *
from graph_process import *

def lig_center(lig_atoms):
    return np.mean(np.array([a.coord for a in lig_atoms]), axis=0)

def find_residue(chain, pos):
    for residue in chain:
        if residue.id[1] == pos:
            return residue
    return None


def get_pocket_graph(pdb_structure_path, ligand_id, graph, dump_path="../data/pockets_nx", cutoff=10):
    parser = MMCIFParser(QUIET=True)
    pdbid = os.path.basename(pdb_structure_path).split(".")[0]
    structure = parser.get_structure("", pdb_structure_path)[0]

    chain,resname, pos = ligand_id.split(":")[1:]

    lig_res_atoms = find_residue(structure[chain], int(pos)).get_atoms()
    lig_coord = lig_center(lig_res_atoms)

    #get atoms within radius
    kd = NeighborSearch(list(structure.get_atoms()))
    pocket = kd.search(lig_coord, cutoff, level='R')

    pocket_nodes = []
    for r in pocket:
        node = find_node(graph, r.get_parent().id, r.id[1])
        if node is not None:
            pocket_nodes.append(node)
    expand = bfs_expand(graph, pocket_nodes, depth=1)
    pocket_graph = graph.subgraph(expand).copy()
    remove_self_loops(pocket_graph)
    to_orig(pocket_graph)

    pickle.dump(pocket_graph, open(os.path.join(dump_path, f"{pdbid}_{ligand_id}.p"), 'wb'))

    pass

def ligand_binding(lig_dict_path, fp_dict_path):
    """
        Get binding site graphs for each ligand in lig_dict.
        Label is corresponding fingerprint in fp_dict.
    """

    lig_dict = pickle.load(open(lig_dict_path, 'rb'))
    fp_dict = pickle.load(open(fp_dict_path, 'rb'))

    failed =0

    for pdbid, ligs in lig_dict.items():
        pdbid =  pdbid.split(".")[0]
        try:
            pdb_graph = pickle.load(open(f'../data/RNA_Graphs/{pdbid}.pickle', 'rb'))
            for lig in ligs:
                get_pocket_graph(f'../data/all_rna_prot_lig_2019/{pdbid}.cif', lig, pdb_graph)
        except FileNotFoundError:
            print(f"{pdbid}")
            failed += 1
    print(failed)

if __name__ == "__main__":
    ligand_binding('../data/lig_dict_c_10A_08rna.p','../data/lig_dict.p')
    pass
