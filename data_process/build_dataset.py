"""
    Build graph dataset for learning.
"""

import pickle

import numpy as np
from Bio.PDB import *

from rna_classes import *

def ligand_filter(lig_id):
    if lig_id.split(":")[1] in ['UNX', 'OHX', 'MPD', 'SO4', 'IRI', 'PG4']:
        return False
    return True

def lig_center(lig_atoms):
    return np.mean(np.array([a.coord for a in lig_atoms]), axis=0)

def find_residue(chain, pos):
    for residue in chain:
        if residue.id[1] == pos:
            return residue
    return None

def find_node(graph, chain, pos):
    for n,d in graph.nodes(data=True):
        if (n[0] == chain) and (d['nucleotide'].pdb_pos == str(pos)):
            return n
    return None

def get_pocket_graph(pdb_structure_path, ligand_id, graph, cutoff=10):
    parser = MMCIFParser(QUIET=True)
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
    pocket_graph = graph.subgraph(pocket_nodes)
    print(pocket_graph.nodes())

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
