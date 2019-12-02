"""
    Build graph dataset for learning and for annotation later.
"""

import sys
if __name__ == "__main__":
    sys.path.append("..") 
import pickle
import itertools
import subprocess

import numpy as np
import networkx as nx
from Bio.PDB import *
from tqdm import tqdm

from tools.rna_draw import *
from post.drawing import rna_draw
from pocket_grid import sample_non_binding_sites
from rna_classes import *
from graph_process import *
from marker_file import *

faces = ['W', 'S', 'H']
orientations = ['C', 'T']
valid_edges = set(['B53'] + [orient + e1 + e2 for e1, e2 in itertools.product(faces, faces) for orient in orientations])

def lig_center(lig_atoms):
    return np.mean(np.array([a.coord for a in lig_atoms]), axis=0)

def find_residue(chain, pos):
    for residue in chain:
        if residue.id[1] == pos:
            return residue
    return None

def graph_from_residues(full_graph, residues):
    """
        Build an NX graph from list of biopython residues.
    """
    pocket_nodes = []
    for r in residues:
        node = find_node(full_graph, r.get_parent().id, r.id[1])
        if node is not None:
            pocket_nodes.append(node)
    expand = bfs_expand(full_graph, pocket_nodes, depth=1)
    pocket_graph = full_graph.subgraph(expand).copy()
    G = to_orig(pocket_graph)
    kill_islands(G)
    G = dangle_trim(G)

    return G

def get_pocket_graph(pdb_structure_path, ligand_id, graph, 
        ablate=None, dump_path="../data/pockets_nx", cutoff=8,
        non_binding=False):
    """
        Main function for extracting a graph from a binding site.

    """
    parser = MMCIFParser(QUIET=True)
    pdbid = os.path.basename(pdb_structure_path).split(".")[0]
    structure = parser.get_structure("", pdb_structure_path)[0]

    chain,resname, pos = ligand_id.split(":")[1:]

    lig_residue = find_residue(structure[chain], int(pos))
    lig_res_atoms = lig_residue.get_atoms()
    lig_coord = lig_center(lig_res_atoms)

    #get atoms within radius
    kd = NeighborSearch(list(structure.get_atoms()))
    pocket = kd.search(lig_coord, cutoff, level='R')


    G = graph_from_residues(graph, pocket)
    #visualize on 3D structure
    # pdb_to_markers_(structure, G, "markers.cmm")
    # subprocess.call(['chimera', pdb_structure_path, 'markers.cmm'])

    # os.remove("markers.cmm")

    labels = {d['label'] for _,_,d in G.edges(data=True)}

    assert labels.issubset(valid_edges)

    # rna_draw(G, title="BINDING")

    # if dump_path and (len(G.nodes()) > 4):
        # nx.write_gpickle(G, os.path.join(dump_path, f"{pdbid}_{ligand_id}.nx"))

    #sample and build non-binding graph.
    if non_binding:
        for pocket in sample_non_binding_sites(pdb_structure_path, lig_residue):
            if pocket:
                non_bind_graph = graph_from_residues(graph, pocket)
                rna_draw(non_bind_graph, title="NON BINDING")
                pdb_to_markers_(structure, non_bind_graph, "markers.cmm")
                subprocess.call(['chimera', pdb_structure_path, 'markers.cmm'])
                os.remove("markers.cmm")
            else:
                pass
            pass
    return G

def get_binding_site_graphs_all(lig_dict_path, dump_path, non_binding=False):
    """
        Get binding site graphs for each ligand in lig_dict.
    """

    lig_dict = pickle.load(open(lig_dict_path, 'rb'))

    failed = 0

    try:
        os.mkdir(dump_path)
    except:
        pass

    done_pdbs = {f.split('_')[0] for f in os.listdir(dump_path)}

    for pdbid, ligs in tqdm(lig_dict.items()):
        pdbid =  pdbid.split(".")[0]
        if pdbid in done_pdbs:
            pass
            # continue
        # try:
        print(">>> ", pdbid)
        pdb_graph = pickle.load(open(f'../data/RNA_Graphs/{pdbid}.pickle', 'rb'))
        # print(f"new guy: {pdbid}")
        # continue
        for lig in ligs:
            pdb_path = f"../data/all_rna_prot_lig_2019/{pdbid}.cif"
            #dump binding site graphs
            get_pocket_graph(pdb_path, lig,
                            pdb_graph, dump_path=dump_path,
                            non_binding=non_binding)
        # except FileNotFoundError:
            # print(f"{pdbid} not found")
            # failed += 1
    # print(failed)


if __name__ == "__main__":
    #take all ligands with 8 angstrom sphere and 0.6 RNA concentration, build a graph for each.
    get_binding_site_graphs_all('../data/lig_dict_c_8A_06rna.p','../data/pockets_nx_2',
                                non_binding=True)
    # get_binding_site_graphs_all('../data/lig_dict_c_8A_06rna.p','')
    pass
