"""
    Cluster ligands in dictionary to obtain ligand classes for training.
"""

import pickle


def ligands_cluster(bs_dict, fp_dict):
    """
        Assign cluster labels to each ligand in ligand_list.
    """
    #get which ligands to use in clustering
    binding_sites = pickle.load(open(bs_dict, 'rb'))
    fingerprints = pickle.load(open(fp_dict, 'rb'))
    ligs_2_cluster = []
    for _,ligs in binding_sites.items():
        ligs_2_cluster.extend([f.split(":")[2] for f in ligs])
    ligs_2_cluster = set(ligs_2_cluster)


    #do the clustering

    pass

if __name__ == "__main__":
    ligands_cluster("../data/lig_dict_c_8A_06rna.p", "../data/all_ligs_maccs.p")
