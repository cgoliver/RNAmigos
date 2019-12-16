"""
    Cluster ligands in dictionary to obtain ligand classes for training.
"""

import pickle

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.cluster import AgglomerativeClustering


def ligands_cluster(bs_dict, fp_dict, n_clusters=8):
    """
        Assign cluster labels to each ligand in ligand_list.
    """
    #get which ligands to use in clustering
    binding_sites = pickle.load(open(bs_dict, 'rb'))
    fingerprints = pickle.load(open(fp_dict, 'rb'))
    ligs_2_cluster = []
    for _,ligs in binding_sites.items():
        ligs_2_cluster.extend([f.split(":")[2] for f in ligs])
    ligs_2_cluster = list(set(ligs_2_cluster))

    fps = []
    for l in ligs_2_cluster:
        try:
            fps.append(fingerprints[l])
        except:
            print(l)

    #do the clustering
    clusterer = AgglomerativeClustering(n_clusters=n_clusters)
    clusterer.fit(fps)
    labels = clusterer.labels_
    sns.distplot(labels)
    plt.show()

    pass

if __name__ == "__main__":
    ligands_cluster("../data/lig_dict_c_8A_06rna.p", "../data/all_ligs_maccs.p")
