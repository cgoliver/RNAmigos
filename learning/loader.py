import pickle
import sys
import os
from collections import Counter
import itertools

if __name__ == '__main__':
    sys.path.append('../')

import rnaglib
import networkx as nx
from tqdm import tqdm
import dgl
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset

from data_processor.node_sim import SimFunctionNode, k_block_list

class V1(Dataset):
    def __init__(self, sim_function="R_1",
                    annotated_path='../data/annotated/pockets_nx_symmetric',
                    edge_map=None,
                    get_sim_mat=False,
                    nucs=True,
                    depth=3,
                    shuffle=False,
                    pocket_only=False,
                    seed=0,
                    clustered=False,
                    num_clusters=8):
        """
            Setup for data loader.

            Arguments:
                sim_function (str): which node kernel to use (default='R1', see `node_sim.py`).
                annotated_path (str): path to annotated graphs (see `annotator.py`).
                get_sim_mat (bool): whether to compute a node similarity matrix (deault=True).
                nucs (bool): whether to include nucleotide ID in node (default=False).
                depth (int): number of hops to use in the node kernel.
        """
        print(f">>> fetching data from {annotated_path}")
        self.path = annotated_path
        self.all_graphs = sorted(os.listdir(annotated_path))
        self.pocket_only = pocket_only
        if seed:
            print(f">>> shuffling with random seed {seed}")
            np.random.seed(seed)
        if shuffle:
            np.random.shuffle(self.all_graphs)
        #build edge map
        if edge_map is None:
            self.edge_map = rnaglib.config.graph_keys.EDGE_MAP_FR3D
        else:
            self.edge_map = edge_map
        self.num_edge_types = len(self.edge_map)
        self.edge_freqs = rnaglib.config.graph_keys.IDF
        self.nucs = nucs
        self.clustered = clustered
        self.num_clusters = num_clusters
        if nucs:
            print(">>> storing nucleotide IDs")
            self.nuc_map = {n:i for i,n in enumerate(['A', 'C', 'G', 'N', 'U'])}

        print(f">>> found {self.num_edge_types} edge types, frequencies: {self.edge_freqs}")

        self.node_sim_func = SimFunctionNode(method=sim_function, IDF=rnaglib.config.graph_keys.IDF, depth=depth)

        self.n = len(self.all_graphs)

        self.get_sim_mat = get_sim_mat


    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        """
            Returns one training item at index `idx`.
        """
        if self.pocket_only:
            graph = pickle.load(open(os.path.join(self.path, self.all_graphs[idx]), 'rb'))
        else:
            graph, _, ring, fp = pickle.load(open(os.path.join(self.path, self.all_graphs[idx]), 'rb'))


        kill_edges = []
        relabel_edges = {}
        for u, v, d in graph.edges(data=True):
            elabel = d['LW']
            if elabel == 'B35':
                kill_edges.append((u, v))    
                continue
            if elabel == 'B53':
                continue

            relabel_edges[(u, v)] = str(elabel[0] + "".join(sorted(elabel[1:]))).upper()


        graph.remove_edges_from(kill_edges)
        nx.set_edge_attributes(graph, name='LW', values=relabel_edges)

        #adding the self edges
        # graph.add_edges_from([(n, n, {'label': 'X'}) for n in graph.nodes()])
        # graph = nx.to_undirected(graph)
        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, 'LW')).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

        node_attrs = None
        if self.nucs:
            one_hot_nucs  = {node: torch.tensor(self.nuc_map[label], dtype=torch.float32) for node, label in
                       (nx.get_node_attributes(graph, 'nt')).items()}
        else:
            one_hot_nucs  = {node: torch.tensor(0, dtype=torch.float32) for node in
                       graph.nodes()}

        nx.set_node_attributes(graph, name='one_hot', values=one_hot_nucs)
        g_dgl = dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'], node_attrs=['one_hot'])
        g_dgl.title = self.all_graphs[idx]

        if self.pocket_only:
            return g_dgl

        if self.clustered:
            one_hot_label = torch.zeros((1,self.num_clusters))
            one_hot_label[fp] = 1.
            fp = one_hot_label

        if self.get_sim_mat:
            # put the rings in same order as the dgl graph
            ring = dict(sorted(ring.items()))
            return g_dgl, ring, fp, [idx]

        else:
            return g_dgl, fp, [idx]

def collate_wrapper(node_sim_func, pocket_only=False, get_sim_mat=True):
    """
        Wrapper for collate function so we can use different node similarities.
    """
    if get_sim_mat:
        def collate_block(samples):
            # The input `samples` is a list of pairs
            #  (graph, label).
            # print(len(samples))
            graphs, rings, fp, idx  = map(list, zip(*samples))
            fp = np.array(fp)
            idx = np.array(idx)
            batched_graph = dgl.batch(graphs)
            K = k_block_list(rings, node_sim_func)
            return batched_graph, torch.from_numpy(K).float(), torch.from_numpy(fp).float(), torch.from_numpy(idx)
    if pocket_only:
        def collate_block(samples):
            return dgl.batch(samples)
    else:
        def collate_block(samples):
            # The input `samples` is a list of pairs
            #  (graph, label).
            # print(len(samples))
            graphs, fp, idx = map(list, zip(*samples))
            fp = np.array(fp)
            idx = np.array(idx)
            batched_graph = dgl.batch(graphs)
            return batched_graph, [1 for _ in samples], torch.from_numpy(fp), torch.from_numpy(idx)
    return collate_block

class Loader():
    def __init__(self,
                 annotated_path='../data/annotated/pockets_nx_symmetric/',
                 edge_map=None,
                 batch_size=128,
                 num_workers=20,
                 sim_function="R_1",
                 shuffle=False,
                 seed=0,
                 get_sim_mat=False,
                 nucs=True,
                 depth=3,
                 pocket_only=False):
        """
        Wrapper class to call with all arguments and that returns appropriate data_loaders
        :param pocket_path:
        :param ligand_path:
        :param batch_size:
        :param num_workers:
        :param augment_flips: perform numpy flips
        :param ram: store whole thing in RAM
        :param siamese: for the batch siamese technique
        :param full_siamese for the true siamese one
        """
        self.all_graphs = sorted(os.listdir(annotated_path))
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = V1(annotated_path=annotated_path,
                          sim_function=sim_function,
                          get_sim_mat=get_sim_mat,
                          edge_map=edge_map,
                          shuffle=shuffle,
                          seed=seed,
                          nucs=nucs,
                          depth=depth,
                          pocket_only=pocket_only)

        self.num_edge_types = self.dataset.num_edge_types

    def get_data(self, k_fold=0):
        n = len(self.dataset)
        indices = list(range(n))
        collate_block = collate_wrapper(self.dataset.node_sim_func, self.dataset.get_sim_mat, pocket_only=pocket_only)

        if k_fold > 1:
            from sklearn.model_selection import KFold
            kf = KFold(n_splits=k_fold)
            for train_indices, test_indices in kf.split(np.array(indices), np.array(indices)):
                train_set = Subset(self.dataset, train_indices)
                test_set = Subset(self.dataset, test_indices)

                train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                          num_workers=self.num_workers, collate_fn=collate_block)
                test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                         num_workers=self.num_workers, collate_fn=collate_block)

                yield train_loader, test_loader


        else:
            split_train, split_valid = 0.8, 0.8
            train_index, valid_index = int(split_train * n), int(split_valid * n)

            train_indices = indices[:train_index]
            test_indices = indices[train_index:]

            train_set = Subset(self.dataset, train_indices)
            test_set = Subset(self.dataset, test_indices)

            print("training graphs ", len(train_set))
            print("testing graphs ", len(test_set))

            train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                      num_workers=self.num_workers, collate_fn=collate_block)
            test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block)

            # return train_loader, valid_loader, test_loader
            yield train_loader, test_loader
        #full loader


class InferenceLoader(Loader):
    def __init__(self,
                 annotated_path,
                 batch_size=5,
                 num_workers=20,
                 edge_map=None,
                 pocket_only=False):
        super().__init__(
            annotated_path=annotated_path,
            batch_size=batch_size,
            num_workers=num_workers,
            edge_map=edge_map,
            pocket_only=pocket_only)
        self.pocket_only = pocket_only
        self.dataset.all_graphs = sorted(os.listdir(annotated_path))

    def get_data(self):
        collate_block = collate_wrapper(None, get_sim_mat=False, pocket_only=self.pocket_only)
        train_loader = DataLoader(dataset=self.dataset,
                                  shuffle=False,
                                  batch_size=self.batch_size,
                                  num_workers=self.num_workers,
                                  collate_fn=collate_block)
        return train_loader
if __name__ == '__main__':
    loader = Loader(shuffle=False,seed=99, batch_size=1, num_workers=1)
    data = loader.get_data(k_fold=5)
    for train, test in data:
        print(len(train), len(test))

    pass
