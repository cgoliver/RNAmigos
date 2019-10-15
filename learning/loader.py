import pickle
from tqdm import tqdm
import dgl
import os
import networkx as nx
import itertools
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
import seaborn as sns
import matplotlib.pyplot as plt
import sys

if __name__ == '__main__':
    sys.path.append('../')

from data_processor.node_sim import SimFunctionNode, k_block_list

faces = ['W', 'S', 'H']
orientations = ['C', 'T']
edge_types = ['X', 'B53'] + [orient + e1 + e2 for e1, e2 in itertools.product(faces, faces) for orient in orientations]
num_edge_types = len(edge_types)
edge_map = {e: i for i, e in enumerate(edge_types)}

#for aids
edge_types = [0, 1]
edge_map = {0:0, 1:1}


# print(edge_types)

class V1(Dataset):
    def __init__(self, annotated_path='../data/annotated/samples', debug=False, shuffled=False):
        self.path = annotated_path
        self.all_graphs = os.listdir(annotated_path)
        self.node_sim = SimFunctionNode(method='rings', depth=4)
        self.n = len(self.all_graphs)

        #build edge map
        self.edge_map = self._get_edge_map()
        self.num_edge_types = len(self.edge_map)

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        graph, _, ring = pickle.load(open(os.path.join(self.path, self.all_graphs[idx]), 'rb'))

        #adding the self edges
        # graph.add_edges_from([(n, n, {'label': 'X'}) for n in graph.nodes()])
        graph = nx.to_undirected(graph)
        one_hot = {edge: torch.tensor(self.edge_map[label]) for edge, label in
                   (nx.get_edge_attributes(graph, 'label')).items()}
        nx.set_edge_attributes(graph, name='one_hot', values=one_hot)

        g_dgl = dgl.DGLGraph()
        g_dgl.from_networkx(nx_graph=graph, edge_attrs=['one_hot'])
        n_nodes = len(g_dgl.nodes())
        g_dgl.ndata['h'] = torch.ones((n_nodes, 128))

        return g_dgl, ring

    def _get_edge_map(self):
        edge_labels = set()
        print("Collecting edge labels.")
        for g in tqdm(os.listdir(self.path)):
            graph,_,_ = pickle.load(open(os.path.join(self.path, g), 'rb'))
            edges = {e_dict['label'] for _,_,e_dict in graph.edges(data=True)}
            edge_labels = edge_labels.union(edges)

        return {label:i for i,label in enumerate(sorted(edge_labels))}

def collate_block(samples):
    # The input `samples` is a list of pairs
    #  (graph, label).
    # print(len(samples))
    graphs, rings = map(list, zip(*samples))
    batched_graph = dgl.batch(graphs)
    K = k_block_list(rings, SimFunctionNode(method='rings', depth=4))
    return batched_graph, torch.from_numpy(K).detach().float()


class Loader():
    def __init__(self,
                 annotated_path='data/annotated/samples/',
                 batch_size=128,
                 num_workers=20,
                 debug=False,
                 shuffled=False):
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
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.dataset = V1(annotated_path=annotated_path,
                          debug=debug,
                          shuffled=shuffled)

        self.num_edge_types = self.dataset.num_edge_types

    def get_data(self):
        n = len(self.dataset)
        indices = list(range(n))
        # np.random.shuffle(indices)

        np.random.seed(0)
        split_train, split_valid = 0.7, 0.7
        train_index, valid_index = int(split_train * n), int(split_valid * n)

        train_indices = indices[:train_index]
        valid_indices = indices[train_index:valid_index]
        test_indices = indices[valid_index:]

        train_set = Subset(self.dataset, train_indices)
        valid_set = Subset(self.dataset, valid_indices)
        test_set = Subset(self.dataset, test_indices)

        print(len(train_set))

        train_loader = DataLoader(dataset=train_set, shuffle=True, batch_size=self.batch_size,
                                  num_workers=self.num_workers, collate_fn=collate_block)
        # valid_loader = DataLoader(dataset=valid_set, shuffle=True, batch_size=self.batch_size,
        #                           num_workers=self.num_workers, collate_fn=collate_block)
        test_loader = DataLoader(dataset=test_set, shuffle=True, batch_size=self.batch_size,
                                 num_workers=self.num_workers, collate_fn=collate_block)

        # return train_loader, valid_loader, test_loader
        return train_loader, 0, test_loader


pass
# for batch_idx, (graph, K) in enumerate(dataloader):
#     print(len(graph.nodes))
#     print(K.shape)


def test_train():
    # dataset = V1()
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=False, num_workers=20, collate_fn=collate_block)

    loader = Loader(annotated_path='../data/annotated/samples', batch_size=129, num_workers=20)
    # loader = Loader(annotated_path=annotated_path, batch_size=batch_size, num_workers=num_workers)

    train_loader, _, test_loader = loader.get_data()
    max_epochs = 800

    model = rgcn.Model(in_dim=128, h_dim=128, out_dim=128, num_rels=num_edge_types, num_bases=-1, num_hidden_layers=1)

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters())
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    plot = False
    model.train()
    # Loop over epochs
    for i, epoch in enumerate(range(max_epochs)):
        # Training

        loser = torch.nn.MSELoss()
        total_loss = 0
        time_epoch = time.perf_counter()

        for batch_idx, (graph, K) in enumerate(train_loader):
            print(len(K))
            optimizer.zero_grad()
            out = model(graph)
            # print(out)
            # size_block = len(out)

            K = torch.ones(K.shape) - K
            K_predict = torch.norm(out[:, None] - out, dim=2, p=2)

            # K_predict = torch.mm(out, out.t())
            # print(K_predict.size())
            # print(size_block)
            # print(K_predict)
            # print(K)
            # sys.exit()

            loss = loser(K_predict, K)
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        if plot and not i % 10:
            fig, (ax1, ax2, ax3) = plt.subplots(3)
            args = {'vmin': 0, 'vmax': 1}
            sns.heatmap(out.detach().numpy(), ax=ax1)
            ax1.set_title("Z")
            sns.heatmap(K_predict.detach().numpy(), ax=ax2, **args)
            ax2.set_title("pred")
            sns.heatmap(K.detach().numpy(), ax=ax3, **args)
            ax3.set_title("K")
            fig.suptitle(f'epoch {i}, loss: {total_loss}')
            plt.tight_layout()
            plt.show()
        print(f'epoch number {i} loss = {total_loss}, time = {time.perf_counter() - time_epoch}')


if __name__ == '__main__':
    import rgcn as rgcn
    import time
    test_train()
