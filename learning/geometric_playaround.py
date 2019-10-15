import os.path as osp

import torch
from torch_geometric.data import Dataset
import pickle

if __name__ == '__main__':
    import sys

    sys.path.append('../')

from data_processor.node_sim import k_block, SimFunctionNode


class MyOwnDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(MyOwnDataset, self).__init__(root, transform, pre_transform)
        self.all_graphs = pickle.load(open('../data/subset_annotated.p', 'rb'))[:30]
        self.node_sim = SimFunctionNode(method='rings', depth=4)
        self.n = len(self.all_graphs)

    @property
    def raw_file_names(self):
        return 0

    @property
    def processed_file_names(self):
        return 0

    def __len__(self):
        return self.n ** 2

    def _download(self):
        pass

    def _process(self):
        pass

    def get(self, idx):
        data = torch.load(osp.join(self.processed_dir, 'data_{}.pt'.format(idx)))
        return data
