"""
DGL : batched graphs is using the collate function
"""

"""
Geometric : Graph object + Batch object
RGCN layer is easier
"""

"""
https://docs.dgl.ai/tutorials/models/1_gnn/4_rgcn.html#sphx-glr-tutorials-models-1-gnn-4-rgcn-py
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl.function as fn
from dgl.nn.pytorch.glob import SumPooling,GlobalAttentionPooling
from functools import partial
import dgl
from dgl import mean_nodes
from dgl.nn.pytorch.conv import RelGraphConv


class Attributor(nn.Module):
    def __init__(self, dims):
        super(Attributor, self).__init__()
        # self.num_nodes = num_nodes
        self.dims = dims

        # create layers
        self.build_model()

    def build_model(self):
        layers = []

        short = self.dims[:-1]
        last_hidden, last = (*self.dims[-2:],)

        for dim_in, dim_out in zip(short, short[1:]):
            # print('in',dim_in, dim_out)
            layers.append(nn.Linear(dim_in, dim_out))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(0.5))
        # hidden to output
        layers.append(nn.Linear(last_hidden, last))
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

###############################################################################
# Define full R-GCN model
# ~~~~~~~~~~~~~~~~~~~~~~~
class Model(nn.Module):
    def __init__(self, dims, attributor_dims, num_rels, pool='sum', num_bases=-1):
        """

        :param dims: the embeddings dimensions
        :param attributor_dims: the number of motifs to look for
        :param num_rels: the number of possible edge types
        :param num_bases: technical rGCN option
        :param rec: the constant in front of reconstruction loss
        :param mot: the constant in front of motif detection loss
        :param orth: the constant in front of dictionnary orthogonality loss
        :param attribute: Wether we want the network to use the attribution module
        """
        super(Model, self).__init__()
        # self.num_nodes = num_nodes
        self.dims = dims
        self.num_rels = num_rels
        self.num_bases = num_bases

        if pool == 'att':
            pooling_gate_nn = nn.Linear(dims[-1], 1)
            self.pool = GlobalAttentionPooling(pooling_gate_nn)
        else:
            self.pool = SumPooling()
        self.attributor = Attributor(attributor_dims)

        # create rgcn layers
        self.build_model()

    def build_model(self):
        self.layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = (*self.dims[-2:],)

        # input feature is just node degree
        i2h = self.build_hidden_layer(1, self.dims[0])
        self.layers.append(i2h)

        for dim_in, dim_out in zip(short, short[1:]):
            # print('in',dim_in, dim_out)
            h2h = self.build_hidden_layer(dim_in, dim_out)
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer(last_hidden, last)
        # print('last',last_hidden,last)
        self.layers.append(h2o)
        print(self.layers)

    def build_hidden_layer(self, in_dim, out_dim):
        return RelGraphConv(in_dim, out_dim, self.num_rels,
                            num_bases=self.num_bases,
                            activation=F.relu)

    # No activation for the last layer
    def build_output_layer(self, in_dim, out_dim):
        return RelGraphConv(in_dim, out_dim, self.num_rels, num_bases=self.num_bases,
                            activation=None)

    def forward(self, g):
        h = g.in_degrees().view(-1, 1).float()
        for layer in self.layers:
            # layer(g)
            h = layer(g, h, g.edata['one_hot'])
        g.ndata['h'] = h
        # print(g.ndata['h'].size())
        embeddings = g.ndata.pop('h')
        fp = self.pool(g, embeddings)
        fp = self.attributor(fp)
        return fp

    # Below are loss computation function related to this model


    def compute_loss(self, target_fp, pred_fp):
        """
        Compute the total loss of the model.
        Includes the reconstruction loss with optional similarity/distance boolean switch
        If the attributions are not None, adds the motif regularisation (orth term) as well as a motif finding
        loss with an optional scaling switch.
        :param embeddings: The first tensor produced by a forward call
        :param attributions: The second one
        :param target_K:
        :param similarity:
        :param scaled:
        :return:
        """
        # loss = torch.nn.BCELoss()(pred_fp, target_fp)
        loss = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor([5], dtype=torch.float))(pred_fp, target_fp)
        return loss

    def draw_rec(self, true_K, predicted_K, title=""):
        """
        A way to assess the reconstruction task visually
        :param true_K:
        :param predicted_K:
        :param loss_value: python float
        :return:
        """
        fig, (ax1, ax2) = plt.subplots(1, 2)
        sns.heatmap(true_K.clone().detach().numpy(), vmin=0, vmax=1, ax=ax1, square=True, cbar=False)
        sns.heatmap(predicted_K.clone().detach().numpy(), vmin=0, vmax=1, ax=ax2, square=True, cbar=False,
                    cbar_kws={"shrink": 1})
        ax1.set_title("Ground Truth")
        ax2.set_title("GCN")
        fig.suptitle(title)
        plt.tight_layout()
        plt.show()

