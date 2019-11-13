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
from functools import partial
import dgl
from dgl import mean_nodes


class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer
        
        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat, self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
        if self.bias:
            nn.init.xavier_uniform_(self.bias,
                                    gain=nn.init.calculate_gain('relu'))

    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = self.weight.view(self.in_feat, self.num_bases, self.out_feat)
            weight = torch.matmul(self.w_comp, weight).view(self.num_rels,
                                                            self.in_feat, self.out_feat)
        else:
            weight = self.weight

        def message_func(edges):
            # print(edges.data['one_hot'].size())
            # print(weight.size(),weight)

            w = weight[edges.data['one_hot']]
            # print(w.size(),w)
            msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
            # msg = msg * edges.data['norm']
            return {'msg': msg}

        def apply_func(nodes):
            h = nodes.data['h']
            if self.bias:
                h = h + self.bias
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.set_n_initializer(dgl.init.zero_initializer)

        g.update_all(message_func, fn.sum(msg='msg', out='h'), apply_func)
        # print(self.in_feat,self.out_feat)
        # print('h', g.ndata['h'].size())
        # print('other',g.ndata['other'].size())
        # g.ndata['h'] = g.ndata.pop('other')


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
    def __init__(self, dims, attributor_dims, num_rels, num_bases=-1):
        super(Model, self).__init__()
        # self.num_nodes = num_nodes
        self.dims = dims
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.attributor_dims = attributor_dims

        # create rgcn layers
        self.build_model()

        if self.attributor_dims is not None:
            self.num_modules = attributor_dims[-1]
            self.dimension_embedding = dims[-1]

            #  create attributor
            self.attributor = Attributor(attributor_dims)

    def build_model(self):
        self.layers = nn.ModuleList()

        short = self.dims[:-1]
        last_hidden, last = (*self.dims[-2:],)

        for dim_in, dim_out in zip(short, short[1:]):
            # print('in',dim_in, dim_out)
            h2h = self.build_hidden_layer(dim_in, dim_out)
            self.layers.append(h2h)
        # hidden to output
        h2o = self.build_output_layer(last_hidden, last)
        # print('last',last_hidden,last)
        self.layers.append(h2o)

    def build_hidden_layer(self, in_dim, out_dim):
        return RGCNLayer(in_dim, out_dim, self.num_rels, self.num_bases,
                         activation=F.relu)

    # No activation for the last layer
    def build_output_layer(self, in_dim, out_dim):
        return RGCNLayer(in_dim, out_dim, self.num_rels, self.num_bases)

    def forward(self, g):
        # print('data', g.edata['one_hot'].size())
        for layer in self.layers:
            # print(g.ndata['h'].size())
            layer(g)
            # print(g.ndata['h'].size())
        attributions = self.attributor(mean_nodes(g, 'h'))
        return g.ndata['h'], attributions
