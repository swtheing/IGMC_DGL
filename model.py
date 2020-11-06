import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from dgl import DGLGraph
import dgl.function as fn
from functools import partial
from torch.nn import Linear, Conv1d
def glorot(tensor):
    if tensor is not None:
        stdv = math.sqrt(6.0 / (tensor.size(-2) + tensor.size(-1)))
        tensor.data.uniform_(-stdv, stdv)
def ones(tensor):
    if tensor is not None:
        tensor.data.fill_(1)

class RGCNLayer(nn.Module):
    def __init__(self, in_feat, out_feat, num_rels, num_bases=-1, bias=None,
                 activation=None, is_input_layer=False):
        super(RGCNLayer, self).__init__()
        self.in_feat = in_feat
        self.out_feat = out_feat
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.has_bias = bias
        self.activation = activation
        self.is_input_layer = is_input_layer

        # sanity check
        if self.num_bases <= 0 or self.num_bases > self.num_rels:
            self.num_bases = self.num_rels

        # weight bases in equation (3)
        self.weight = nn.Parameter(torch.Tensor(self.num_bases, self.in_feat,
                                                self.out_feat))
        self.root_weight = nn.Parameter(torch.Tensor(self.in_feat,
                                                self.out_feat))
        if self.num_bases < self.num_rels:
            # linear combination coefficients in equation (3)
            self.w_comp = nn.Parameter(torch.Tensor(self.num_rels, self.num_bases))

        # add bias
        if self.has_bias:
            self.bias = nn.Parameter(torch.Tensor(out_feat))

        # init trainable parameters
        #ones(self.weight)
        #glorot(self.weight)
        nn.init.xavier_uniform_(self.weight,
                                gain=nn.init.calculate_gain('relu'))
        
        nn.init.xavier_uniform_(self.root_weight,
                                gain=nn.init.calculate_gain('relu'))
        #ones(self.root_weight)
        #glorot(self.root_weight)
        if self.num_bases < self.num_rels:
            nn.init.xavier_uniform_(self.w_comp,
                                    gain=nn.init.calculate_gain('relu'))
            #glorot(self.w_comp)
            #ones(self.w_comp)
        if self.has_bias:
            nn.init.constant_(self.bias, 0)
    
    def forward(self, g):
        if self.num_bases < self.num_rels:
            # generate all weights from bases (equation (3))
            weight = torch.matmul(self.w_comp, self.weight.view(self.num_bases, -1))
            weight = weight.view(self.num_rels, self.in_feat, self.out_feat)
        else:
            weight = self.weight


        if self.is_input_layer:
            def message_func(edges):
                # for input layer, matrix multiply can be converted to be
                # an embedding lookup using source node id
                #print(edges.data['norm'])
                #print(edges.data['rel_type'])
                #w = weight[edges.data['rel_type']]
                w = torch.index_select(weight, 0, edges.data['rel_type'])
                msg = torch.bmm(edges.src['node_label'].unsqueeze(1), w).squeeze() #* edges.data['norm']
                
                return {'msg': msg}
        else:
            def message_func(edges):
                #w = weight[edges.data['rel_type']]
                w = torch.index_select(weight, 0, edges.data['rel_type'])
                msg = torch.bmm(edges.src['h'].unsqueeze(1), w).squeeze()
                msg = msg #* edges.data['norm']
                return {'msg': msg}

        def apply_func(nodes):
            if self.is_input_layer:
                h = torch.mm(nodes.data['node_label'], self.root_weight) + nodes.data['o']
            else:
                h = torch.mm(nodes.data['h'], self.root_weight) + nodes.data['o']
            if self.has_bias:
                h = h + self.bias
            #print(h)
            if self.activation:
                h = self.activation(h)
            return {'h': h}

        g.update_all(message_func, fn.sum(msg='msg', out='o'), apply_func)
        
class IGMC(nn.Module):
    def __init__(self,  in_fea, latent_dim = [32, 32, 32, 32],  num_rels = 5,
                 num_bases=-1):
        super(IGMC, self).__init__()
        self.num_rels = num_rels
        self.num_bases = num_bases
        self.convs = torch.nn.ModuleList()
        self.convs.append(RGCNLayer(in_fea, latent_dim[0], num_rels, num_bases, bias=True,
                                    activation=torch.tanh, is_input_layer=True))
        for i in range(0, len(latent_dim)-1):
            self.convs.append(RGCNLayer(latent_dim[i], latent_dim[i+1], num_rels, num_bases, bias=True,
                                    activation=torch.tanh, is_input_layer=False))
        self.lin1 = Linear(2*sum(latent_dim), 128)
        self.lin2 = Linear(128, 1)

    def forward(self, g_batch):
        h_s = []
        for layer in self.convs:
            layer(g_batch)
            h = g_batch.ndata['h']
            h_s.append(h)
        h = torch.cat(h_s, 1)
        x = g_batch.ndata['node_label']
        users = x[:, 0] == 1
        items = x[:, 1] == 1
        h = torch.cat([h[users], h[items]], 1)
        h = F.relu(self.lin1(h))
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)
        return h[:, 0]
