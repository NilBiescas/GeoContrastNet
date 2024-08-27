import torch
import torch.nn as nn
import math
import dgl.function as fn
import torch.nn.functional as F

class GCN_LAYER(nn.Module):
    def __init__(self,
                 in_feats,
                 out_feats,
                 activation,
                 Tresh_distance,
                 added_features = 24,
                 bias=True,
                 use_pp=False,
                 use_lynorm=True):
        
        super(GCN_LAYER, self).__init__()

        self.added_features = added_features
        self.in_feats = in_feats + self.added_features #+ 24 # With distance, angle 11
        self.linear = nn.Linear(self.in_feats, out_feats, bias=bias)
        self.activation = activation
        self.use_pp = use_pp
        self.Tresh_distance = Tresh_distance

        if use_lynorm:
            self.lynorm = nn.LayerNorm(out_feats, elementwise_affine=True)
        else:
            self.lynorm = lambda x: x
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.linear.weight.size(1))
        self.linear.weight.data.uniform_(-stdv, stdv)
        if self.linear.bias is not None:
            self.linear.bias.data.uniform_(-stdv, stdv)

    def forward(self, g, h):
        g = g.local_var()
        
        norm = g.ndata['norm']

        g.send_and_recv(g.edges(), fn.copy_e('m', 'h'), fn.sum('h', 'sum_h'))
        ah = g.ndata.pop('sum_h')
        h = self.concat(h, ah, norm)

        h = self.linear(h)
        h = self.lynorm(h)
        if self.activation:
            h = self.activation(h)
        return h

    def concat(self, h, ah, norm):
        ah = ah * norm
        h = torch.cat((h, ah), dim=1)
        return h

class class_contrastive_model(nn.Module):
    def __init__(self, layers_dimensions, dropout, Tresh_distance, added_features = 24, concat_hidden=False, **kwargs):
        
        super(class_contrastive_model, self).__init__()
        self._concat_hidden = concat_hidden
        self.dropout = nn.Dropout(dropout)
        self.encoder = nn.ModuleList()
        self.Tresh_distance = Tresh_distance
        self.layers_dimensions = layers_dimensions
        
        for i in range(len(layers_dimensions) - 1):
            # ENCODER
            self.encoder.append(GCN_LAYER(   in_feats           = layers_dimensions[i],  
                                                out_feats          = layers_dimensions[i+1],
                                                Tresh_distance     = self.Tresh_distance,
                                                activation         = F.relu,
                                                added_features     = added_features))
    def forward(self, g):
        h = g.ndata['Geometric']
        all_hidden = []

        for conv in self.encoder:
            h = conv(g, h)

            if self._concat_hidden:
                all_hidden.append(h)
                
        h = self.dropout(h) 
        return h

    def get_embeddigns(self, loader):
        import numpy as np
        with torch.no_grad():
            embeddings = []
            labels = []
            for graph, label in loader:
                graph = graph.to('cuda:0')
                x = graph.ndata['Geometric'].to('cuda:0')
                out = self.forward(graph)
                embeddings.append(out.cpu().numpy())
                labels.append(label.cpu().numpy())

            embeddings = np.concatenate(embeddings, axis=0)
            labels = np.concatenate(labels, axis=0)
        return embeddings, labels