import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch_geometric.nn as pygnn

from model import layers

class ChebGibbsNet(nn.Module):
    def __init__(self, dataset, args, homophily):
        super(ChebGibbsNet, self).__init__()
        self.lin1 = nn.Linear(in_features=dataset.num_features, out_features=args.num_hid)
        self.lin2 = nn.Linear(in_features=args.num_hid, out_features=dataset.num_classes)
        self.prop = layers.ChebGibbs(K=args.order, gibbs_type=args.gibbs_type, 
                                     mu=args.mu, homophily=homophily)
        self.tanh = nn.Tanh()
        self.dropout_pre = nn.Dropout(p=args.droprate_pre)
        self.dropout_in = nn.Dropout(p=args.droprate_in)
        self.dropout_suf = nn.Dropout(p=args.droprate_suf)
        self.log_softmax = nn.LogSoftmax(dim=1)

    #     self.reset_parameters()

    # def reset_parameters(self):
    #     init.orthogonal_(self.lin1.weight)
    #     init.zeros_(self.lin1.bias)
    #     init.orthogonal_(self.lin2.weight)
    #     init.zeros_(self.lin2.bias)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.dropout_pre(x)
        x = self.lin1(x)
        x = self.tanh(x)
        x = self.dropout_in(x)
        x = self.lin2(x)
        x = self.dropout_suf(x)
        x = self.prop(x, edge_index, edge_weight)
        x = self.log_softmax(x)
        
        return x