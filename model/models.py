import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GCNConv
from model import layers
from utils import activation as act


class MLP(nn.Module):
    def __init__(self, dataset, args):
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_features=dataset.num_features, out_features=args.num_hid, bias=True)
        self.linear2 = nn.Linear(in_features=args.num_hid, out_features=dataset.num_classes, bias=True)
        self.relu = nn.ReLU()
        self.dropout_pre = nn.Dropout(p=args.droprate_pre)
        self.dropout_in = nn.Dropout(p=args.droprate_in)
        self.dropout_suf = nn.Dropout(p=args.droprate_suf)

    def forward(self, data):
        x = data.x

        x = self.dropout_pre(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout_in(x)
        x = self.linear2(x)
        x = self.dropout_suf(x)

        return x


class ChebNet(nn.Module):
    def __init__(self, dataset, args):
        super(ChebNet, self).__init__()
        self.chebconv1 = layers.ChebConv(in_channels=dataset.num_features, out_channels=args.num_hid, 
                                         K=args.order, normalization='sym', bias=True)
        self.chebconv2 = layers.ChebConv(in_channels=args.num_hid, out_channels=dataset.num_classes, 
                                         K=args.order, normalization='sym', bias=True)
        self.relu = nn.ReLU()
        self.dropout_pre = nn.Dropout(p=args.droprate_pre)
        self.dropout_in = nn.Dropout(p=args.droprate_in)
        self.dropout_suf = nn.Dropout(p=args.droprate_suf)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.dropout_pre(x)
        x = self.chebconv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.dropout_in(x)
        x = self.chebconv2(x, edge_index, edge_weight)
        x = self.dropout_suf(x)
        
        return x


class GCN(nn.Module):
    def __init__(self, dataset, args):
        super(GCN, self).__init__()
        self.gcnconv1 = GCNConv(in_channels=dataset.num_features, out_channels=args.num_hid, bias=True)
        self.gcnconv2 = GCNConv(in_channels=args.num_hid, out_channels=dataset.num_classes, bias=True)
        self.relu = nn.ReLU()
        self.dropout_pre = nn.Dropout(p=args.droprate_pre)
        self.dropout_in = nn.Dropout(p=args.droprate_in)
        self.dropout_suf = nn.Dropout(p=args.droprate_suf)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.dropout_pre(x)
        x = self.gcnconv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.dropout_in(x)
        x = self.gcnconv2(x, edge_index, edge_weight)
        x = self.dropout_suf(x)

        return x


class ChebGibbsNet(nn.Module):
    def __init__(self, dataset, args, homophily):
        super(ChebGibbsNet, self).__init__()
        assert args.act in ['silu', 'gelu', 'mish', 'tanhexp', 'sinsig', 'diracrelu', 'smu']

        self.lin1 = nn.Linear(in_features=dataset.num_features, out_features=args.num_hid)
        self.lin2 = nn.Linear(in_features=args.num_hid, out_features=dataset.num_classes)
        self.prop = layers.ChebGibbsProp(K=args.order, gibbs_type=args.gibbs_type, 
                                         mu=args.mu, homophily=homophily)

        if args.act == 'silu':
            self.act = nn.SiLU()
        elif args.act == 'gelu':
            self.act = nn.GELU()
        elif args.act == 'mish':
            self.act = nn.Mish()
        elif args.act == 'tanhexp':
            self.act = act.TanhExp()
        elif args.act == 'sinsig':
            self.act = act.SinSig()
        elif args.act == 'diracrelu':
            self.act = act.DiracReLU()
        elif args.act == 'smu':
            self.act = act.SMU()
        
        self.dropout_pre = nn.Dropout(p=args.droprate_pre)
        self.dropout_in = nn.Dropout(p=args.droprate_in)
        self.dropout_suf = nn.Dropout(p=args.droprate_suf)

    def forward(self, data):
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.dropout_pre(x)
        x = self.lin1(x)
        x = self.act(x)
        x = self.dropout_in(x)
        x = self.lin2(x)
        x = self.act(x)
        x = self.prop(x, edge_index, edge_weight)
        x = self.dropout_suf(x)
        
        return x