import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from model.layers import ChebConv, ChebGibbsProp
from torch import Tensor
from torch_geometric.nn.conv import GCNConv
from torch_geometric.data import Data
from utils import activation as act


class MLP(nn.Module):
    def __init__(self, dataset, args) -> None:
        super(MLP, self).__init__()
        self.linear1 = nn.Linear(in_features=dataset.num_features, out_features=args.num_hid, bias=True)
        self.linear2 = nn.Linear(in_features=args.num_hid, out_features=dataset.num_classes, bias=True)
        self.relu = nn.ReLU()
        self.dropout_pre = nn.Dropout(p=args.dropout_pre)
        self.dropout_in = nn.Dropout(p=args.dropout_in)
        self.dropout_suf = nn.Dropout(p=args.dropout_suf)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.linear1.weight, nonlinearity='relu')
        if self.linear1.bias is not None:
            init.zeros_(self.linear1.bias)
        
        init.xavier_normal_(self.linear2.weight)
        if self.linear2.bias is not None:
            init.zeros_(self.linear2.bias)

    def forward(self, data: Data) -> Tensor:
        x = data.x

        x = self.dropout_pre(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.dropout_in(x)
        x = self.linear2(x)
        x = self.dropout_suf(x)

        return x


class ChebNet(nn.Module):
    def __init__(self, dataset, args) -> None:
        super(ChebNet, self).__init__()
        self.chebconv1 = ChebConv(in_channels=dataset.num_features, out_channels=args.num_hid, 
                                K=args.order, normalization='sym', bias=True)
        self.chebconv2 = ChebConv(in_channels=args.num_hid, out_channels=dataset.num_classes, 
                                K=args.order, normalization='sym', bias=True)
        self.relu = nn.ReLU()
        self.dropout_pre = nn.Dropout(p=args.dropout_pre)
        self.dropout_in = nn.Dropout(p=args.dropout_in)
        self.dropout_suf = nn.Dropout(p=args.dropout_suf)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.chebconv1.lin.weight, nonlinearity='relu')
        if self.chebconv1.lin.bias is not None:
            init.zeros_(self.chebconv1.lin.bias)
        
        init.xavier_normal_(self.chebconv2.lin.weight)
        if self.chebconv2.lin.bias is not None:
            init.zeros_(self.chebconv2.lin.bias)

    def forward(self, data: Data) -> Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.dropout_pre(x)
        x = self.chebconv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.dropout_in(x)
        x = self.chebconv2(x, edge_index, edge_weight)
        x = self.dropout_suf(x)
        
        return x


class GCN(nn.Module):
    def __init__(self, dataset, args) -> None:
        super(GCN, self).__init__()
        self.gcnconv1 = GCNConv(in_channels=dataset.num_features, out_channels=args.num_hid, bias=True)
        self.gcnconv2 = GCNConv(in_channels=args.num_hid, out_channels=dataset.num_classes, bias=True)
        self.relu = nn.ReLU()
        self.dropout_pre = nn.Dropout(p=args.dropout_pre)
        self.dropout_in = nn.Dropout(p=args.dropout_in)
        self.dropout_suf = nn.Dropout(p=args.dropout_suf)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.gcnconv1.lin.weight, nonlinearity='relu')
        if self.gcnconv1.bias is not None:
            init.zeros_(self.gcnconv1.bias)
        
        init.xavier_normal_(self.gcnconv2.lin.weight)
        if self.gcnconv2.bias is not None:
            init.zeros_(self.gcnconv2.bias)

    def forward(self, data: Data) -> Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.dropout_pre(x)
        x = self.gcnconv1(x, edge_index, edge_weight)
        x = self.relu(x)
        x = self.dropout_in(x)
        x = self.gcnconv2(x, edge_index, edge_weight)
        x = self.dropout_suf(x)

        return x


class ChebGibbsNet(nn.Module):
    def __init__(self, dataset, args):
        super(ChebGibbsNet, self).__init__()
        self.linear1 = nn.Linear(dataset.num_features, args.num_hid, bias=True)
        self.linear2 = nn.Linear(args.num_hid, dataset.num_classes, bias=True)
        self.chebgibbs_prop = ChebGibbsProp(
                                            K=args.order, 
                                            gibbs_type=args.gibbs_type, 
                                            mu=args.mu, 
                                            xi=args.xi, 
                                            stigma=args.stigma, 
                                            heta=args.heta
                                        )
        self.dropout_pre = nn.Dropout(p=args.dropout_pre)
        self.dropout_in = nn.Dropout(p=args.dropout_in)
        self.dropout_suf = nn.Dropout(p=args.dropout_suf)
        self.leaky_relu = nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_normal_(self.linear1.weight)
        if self.linear1.bias is not None:
            init.zeros_(self.linear1.bias)

        self.chebgibbs_prop.reset_parameters()

        init.xavier_normal_(self.linear2.weight)
        if self.linear2.bias is not None:
            init.zeros_(self.linear2.bias)

    def forward(self, data: Data) -> Tensor:
        x, edge_index, edge_weight = data.x, data.edge_index, data.edge_weight

        x = self.dropout_pre(x)
        x = self.linear1(x)
        x = self.leaky_relu(x)
        x = self.dropout_in(x)
        x = self.linear2(x)
        x = self.dropout_suf(x)
        x = self.chebgibbs_prop(x, edge_index, edge_weight)
        
        return x