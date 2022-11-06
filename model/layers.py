import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Tuple
from torch import Tensor
from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor
from torch_sparse import matmul, matmul, mul, fill_diag
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import add_remaining_self_loops, remove_self_loops
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_geometric.typing import Adj,OptTensor, PairTensor

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (Tensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> PairTensor  # noqa
    pass

@torch.jit._overload
def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):
    # type: (SparseTensor, OptTensor, Optional[int], bool, bool, str, Optional[int]) -> SparseTensor  # noqa
    pass

def gcn_norm(edge_index, edge_weight=None, num_nodes=None, improved=False,
             add_self_loops=True, flow="source_to_target", dtype=None):

    fill_value = 2. if improved else 1.

    if isinstance(edge_index, SparseTensor):
        assert flow in ["source_to_target"]
        adj_t = edge_index
        if not adj_t.has_value():
            adj_t = adj_t.fill_value(1., dtype=dtype)
        if add_self_loops:
            adj_t = fill_diag(adj_t, fill_value)
        deg = sparsesum(adj_t, dim=1)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0)
        deg_inv_sqrt.masked_fill_(torch.isnan(deg_inv_sqrt), 0)
        adj_t = mul(adj_t, deg_inv_sqrt.view(-1, 1))
        adj_t = mul(adj_t, deg_inv_sqrt.view(1, -1))

        return adj_t

    else:
        assert flow in ["source_to_target", "target_to_source"]
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=dtype,
                                     device=edge_index.device)

        if add_self_loops:
            edge_index, tmp_edge_weight = add_remaining_self_loops(
                edge_index, edge_weight, fill_value, num_nodes)
            assert tmp_edge_weight is not None
            edge_weight = tmp_edge_weight

        row, col = edge_index[0], edge_index[1]
        idx = col if flow == "source_to_target" else row
        deg = scatter_add(edge_weight, idx, dim=0, dim_size=num_nodes)
        deg_inv_sqrt = deg.pow_(-0.5)
        deg_inv_sqrt.masked_fill_(torch.isinf(deg_inv_sqrt), 0)
        deg_inv_sqrt.masked_fill_(torch.isnan(deg_inv_sqrt), 0)

        return edge_index, deg_inv_sqrt[row] * edge_weight * deg_inv_sqrt[col]

class ChebGibbs(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int, gibbs_type: str, mu: int, 
                 homophily: float, improved: bool = False, 
                 cached: bool = False, add_self_loops: bool = True, 
                 normalize: bool = True, dropout: float = 0., **kwargs):
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        assert K >= 0
        assert gibbs_type in ['none', 'jackson', 'lanczos', 'zhang']
        assert mu >= 1

        self.K = K
        self.gibbs_type = gibbs_type
        self.mu = mu
        self.homophily = homophily
        # self.cheb_coef = nn.Parameter(torch.empty(K+1))
        self.gibbs_damp = torch.ones(K+1)
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.dropout = dropout

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        # init.ones_(self.cheb_coef)

        if self.gibbs_type == 'jackson':
            c = torch.tensor(torch.pi / (self.K+2))
            for k in range(1, self.K+1):
                self.gibbs_damp[k] = ((self.K+2-k) * torch.sin(c) * torch.cos(k * c) \
                                   + torch.cos(c) * torch.sin(k * c)) / ((self.K+2) * torch.sin(c))
        elif self.gibbs_type == 'lanczos':
            for k in range(1, self.K+1):
                self.gibbs_damp[k] = torch.sinc(torch.tensor(k / (self.K+1)))
            self.gibbs_damp = torch.pow(self.gibbs_damp, self.mu)
        elif self.gibbs_type == 'zhang':
            for k in range(2, self.K+1):
                self.gibbs_damp[k] = self.gibbs_damp[k] * math.pow(2, 1-k)

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        self.gibbs_damp = self.gibbs_damp.to(x.device)

        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow)
                    if self.homophily <= 0.5:
                        edge_weight = -edge_weight
                    if self.cached:
                        self._cached_edge_index = (edge_index, edge_weight)
                else:
                    edge_index, edge_weight = cache[0], cache[1]

            elif isinstance(edge_index, SparseTensor):
                cache = self._cached_adj_t
                if cache is None:
                    edge_index = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow)
                    if self.cached:
                        self._cached_adj_t = edge_index
                else:
                    edge_index = cache

        if self.dropout > 0 and self.training:
            if isinstance(edge_index, Tensor):
                assert edge_weight is not None
                edge_weight = F.dropout(edge_weight, p=self.dropout)
            else:
                value = edge_index.storage.value()
                assert value is not None
                value = F.dropout(value, p=self.dropout)
                edge_index = edge_index.set_value(value, layout='coo')

        Tx_0 = x
        out = Tx_0

        Tx_1 = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
        out = out + Tx_1 * self.gibbs_damp[1]

        for k in range(2, self.K+1):
            Tx_2 = self.propagate(edge_index, x=Tx_1, edge_weight=edge_weight, size=None)
            Tx_2 = 2. * Tx_2 - Tx_0
            out = out + Tx_2 * self.gibbs_damp[k]
            Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'({self.__class__.__name__}(K={self.K}, gibbs_type={self.gibbs_type}, ' 
                f'mu={self.mu}, homophily={self.homophily})')