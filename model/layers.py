import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from typing import Optional, Tuple
from torch import Tensor
from torch_scatter import scatter, scatter_add
from torch_sparse import SparseTensor
from torch_sparse import matmul, matmul, mul, fill_diag
from torch_sparse import sum as sparsesum
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import get_laplacian, add_remaining_self_loops, remove_self_loops
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


class ChebConv(MessagePassing):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        K: int,
        normalization: Optional[str] = 'sym',
        bias: bool = True,
        **kwargs,
    ) -> None:
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)

        assert K >= 0
        assert normalization in [None, 'sym', 'rw'], 'Invalid normalization'

        self.K = K
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalization = normalization
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.kaiming_uniform_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def __norm__(
        self,
        edge_index: Tensor,
        num_nodes: Optional[int],
        edge_weight: OptTensor,
        normalization: Optional[str],
        lambda_max: OptTensor = None,
        dtype: Optional[int] = None,
        batch: OptTensor = None,
    ):
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
        edge_index, edge_weight = get_laplacian(edge_index, edge_weight,
                                                normalization, dtype,
                                                num_nodes)
        assert edge_weight is not None

        if lambda_max is None:
            lambda_max = torch.norm(edge_weight, p=2)
        elif not isinstance(lambda_max, Tensor):
            lambda_max = torch.tensor(lambda_max, dtype=dtype,
                                      device=edge_index.device)
        assert lambda_max is not None

        if batch is not None and lambda_max.numel() > 1:
            lambda_max = lambda_max[batch[edge_index[0]]]

        edge_weight = (2.0 * edge_weight) / lambda_max
        edge_weight.masked_fill_(torch.isinf(edge_weight), 0)
        edge_weight.masked_fill_(torch.isnan(edge_weight), 0)

        loop_mask = edge_index[0] == edge_index[1]
        edge_weight[loop_mask] -= 1

        return edge_index, edge_weight

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: OptTensor = None,
        batch: OptTensor = None,
        lambda_max: OptTensor = None,
    ) -> Tensor:

        edge_index, norm = self.__norm__(
            edge_index,
            x.size(self.node_dim),
            edge_weight,
            self.normalization,
            lambda_max,
            dtype=x.dtype,
            batch=batch,
        )

        Tx_0 = x
        cheb = Tx_0

        if self.K >= 1:
            # propagate_type: (x: Tensor, norm: Tensor)
            Tx_1 = self.propagate(edge_index, x=x, norm=norm)
            cheb = cheb + Tx_1

        if self.K >= 2:
            for k in range(2, self.K + 1):
                Tx_2 = self.propagate(edge_index, x=Tx_1, norm=norm)
                Tx_2 = 2. * Tx_2 - Tx_0
                cheb = cheb + Tx_2
                Tx_0, Tx_1 = Tx_1, Tx_2

        out = F.linear(cheb, self.weight, self.bias)

        return out

    def message(self, x_j: Tensor, norm: Tensor) -> Tensor:
        return norm.view(-1, 1) * x_j

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.in_channels}, '
                f'{self.out_channels}, K={len(self.lins)}, '
                f'normalization={self.normalization})')


class ChebGibbsProp(MessagePassing):
    _cached_edge_index: Optional[Tuple[Tensor, Tensor]]
    _cached_adj_t: Optional[SparseTensor]

    def __init__(self, K: int = 2, gibbs_type: str = 'jackson', 
                 mu: int = 3, xi: float = 4.0, stigma: float = 0.5, heta: int = 2,
                 homophily: float = 0.8, improved: bool = False, 
                 cached: bool = False, add_self_loops: bool = True, 
                 normalize: bool = True, dropout: float = 0., **kwargs) -> None:
        kwargs.setdefault('aggr', 'add')
        super().__init__(**kwargs)
        assert K >= 0
        assert gibbs_type in ['none', 'dirichlet', 'fejer', 'jackson', 'lanczos', 'lorentz', 'vekic', 'wang']
        assert mu >= 1

        self.K = K
        self.gibbs_type = gibbs_type
        self.mu = mu
        self.xi = xi
        self.stigma = stigma
        self.heta = heta
        self.homophily = homophily
        self.cheb_coef = nn.Parameter(torch.empty(K + 1))
        self.gibbs_damp = torch.ones(K + 1)
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize
        self.dropout = dropout

        self._cached_edge_index = None
        self._cached_adj_t = None

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.cheb_coef, 2 / (self.K + 1))
        init.constant_(self.cheb_coef[0], 1 / (self.K + 1))
        init.ones_(self.gibbs_damp)

        if self.gibbs_type == 'fejer':
            for k in range(1, self.K + 1):
                self.gibbs_damp[k] = torch.tensor(1 - k / (self.K + 1))
        if self.gibbs_type == 'jackson':
            # Weiße, A., Wellein, G., Alvermann, A., & Fehske, H. (2006). 
            # The kernel polynomial method. 
            # Reviews of Modern Physics, 78, 275–306.
            # Weiße, A., & Fehske, H. (2008). Chebyshev Expansion Techniques. 
            # In Computational Many-Particle Physics (pp. 545–577). 
            # Springer Berlin Heidelberg.
            c = torch.tensor(torch.pi / (self.K+2))
            for k in range(1, self.K+1):
                self.gibbs_damp[k] = ((self.K+2-k) * torch.sin(c) * torch.cos(k * c) \
                                   + torch.cos(c) * torch.sin(k * c)) / ((self.K+2) * torch.sin(c))
        elif self.gibbs_type == 'lanczos':
            for k in range(1, self.K+1):
                self.gibbs_damp[k] = torch.sinc(torch.tensor(k / (self.K+1)))
            self.gibbs_damp = torch.pow(self.gibbs_damp, self.mu)
        elif self.gibbs_type == 'lorentz':
            # Vijay, A., Kouri, D., & Hoffman, D. (2004). 
            # Scattering and Bound States: 
            # A Lorentzian Function-Based Spectral Filter Approach. 
            # The Journal of Physical Chemistry A, 108(41), 8987-9003.
            for k in range(1, self.K + 1):
                self.gibbs_damp[k] = torch.sinh(self.xi * torch.tensor(1 - k / (self.K + 1))) / math.sinh(self.xi)
        elif self.gibbs_type == 'vekic':
            # M. Vekić, & S. R. White (1993). Smooth boundary 
            # conditions for quantum lattice systems. 
            # Physical Review Letters, 71, 4283–4286.
            for k in range(1, self.K + 1):
                self.gibbs_damp[k] = torch.tensor(k / (self.K + 1))
                self.gibbs_damp[k] = 0.5 * (1 - torch.tanh((self.gibbs_damp[k] - 0.5) / \
                                    (self.gibbs_damp[k] * (1 - self.gibbs_damp[k]))))
        elif self.gibbs_type == 'wang':
            # Wang, L.W. (1994). Calculating the density of 
            # states and optical-absorption spectra of 
            # large quantum systems by the plane-wave moments method. 
            # Physical Review B, 49, 10154–10158.
            for k in range(1, self.K + 1):
                self.gibbs_damp[k] = torch.tensor(k / (self.stigma * (self.K + 1)))
            self.gibbs_damp = -torch.pow(self.gibbs_damp, self.heta)
            self.gibbs_damp = torch.exp(self.gibbs_damp)

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
        out = Tx_0 * self.cheb_coef[0]

        if self.K >= 1:
            Tx_1 = self.propagate(edge_index, x=x, edge_weight=edge_weight, size=None)
            out = out + Tx_1 * self.cheb_coef[1] * self.gibbs_damp[1]

        if self.K >= 2:
            for k in range(2, self.K+1):
                Tx_2 = self.propagate(edge_index, x=Tx_1, edge_weight=edge_weight, size=None)
                Tx_2 = 2. * Tx_2 - Tx_0
                out = out + Tx_2 * self.cheb_coef[k] * self.gibbs_damp[k]
                Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: SparseTensor, x: Tensor) -> Tensor:
        return matmul(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'({self.__class__.__name__}(K={self.K}, gibbs_type={self.gibbs_type}, ' 
                f'mu={self.mu}, homophily={self.homophily})')