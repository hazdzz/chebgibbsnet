import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from torch import Tensor
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.conv.gcn_conv import gcn_norm
from torch_geometric.utils import get_laplacian, homophily, remove_self_loops
from torch_geometric.typing import Adj, OptTensor, PairTensor, SparseTensor
from typing import Optional, Tuple, Union


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

    def __init__(
        self, 
        K: int, 
        gibbs_type: str, 
        mu: int, 
        xi: float, 
        stigma: float, 
        heta: int,
        improved: bool = False, 
        cached: bool = False, 
        add_self_loops: bool = False, 
        normalize: bool = True, 
        **kwargs
    ) -> None:
        kwargs.setdefault('aggr', 'add')
        super(ChebGibbsProp, self).__init__(**kwargs)
        assert K >= 0
        assert gibbs_type in ['none', 'dirichlet', 'fejer', 'jackson', 'lanczos', 'lorentz', 'vekic', 'wang']
        assert mu >= 1

        self.K = K
        self.gibbs_type = gibbs_type
        self.mu = mu
        self.xi = xi
        self.stigma = stigma
        self.heta = heta
        self.cheb_coef = nn.Parameter(torch.empty(K + 1))
        self.gibbs_damp = nn.Parameter(torch.empty(K + 1), requires_grad=False)
        self.improved = improved
        self.cached = cached
        self.add_self_loops = add_self_loops
        self.normalize = normalize

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
            c = torch.tensor(math.pi / (self.K+2))
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
            # M. Vekić, & S. R. White (1993). 
            # Smooth boundary conditions for quantum lattice systems. 
            # Physical Review Letters, 71, 4283–4286.
            for k in range(1, self.K + 1):
                self.gibbs_damp[k] = torch.tensor(k / (self.K + 1))
                self.gibbs_damp[k] = 0.5 * (1 - torch.tanh((self.gibbs_damp[k] - 0.5) / \
                                    (self.gibbs_damp[k] * (1 - self.gibbs_damp[k]))))
        elif self.gibbs_type == 'wang':
            # Wang, L.W. (1994). 
            # Calculating the density of 
            # states and optical-absorption spectra of 
            # large quantum systems by the plane-wave moments method. 
            # Physical Review B, 49, 10154–10158.
            for k in range(1, self.K + 1):
                self.gibbs_damp[k] = torch.tensor(k / (self.stigma * (self.K + 1)))
            self.gibbs_damp = -torch.pow(self.gibbs_damp, self.heta)
            self.gibbs_damp = torch.exp(self.gibbs_damp)

        self._cached_edge_index = None
        self._cached_adj_t = None

    def forward(self, x: Tensor, edge_index: Adj, edge_weight: OptTensor = None) -> Tensor:
        if edge_weight is None:
            edge_weight = torch.ones((edge_index.size(1), ), dtype=x.dtype, device=x.device)
        edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)

        x_discrete = torch.argmax(x, dim=1)
        node_homophily = homophily(edge_index=edge_index, y=x_discrete, method='node')

        if self.normalize:
            if isinstance(edge_index, Tensor):
                cache = self._cached_edge_index
                if cache is None:
                    edge_index, edge_weight = gcn_norm(
                        edge_index, edge_weight, x.size(self.node_dim),
                        self.improved, self.add_self_loops, self.flow)
                    if node_homophily <= 0.5:
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

        Tx_0 = x
        out = Tx_0 * self.cheb_coef[0]

        if self.K >= 1:
            Tx_1 = self.propagate(edge_index, x=Tx_0, edge_weight=edge_weight)
            out = out + Tx_1 * self.cheb_coef[1] * self.gibbs_damp[1]

        if self.K >= 2:
            for k in range(2, self.K+1):
                Tx_2 = self.propagate(edge_index, x=Tx_1, edge_weight=edge_weight)
                Tx_2 = 2. * Tx_2 - Tx_0
                out = out + Tx_2 * self.cheb_coef[k] * self.gibbs_damp[k]
                Tx_0, Tx_1 = Tx_1, Tx_2

        return out

    def message(self, x_j: Tensor, edge_weight: OptTensor) -> Tensor:
        return x_j if edge_weight is None else edge_weight.view(-1, 1) * x_j

    def message_and_aggregate(self, adj_t: Adj, x: Tensor) -> Tensor:
        return spmm(adj_t, x, reduce=self.aggr)

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'K={self.K}, '
                f'gibbs_type={self.gibbs_type!r}, '
                f'mu={self.mu}, '
                f'xi={self.xi}, '
                f'stigma={self.stigma}, '
                f'heta={self.heta}, '
                f'improved={self.improved}, '
                f'cached={self.cached}, '
                f'add_self_loops={self.add_self_loops}, '
                f'normalize={self.normalize})'
            )