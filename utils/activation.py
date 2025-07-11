import math
import numbers
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

from utils import functional as func
from torch import Size, Tensor
from typing import Union, List, Optional, Tuple


class SReLU(nn.Module):
    def __init__(self, normalized_shape: Union[int, List[int], Size], bias_init: float = 0.0):
        super(SReLU, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        self.bias_init = bias_init
        self.relu = nn.ReLU()

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.bias, self.bias_init)

    def forward(self, input: Tensor) -> Tensor:
        return self.relu(input - self.bias) + self.bias


class SMU(nn.Module):
    __constants__ = ['alpha', 'beta', 'requires_grad', 'inplace']
    alpha: float
    beta: float
    requires_grad: bool
    inplace: bool

    def __init__(
        self, 
        alpha: float = 0.01, 
        beta: float = 100.0, 
        requires_grad: Optional[bool] = True, 
        inplace: Optional[bool] = False
    ) -> None:
        super(SMU, self).__init__()
        self.alpha = alpha
        self.beta = nn.Parameter(torch.empty(1), requires_grad=requires_grad)
        self.beta_init = beta
        self.inplace = inplace

        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.beta, self.beta_init)
        
    def forward(self, input: Tensor) -> Tensor:
        return func.smu(input, self.alpha, self.beta, self.inplace)


class CaLU(nn.Module):
    def __init__(self) -> None:
        super(CaLU, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return func.calu(input)


class DyT(nn.Module):
    def __init__(
        self, 
        normalized_shape: Union[int, List[int], Size], 
        alpha: float = 1.0, 
        bias: bool = False
    ) -> None:
        super(DyT, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.alpha = nn.Parameter(torch.empty(1))
        self.alpha_init = alpha
        self.weight = nn.Parameter(torch.empty(self.normalized_shape))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.normalized_shape))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()

    def reset_parameters(self) -> None:
        init.constant_(self.alpha, self.alpha_init)
        init.ones_(self.weight)
        if self.bias is not None:
            init.zeros_(self.bias)

    def forward(self, input: Tensor) -> Tensor:
        result = self.weight * torch.tanh(self.alpha * input)
        
        if self.bias is not None:
            result = result + self.bias
        
        return result


class Siglog(nn.Module):
    def __init__(self) -> None:
        super(Siglog, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return func.siglog(input)


class CCardioid(nn.Module):
    def __init__(self) -> None:
        super(CCardioid, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return func.c_cardioid(input)


class iGaussian(nn.Module):
    def __init__(self, sigma: float = 1.0) -> None:
        super(iGaussian, self).__init__()
        self.sigma = sigma

    def forward(self, input: Tensor) -> Tensor:
        return func.igaussian(input, self.sigma)
    

class CSigmoid(nn.Module):
    def __init__(self) -> None:
        super(CSigmoid, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return func.c_sigmoid(input)
    

class CTanh(nn.Module):
    def __init__(self) -> None:
        super(CTanh, self).__init__()

    def forward(self, input: Tensor) -> Tensor:
        return func.c_tanh(input)


class CSoftsign(nn.Module):
    def __init__(self) -> None:
        super(CSoftsign, self).__init__()

    def forward(self, x: Tensor) -> Tensor:
        return func.c_softsign(x)


class zReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(zReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return func.z_relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class modReLU(nn.Module):
    __constants__ = ['bias', 'rounding_mode', 'inplace']
    bias: float
    rounding_mode: str
    inplace: bool

    def __init__(
        self, 
        bias: float = -math.sqrt(2), 
        rounding_mode: str = None, 
        inplace: bool = False
    ) -> None:
        super(modReLU, self).__init__()
        self.bias = bias
        self.rounding_mode = rounding_mode
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return func.mod_relu(
                        input, 
                        bias=self.bias, 
                        rounding_mode=self.rounding_mode, 
                        inplace=self.inplace
                    )

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return 'bias={}, rounding_mode={}, inplace_str={}'.format(
                                                                self.bias, 
                                                                self.rounding_mode, 
                                                                inplace_str
                                                            )


class CReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False) -> None:
        super(CReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return func.c_relu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class CLeakyReLU(nn.Module):
    __constants__ = ['negative_slope', 'inplace']
    negative_slope: float
    inplace: bool

    def __init__(self, negative_slope: float = 1e-2, inplace: bool = False) -> None:
        super(CLeakyReLU, self).__init__()
        self.negative_slope = negative_slope
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return func.c_leaky_relu(input, self.negative_slope, self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'negative_slope={}{}'.format(self.negative_slope, inplace_str)


# Anil, C., Lucas, J., & Grosse, R. (2019). 
# Sorting Out Lipschitz Function Approximation. 
# In Proceedings of the 36th International Conference on Machine Learning (pp. 291–301). PMLR.
class GroupSort(nn.Module):
    def __init__(self, num_units: int, dim: int = -1) -> None:
        super(GroupSort, self).__init__()
        self.num_units = num_units
        self.dim = dim
        
    def groupsort(self, input: Tensor) -> Tensor:
        num_channels = input.size(self.dim)
        assert num_channels % self.num_units == 0, \
            f"Number of channels ({num_channels}) is not a multiple of num_units ({self.num_units})"
        
        shape = list(input.size())
        group_size = num_channels // self.num_units
        
        if self.dim == -1:
            shape[self.dim] = -1
            shape.append(group_size)
        else:
            shape[self.dim] = -1
            shape.insert(self.dim + 1, group_size)
        
        grouped_input = input.view(*shape)
        sort_dim = self.dim if self.dim == -1 else self.dim + 1
        sorted_grouped_input, _ = torch.sort(grouped_input, dim=sort_dim, descending=True, stable=True)
        sorted_input = sorted_grouped_input.view(*list(input.shape))
        
        return sorted_input

    def forward(self, input: Tensor) -> Tensor:
        if input.is_complex():
            input_cat = torch.cat([input.real, input.imag], dim=-1)
            input_sorted = self.groupsort(input_cat)
            split_size = input_sorted.shape[-1] // 2
            real_sorted, imag_sorted = torch.split(input_sorted, split_size, dim=-1)
            return torch.complex(real_sorted, imag_sorted)
        else:
            return self.groupsort(input)


# Anil, C., Lucas, J., & Grosse, R. (2019). 
# Sorting Out Lipschitz Function Approximation. 
# In Proceedings of the 36th International Conference on Machine Learning (pp. 291–301). PMLR.
# If each group contains only 2 elements, then GroupSort is equivalent to MaxMix.
class MaxMin(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        super(MaxMin, self).__init__()
        self.dim = dim

    def process_maxmin_size(self, input: Tensor, num_units: int, dim: int = -1) -> Tensor:
        size = list(input.size())
        num_channels = size[dim]

        assert num_channels % self.num_units == 0, \
            f"Number of channels ({num_channels}) is not a multiple of num_units ({self.num_units})"
        
        size[dim] = -1
        if dim == -1:
            size += [num_channels // num_units]
        else:
            size.insert(dim+1, num_channels // num_units)
        return size

    def maxmin(self, input: Tensor) -> Tensor:
        original_shape = input.shape
        num_units = input.size(self.dim) // 2
        size = self.process_maxmin_size(input, num_units, self.dim) 
        sort_dim = self.dim if self.dim == -1 else self.dim + 1
        
        mins = torch.min(input.view(*size), sort_dim, keepdim=True)[0]
        maxes = torch.max(input.view(*size), sort_dim, keepdim=True)[0]
        maxmin = torch.cat((maxes, mins), dim=sort_dim).view(original_shape)
            
        return maxmin

    def forward(self, input: Tensor) -> Tensor:
        if input.is_complex():
            input_cat = torch.cat([input.real, input.imag], dim=-1)
            input_sorted = self.maxmin(input_cat)
            split_size = input_sorted.shape[-1] // 2
            real_sorted, imag_sorted = torch.split(input_sorted, split_size, dim=-1)
            return torch.complex(real_sorted, imag_sorted)
        else:
            return self.maxmin(input)


# Artem Chernodub, & Dimitri Nowicki. (2016). 
# Norm-preserving Orthogonal Permutation Linear Unit Activation Functions (OPLU).
# MaxMin is also called Orthogonal Permutation Linear Unit (OPLU).
class OPLU(MaxMin):
    def __init__(self, dim: int = -1) -> None:
        super(OPLU, self).__init__(dim = dim)