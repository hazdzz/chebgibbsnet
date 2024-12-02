import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import utils.functional as uF
from torch import Tensor


class dSiLU(nn.Module):
    def forward(self, input: Tensor) -> Tensor:
        return uF.d_silu(input)


class TanhExp(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(TanhExp, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return uF.tanhexp(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class SinSig(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(SinSig, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return uF.sinsig(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class SquaredReLU(nn.Module):
    __constants__ = ['inplace']
    inplace: bool

    def __init__(self, inplace: bool = False):
        super(SquaredReLU, self).__init__()
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return uF.sqrrelu(input, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = 'inplace=True' if self.inplace else ''
        return inplace_str


class Squareplus(nn.Module):
    __constants__ = ['bias', 'threshold', 'inplace']
    bias: float
    threshold: float
    inplace: bool

    def __init__(self, bias: float = 4 * (math.log(2) ** 2), threshold: float = 20.0, inplace: bool = False):
        super(Squareplus, self).__init__()
        self.bias = bias
        self.threshold = threshold
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return uF.squareplus(input, self.bias, self.threshold, inplace=self.inplace)

    def extra_repr(self) -> str:
        return f'bias={self.bias}, threshold={self.threshold}, inplace={self.inplace}'

   
class DiracReLU(nn.Module):
    __constants__ = ['beta', 'inplace']
    beta: float
    inplace: bool

    def __init__(self, bias: float = 1.0, inplace: bool = False):
        super(DiracReLU, self).__init__()
        self.bias = bias
        self.inplace = inplace

    def forward(self, input: Tensor) -> Tensor:
        return uF.diracrelu(input, self.bias, inplace=self.inplace)

    def extra_repr(self) -> str:
        inplace_str = ', inplace=True' if self.inplace else ''
        return 'bias={}{}'.format(self.bias, inplace_str)

   
class SMU(nn.Module):
    def __init__(self, alpha: float = 0.0, beta: float = 1.0, inplace: bool = False) -> None:
        super(SMU, self).__init__()
        self.alpha = alpha
        self.beta = nn.Parameter(torch.tensor(beta))
        self.inplace = inplace
        
    def forward(self, input: Tensor) -> Tensor:
        return uF.smu(input, self.alpha, self.beta, self.inplace)
