import math
import torch
import torch.nn.functional as F
from torch import Tensor

def d_silu(input: Tensor):
    return torch.sigmoid(input) * (1 + input * (1 - torch.sigmoid(input)))

def tanhexp(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = torch.mul(input, torch.tanh(torch.exp(input))).type(input.type())
        return input
    else:
        tanhexp = torch.mul(input, torch.tanh(torch.exp(input))).type(input.type())
        return tanhexp

def sinsig(input: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = torch.mul(input, torch.sin(0.5 * math.pi * torch.sigmoid(input))).type(input.type())
        return input
    else:
        sinsig = torch.mul(input, torch.sin(0.5 * math.pi * torch.sigmoid(input))).type(input.type())
        return sinsig

def squareplus(input: Tensor, bias: float = 4 * (math.log(2) ** 2), threshold: float = 20.0, inplace: bool = False) -> Tensor:
    if inplace:
        result = input
    else:
        result = input.clone()
    mask = result < threshold
    result[mask] = (torch.sqrt(torch.pow(result[mask], 2) + bias) + result[mask]) / 2

    return result
    
def diracrelu(input: Tensor, beta: float = 1.0, inplace: bool = False) -> Tensor:
    if inplace:
        term1 = input * torch.erf(input / (math.sqrt(2.0) * beta))
        term2 = input
        term3 = math.sqrt(2.0 / math.pi) * beta * torch.exp(-input ** 2 / (2 * beta ** 2))
        input = 0.5 * (term1 + term2 + term3)
        return input
    else:
        term1 = input * torch.erf(input / (math.sqrt(2.0) * beta))
        term2 = input
        term3 = math.sqrt(2.0 / math.pi) * beta * torch.exp(-input ** 2 / (2 * beta ** 2))
        diracrelu = 0.5 * (term1 + term2 + term3)
        return diracrelu
    
def smu(input: Tensor, alpha: float, beta: Tensor, inplace: bool = False):
    if inplace:
        input = ((1 + alpha) * input + (1 - alpha) * input * torch.erf(beta * (1 - alpha) * input)) / 2
        return input
    else:
        smu = ((1 + alpha) * input + (1 - alpha) * input * torch.erf(beta * (1 - alpha) * input)) / 2
        return smu