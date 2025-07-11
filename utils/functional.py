import math
import torch
import torch.nn.functional as F

from torch import Tensor
from typing import List, Optional, Tuple, Union


def complex_fcaller(funtional_handle, *args):
    return torch.complex(funtional_handle(args[0].real, *args[1:]), funtional_handle(args[0].imag, *args[1:]))


def smu(input: Tensor, alpha: float, beta: Tensor, inplace: bool = False) -> Tensor:
    if inplace:
        input = ((1.0 + alpha) * input + (1.0 - alpha) * input * torch.erf(beta * (1.0 - alpha) * input)) / 2.0
        return input
    else:
        smu = ((1.0 + alpha) * input + (1.0 - alpha) * input * torch.erf(beta * (1.0 - alpha) * input)) / 2.0
        return smu


def calu(input: Tensor) -> Tensor:
    return input * (torch.arctan(input) / math.pi + 0.5)


def siglog(input: Tensor) -> Tensor:
    return torch.div(input, 1.0 + torch.abs(input))


def c_cardioid(input: Tensor) -> Tensor:
    phase = torch.angle(input)
    return 0.5 * torch.mul(1.0 + torch.cos(phase), input)


def igaussian(input: Tensor, sigma: float = 1.0) -> Tensor:
    norm_sq = torch.abs(input) ** 2
    g = 1.0 - torch.exp(-norm_sq / (2.0 * (sigma ** 2)))
    arg = torch.angle(input)
    n = torch.exp(1j * arg)
    return g * n


def c_sigmoid(input: Tensor) -> Tensor:
    if input.is_complex():
        return torch.sigmoid(input.real) + 1j * torch.sigmoid(input.imag)
    else:
        return torch.sigmoid(input)
    

def c_tanh(input: Tensor) -> Tensor:
    if input.is_complex():
        return torch.tanh(input.real) + 1j * torch.tanh(input.imag)
    else:
        return torch.tanh(input)


def c_softsign(input: Tensor) -> Tensor:
    if input.is_complex():
        return F.softsign(input.real) + 1j * F.softsign(input.imag)
    else:
        return F.softsign(input)


def c_relu(input: Tensor, inplace: bool = False) -> Tensor:
    if input.is_complex():
        return torch.complex(F.relu(input.real, inplace=inplace), F.relu(input.imag, inplace=inplace))
    else:
        return F.relu(input, inplace=inplace)


def mod_relu(input: Tensor, bias: float = -math.sqrt(2.0), rounding_mode: str = None, inplace: bool = False) -> Tensor:
    if input.is_complex():
        magnitude = torch.abs(input)
        phase = torch.angle(input)
        euler_phase = torch.cos(phase) + 1j * torch.sin(phase)
        if inplace:
            input = torch.mul(F.relu(magnitude + bias, inplace=False), euler_phase).type(input.type())
            return input
        else:
            mod_relu = torch.mul(F.relu(magnitude + bias, inplace=inplace), euler_phase).type(input.type())
            return mod_relu
    else:
        return F.relu(input, inplace=inplace)


def z_relu(input: Tensor, inplace: bool = False) -> Tensor:
    if input.is_complex():
        if inplace:
            mask = torch.zeros_like(input)
            input = torch.where(torch.angle(input) < 0.0, mask, input)
            input = torch.where(torch.angle(input) > (math.pi / 2.0), mask, input)
            return input
        else:
            mask = torch.zeros_like(input)
            z_relu = torch.where(torch.angle(input) < 0.0, mask, input)
            z_relu = torch.where(torch.angle(z_relu) > (math.pi / 2.0), mask, z_relu)
            return z_relu
    else:
        return F.relu(input, inplace=inplace)
    

def c_leaky_relu(input: Tensor, negative_slope: float = 0.01, inplace: bool = False) -> Tensor:
    if input.is_complex():
        return torch.complex(F.leaky_relu(input=input.real, negative_slope=negative_slope, inplace=inplace), 
                             F.leaky_relu(input=input.imag, negative_slope=negative_slope, inplace=inplace))
    else:
        return F.leaky_relu(input=input, negative_slope=negative_slope, inplace=inplace)