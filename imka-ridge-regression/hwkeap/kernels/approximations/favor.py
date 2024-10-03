import math

import torch

from hwkeap.kernels import Softmax
from hwkeap.kernels.approximations.base import BaseApproximator
from hwkeap.utils.misc import get_orthogonal_matrix


class FavorPlus(BaseApproximator):
    def __init__(self,kernel,s: int,d: int,device: torch.device,ort: bool = True,reg: bool = False,fast: bool = True,funct: str = "pos",):

        assert isinstance(
            kernel, Softmax
        ), "Favor can only approximate the Softmax kernel"
        super().__init__(kernel, s, d, device)

        # Initialize functions.
        if funct == "trig":

            def sin(x): return torch.sin(2 * math.pi * x)

            def cos(x): return torch.cos(2 * math.pi * x)

            self.f = [sin, cos]
            self.h = lambda x: torch.exp(torch.square(x).sum(axis=-1, keepdims=True) / 2)
        elif funct == "relu":
            self.f = [torch.relu]
            self.h = lambda x: torch.exp(-torch.square(x).sum(axis=-1, keepdims=True) / 2)
        elif funct == "pos":

            def ep(x):return torch.exp(x)

            def em(x):return torch.exp(-x)

            self.f = [ep, em]
            self.h = lambda x: (2**-0.5) * torch.exp(-torch.square(x).sum(axis=-1, keepdims=True) / 2)

        # Initialize weight matrix.
        if ort: W_ = get_orthogonal_matrix(s, d, device, fast=fast, reg=reg, trunc=True)
        else: W_ = torch.randn(s, d, device=device)
        b_ = torch.zeros(self.s, device=self.device)

        # MVM layer initialization.
        self.linear_layer.weight = torch.nn.Parameter(W_.float())
        self.linear_layer.bias = torch.nn.Parameter(b_.float())

    def __str__(self) -> str:
        """toString method

        Returns:
            str: string description of the object
        """
        return f"favor+{'_ort' if self.ort else ''}"
