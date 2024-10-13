import math

import torch

from hwkeap.kernels import RBF, ArcCosine
from hwkeap.kernels.approximations.base import BaseApproximator
from hwkeap.utils.f import heaviside
from hwkeap.utils.misc import get_orthogonal_matrix


class StructuredOrthogonalFeatures(BaseApproximator):
    """Structured Orthogonal Random Features (ORF) for kernel approximation.

    Args:
        kernel : kernel object to be approximated
        s : number of features for the approximations
        d (int): number of original features.
    """

    def __init__(self, kernel, s, d, device):
        # Init number of features, which equals to the next power of 2 from s.
        super().__init__(kernel, s, d, device)
        # Build orthogonal matrix.
        W_ = get_orthogonal_matrix(s, d, device, fast=True)
        # Kernel dependent initialization.
        if isinstance(self.k, RBF):
            W_ = math.sqrt(2.0 * self.k.gamma) * W_[: self.s, : self.d]
            b_ = 2 * math.pi * torch.randn(self.s, device=self.device)
            self.f = [torch.cos, torch.sin]
            self.h = lambda x: 1
        elif isinstance(self.k, ArcCosine):
            W_ = W_[: self.s, : self.d]
            b_ = torch.zeros(self.s, device=self.device)

            def f(y): return heaviside(y) * (y**self.k.n)

            self.f = [f]
            self.h = lambda x: 2**0.5
        # MVM layer initialization.
        self.linear_layer.weight = torch.nn.Parameter(W_.float())
        self.linear_layer.bias = torch.nn.Parameter(b_.float())

    def __str__(self) -> str:
        """toString method

        Returns:
            str: string description of the object
        """
        return "sorf"
