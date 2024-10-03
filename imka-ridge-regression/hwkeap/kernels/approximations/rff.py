import math

import torch

from hwkeap.kernels import RBF, ArcCosine
from hwkeap.kernels.approximations.base import BaseApproximator
from hwkeap.utils.f import heaviside


class RandomFourierFeatures(BaseApproximator):
    """Random Fourier Features (RFF) for kernel approximation.

    Args:
        kernel : kernel object to be approximated.
        s : number of features for the approximations.
        d : number of original features.
        device : torch device
        trunc : flag to sample from a gaussian normal truncated at [-3, 3]
    """

    def __init__(self, kernel, s, d, device, trunc=True):
        super().__init__(kernel, s, d, device)
        if trunc:trunc_norm = torch.nn.init.trunc_normal_(torch.zeros((self.s, self.d), device=device), a=-3, b=3)
        else: trunc_norm = torch.randn(self.s, self.d, device=self.device)  # un-truncated version

        if isinstance(self.k, RBF):
            W_ = math.sqrt(2.0 * self.k.gamma) * trunc_norm
            b_ = 2 * math.pi * torch.randn(self.s, device=self.device)
            self.f = [torch.cos, torch.sin]
            self.h = lambda x: 1
        elif isinstance(self.k, ArcCosine):
            W_ = trunc_norm
            b_ = torch.zeros(self.s, device=self.device)

            def app(y): return heaviside(y) * (y**self.k.n)

            self.f = [app]
            self.h = lambda x: 2**0.5

        self.linear_layer.weight = torch.nn.Parameter(W_.float())
        self.linear_layer.bias = torch.nn.Parameter(b_.float())

    def __str__(self) -> str:
        """toString method

        Returns:
            str: string description of the object
        """
        return "rff"
