import torch

from hwkeap.kernels import ArcCosine, rbf
from hwkeap.utils.f import heaviside


class RandomFeatures(torch.nn.Module):
    def __init__(self, kernel, s, d, device):
        super().__init__()
        self.k = kernel
        self.s = s
        self.d = d
        self.device = device
        self.linear_layer = torch.nn.Linear(out_features=self.s, in_features=self.d).to(device)
        W_ = torch.randn(self.s, self.d, device=self.device)
        b_ = torch.zeros(self.s, device=self.device)
        self.linear_layer.weight = torch.nn.Parameter((W_ / torch.max(torch.abs(W_))).float())
        if isinstance(self.k, rbf):
            self.f = [torch.cos, torch.sin]
            self.h = lambda x: 1
        elif isinstance(self.k, ArcCosine):

            def f(y): return heaviside(y) * (y**self.k.n)

            self.f = [f]
            self.h = lambda x: 2**0.5
        self.linear_layer.bias = torch.nn.Parameter(b_.float())
