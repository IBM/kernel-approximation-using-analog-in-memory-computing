import numpy as np
import torch

from hwkeap.utils.f import acos_safe


class ArcCosine:
    """Exact arc-cosine kernel implementation. Supports arc-cosine kernels up to n=2.

    Args:
        n : order of the kernel
    """

    def __init__(self, n) -> None:
        assert n <= 2, "ArcCos kernel support only up to degree 2"
        self.n = n

    def __call__(self, X, Y=None):
        """Compute the arccos kernel of degree n between matrices X and Y.
           If Y is None, compute the kernel k(X,X).

        Args:
            X (tensor): first matrix
            Y (tensor, optional): second matrix. Defaults to None.

        Returns:
            tensor: returns k(X, Y) if Y is not None, k(X,X) othw.
        """
        if Y is None:
            Y = X
        norm_x = torch.norm(X, p=2, dim=1)
        norm_y = torch.norm(Y, p=2, dim=1)
        dot_prod = X @ Y.T / torch.outer(norm_x, norm_y)
        theta = acos_safe(dot_prod)
        if self.n == 0: J_theta = np.pi - theta
        elif self.n == 1: J_theta = torch.sin(theta) + (np.pi - theta) * torch.cos(theta)
        elif self.n == 2: J_theta = 3 * torch.sin(theta) * torch.cos(theta) + (np.pi - theta) * (1 + 2 * (torch.cos(theta) ** 2))
        K = 1 / np.pi * torch.outer(norm_x, norm_y) ** self.n * J_theta
        return K

    def __str__(self) -> str:
        """toString method

        Returns:
            str: string description of the object
        """
        return "arccos" + str(self.n)
