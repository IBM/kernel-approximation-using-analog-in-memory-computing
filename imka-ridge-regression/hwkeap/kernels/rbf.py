import torch


class RBF:
    """Radial basis function kernel (aka squared-exponential kernel) implementation.

    Args:
        gamma : kernel coefficient
    """

    def __init__(self, gamma: float = 1) -> None:
        self.gamma = gamma

    def __call__(self, X, Y=None):
        """Compute and return K = k(X, Y).

        Args:
            X (tensor): first operand
            Y (tensor): second operand; if None, k(X, X) is computed

        Returns:
            tensor: returns the kernel matrix.
        """
        if Y is None: Y = X
        K = torch.exp(-self.gamma * torch.cdist(X, Y, p=2) ** 2)
        return K

    def __str__(self) -> str:
        """toString method

        Returns:
            str: returns a string description of the object
        """
        return "rbf"
