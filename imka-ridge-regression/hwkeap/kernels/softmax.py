import torch


class Softmax:
    """Exact softmax kernel implementation."""

    def __init__(self,) -> None:
        pass

    def __call__(self, Q, K):
        """Compute the exponential kernel between matrices Q and K.
        The exponential kernel corresponds to the softmax of the
        product between queries and keys. The values are not
        normalized by sqrt(d_k), hence the matrices have to be.

        Args:
            Q (tensor): query matrix.
            K (tensor, optional): key matrix.

        Returns:
            tensor: returns k(Q, K) if Y is not None, k(Q, Q) othw.
        """
        return torch.softmax(Q @ K.T / (Q.shape[-1] ** 0.5), dim=-1)

    def __str__(self) -> str:
        """toString method

        Returns:
            str: string description of the object
        """
        return "softmax"
