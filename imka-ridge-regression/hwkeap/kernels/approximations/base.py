from copy import deepcopy

import torch


class BaseApproximator(torch.nn.Module):
    """
    Base layer for kernel approximation.

    Args:
        kernel (obj): kernel to approximate.
        s (int): number of hidden features.
        d (int): number of original features.
    """

    def __init__(self, kernel, s, d, device):
        super().__init__()
        self.k = kernel
        self.s = s
        self.d = d
        self.device = device
        self.linear_layer = torch.nn.Linear(out_features=self.s, in_features=self.d).to(device)
        self.replication_factor = 1.0

    def forward(self, x, platform=None):
        """Forward method.
        Computes the random features for the given input matrix.

        Args:
            X (tensor): input matrix.

        Returns:
            (tensor): returns the transformed feature matrix.
        """
        # MVM projection.
        if platform is not None:
            proj = self.linear_layer(x[None, :, :], platform=platform)[0].squeeze()/ self.replication_factor
        else:
            proj = self.linear_layer(x) / self.replication_factor
        # Define feature extraction function.
        def phi(x, wx):
            return self.h(x)* self.s ** (-0.5)* torch.concatenate([f(wx) for f in self.f],axis=-1,)

        # Compute and return features.
        return phi(x, proj).float()

    def set_linear_layer(self, linear_layer):
        """Set linear layer for the current approximation module.

        Args:
            linear_layer (torch.nn.Linear): new linear layer
        """
        self.linear_layer = linear_layer

    def copy(self):
        """Deep copy this kernel approximation module.

        Returns:
            BaseApproximator : deep copy of self
        """
        return deepcopy(self)
