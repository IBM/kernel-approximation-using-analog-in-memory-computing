from collections import OrderedDict

import torch

from hwkeap.kernels.approximations.base import BaseApproximator


class RidgeClassifier(torch.nn.Module):
    """
    Ridge classification model.
    """

    def __init__(self, in_features, out_features, alpha=0.0, fit_intercept=True):
        """Ridge classification model.

        Args:
            alpha (float, optional): regularization parameter. Defaults to 0.
            fit_intercept (bool, optional): flag to activate bias fitting. Defaults to True.
        """
        super().__init__()
        self.alpha = alpha
        self.linear_layer = torch.nn.Linear(
            in_features=in_features, out_features=out_features, bias=False
        )
        self.fit_intercept = fit_intercept
        self.fitted = False

    def fit(self, X: torch.tensor, y: torch.tensor) -> None:
        """Fit the classifier on training data.

        Args:
            X (torch.tensor): training data.
            y (torch.tensor): training labels.
        """
        if self.fit_intercept:
            X = torch.cat([torch.ones(X.shape[0], 1, device=X.device), X], dim=1)
        W = (
            torch.inverse(X.T @ X + self.alpha * torch.eye(X.shape[1], device=X.device))
            @ X.T
            @ y.float()
        )
        self.linear_layer.weight = torch.nn.Parameter(W)
        self.fitted = True

    def forward(self, x: torch.tensor):
        """Forward pass.

        Args:
            x (torch.tensor): input data.

        Returns:
            torch.tensor: returns predictions for the input data.
        """
        assert self.fitted, "Model not fitted yet!"
        if self.fit_intercept:
            x = torch.cat([torch.ones(x.shape[0], 1, device=x.device), x], dim=1)
        return self.linear_layer(x)


class ClassificationModel(torch.nn.Sequential):
    def __init__(self, layers, platform=None):
        super().__init__(self.init_modules(layers))
        self.platform = platform

    def init_modules(self, layers):
        assert isinstance(
            layers[0], BaseApproximator
        ), "First layer must be a kernel approximator!"
        modules = OrderedDict()
        modules["sampler"] = layers[0]
        modules["clf"] = layers[1]
        return modules

    def forward(self, input):
        features = self[0](input, self.platform)
        return self[1](features)
