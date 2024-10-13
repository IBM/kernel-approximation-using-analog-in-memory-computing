import torch


def acos_safe(X, eps=1e-20):
    """Compute pytorch arccos safely, avoiding nan issue https://github.com/pytorch/pytorch/issues/8069.


    Args:
        X (tensor): input matrix
        eps (double, optional): precision. Defaults to 1e-20.

    Returns:
        tensor: returns the cos^-1(X).
    """
    return torch.acos(torch.clamp(X, -1 + eps, 1 - eps))


def heaviside(X):
    """Heaviside function.

    Args:
        x (tensor): input tensor

    Returns:
        tensor: output tensor.
    """
    return torch.heaviside(X.double(), torch.tensor(0.0).double())


def approx_loss(K, K_hat):
    """Compute approximation loss between original and approximate kernels.

    Args:
        K (tensor): original kernel
        K_hat (tensor): approximated kernel

    Returns:
        float: approximation loss
    """
    return float(torch.norm(K - K_hat, p="fro") / torch.norm(K, p="fro"))


def accuracy(y_pred, y_true):
    """Compute the accuracy given predictions and gold truth.

    Args:
        y_pred (tensor): predictions.
        y_true (tensor): gold labels.

    Returns:
        float: returns the accuracy on the predictions.
    """
    y_pred = torch.round(torch.clamp(y_pred, min=0, max=1))
    return (torch.sum(torch.squeeze(y_pred) == y_true) / y_pred.shape[0]).item()
