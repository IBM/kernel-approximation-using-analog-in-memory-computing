import math
import random

import numpy as np
import torch
from hadamard_transform import hadamard_transform
from scipy.stats import chi

from hwkeap.kernels import RBF, ArcCosine, Softmax

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def join_datasets(d1, d2):
    """Join datasets d1 and d2.

    Args:
        d1 (tuple): first dataset tuple (X, y)
        d2 (tuple): second dataset tuple (X, y)

    Returns:
        tuple: returns new dataset tuple (X, y)
    """
    X = torch.concatenate((d1[0], d2[0]))
    y = []
    if len(d1) > 1 and len(d2) > 1:
        y = torch.concatenate((d1[1], d2[1]))
    return X, y


def shuffle_and_split(X, Y):
    n_samp = X.shape[0]
    idx = torch.randperm(n_samp)
    X_shuffle = X[idx]
    Y_shuffle = Y[idx]
    train = n_samp // 2
    val = (n_samp - train) // 2 + train
    X_train, X_val, X_test = X_shuffle[:train], X_shuffle[train:val], X_shuffle[val:]
    Y_train, Y_val, Y_test = Y_shuffle[:train], Y_shuffle[train:val], Y_shuffle[val:]
    assert (n_samp - X_train.shape[0] - X_val.shape[0] - X_test.shape[0]) == 0
    return [(X_train, Y_train), (X_val, Y_val), (X_test, Y_test)]


def init_kernel(name):
    if name == "rbf":return RBF(gamma=0.05)
    if name == "arccos0":return ArcCosine(n=0)
    if name == "arccos1":return ArcCosine(n=1)
    if name == "arccos2":return ArcCosine(n=2)
    if name == "softmax":return Softmax()


def fix_random(seed):
    """Fix random behaviours of the libraries.
       Allows reproducibility.

    Args:
        seed (int): random seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)


def next_power_2(x):
    """Returns the next power of 2 from x.

    Args:
        x (float): input number

    Returns:
        float: next power of two
    """
    if not isinstance(x, int):
        raise TypeError("x is not integer.")
    if x < 0:
        raise ValueError("x is negative.")
    return 1 if x == 0 else 2 ** (x - 1).bit_length()


def get_orthogonal_matrix(s, d, device, trunc=True, fast=True, reg=False):
    """Build an orthogonal gaussian random matrix using Gram-Schmidt orthogonalization.

    Args:
        s (int): number of columns (eq. # of sampled features for rf)
        d (int): number of rows (eq. # of orginal sample features)
        device (torch.device): device
        trunc (bool): flag to use a gaussian normal truncated at [-3, 3]

    Returns:
        torch.tensor: orthogonal matrix
    """

    def get_square_block(size):
        if trunc: G = torch.nn.init.trunc_normal_(torch.zeros((size, size), device=device), a=-3, b=3)
        else: G = torch.randn(size, size, device=device)
        q, _ = torch.linalg.qr(G)
        return q.T

    def get_square_block_fast(size):
        size = next_power_2(size)
        ds = 2 * (torch.rand(size=(size, 3), device=device) < 0.5) - 1
        HD1 = hadamard_transform(torch.diag(ds[:, 0]))
        HD2 = hadamard_transform(torch.diag(ds[:, 1]))
        HD3 = hadamard_transform(torch.diag(ds[:, 2]))
        return math.sqrt(size) * (HD1 @ HD2 @ HD3)

    num_full_blocks = s // d
    blocks = [get_square_block_fast(d) if fast else get_square_block(d)for _ in range(num_full_blocks)]

    remaining_rows = s - num_full_blocks * d
    if remaining_rows:
        q = get_square_block_fast(d) if fast else get_square_block(d)
        blocks.append(q[:remaining_rows])
    mat = torch.vstack(blocks)

    if not fast and not reg:
        # Re-normalize the resulting matrix.
        S = torch.diag(torch.tensor(chi.rvs(df=d, size=(s,)), device=device)).float()
        mat = S @ mat
    elif not fast:
        # SMREG matrix
        mat /= math.sqrt(num_full_blocks + remaining_rows / d)

    return mat
