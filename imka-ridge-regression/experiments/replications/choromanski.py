from collections import defaultdict
from itertools import product

import numpy as np
import torch
from easydict import EasyDict as edict
from tqdm import tqdm

from hwkeap.kernels import Softmax
from hwkeap.kernels.approximations import FavorPlus
from hwkeap.utils.misc import device, fix_random
from hwkeap.utils.plot import plt


def main():
    # Load configuration and dataset.
    config = edict({})
    config.s = range(16, 200, 16)
    config.seeds = list(range(0, 15))
    config.device = device
    # Get original number of features per sample.
    d = 16
    # Instantiate kernel.
    kernel = Softmax()
    # Instantiate logger.
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # Set sweep for (kernel, rf) combinations.
    sweep = list(product(config.s, config.seeds, [True, False], ["trig", "pos"]))
    # Run the sweep.
    for s, seed, ort, funct in tqdm(sweep):
        results[s][ort][funct].append(
            experiment(
                kernel=kernel, s=s, d=d, seed=seed, ort=ort, funct=funct, device=device
            )
        )
    # Plot the results
    plot(results, config)


def experiment(kernel, s, d, seed, ort, funct, device):
    """
    Test the approximation error on a specific combination of hyperparameters.
    """
    # Set experiment seed.
    fix_random(seed)
    # Initialize random features.
    rf = FavorPlus(kernel, s, d, device, ort=ort, fast=False, reg=True, funct=funct)
    # Init random matrices.
    Q = torch.randn(4096, 16)
    K = torch.randn(4096, 16)
    V = torch.randn(4096, 16)
    # Compute vanilla attention.
    A = kernel(Q, K) @ V
    # Compute approx attention.
    seq_len, _ = Q.shape
    Q_prime = rf(Q * (d**-0.25))
    K_prime = rf(K * (d**-0.25))
    D_inv = torch.diag(1 / (Q_prime @ (K_prime.T @ torch.ones(seq_len))))
    A_hat = D_inv @ (Q_prime @ (K_prime.T @ V))
    return float(loss(A, A_hat))


def loss(A, B):
    return torch.square(A - B).mean()


def plot(results, config):
    """
    Plot experiment results.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2)
    fig.set_size_inches(16, 6)

    def mean(x, ort, s, funct):
        return np.array([np.mean(x[n][ort][funct]) for n in s])

    def std(x, ort, s, funct):
        return np.array([np.std(x[n][ort][funct]) for n in s])

    y_label = "MSE error"
    file_name = "resources/replications/choromanski.png"

    # ort
    ax1.set_yscale("log")
    for ort in [False, True]:
        label = "ort" if ort else "iid"
        ax1.plot(
            config.s,
            mean(results, ort, config.s, "trig"),
            marker=".",
            label=label,
        )
        ax1.fill_between(
            config.s,
            mean(results, ort, config.s, "trig") - std(results, ort, config.s, "trig"),
            mean(results, ort, config.s, "trig") + std(results, ort, config.s, "trig"),
            alpha=0.5,
            linewidth=0,
        )
    ax1.set_title("IID vs Orthogonal", fontweight="bold", pad=10)
    ax1.set_xlabel("Sampled features", fontsize=14)
    ax1.set_ylabel(y_label, fontsize=14)
    ax1.margins(x=0.05)
    ax1.legend(ncol=2, loc="upper right")

    # positive
    ax2.set_yscale("log")

    def relu(x):
        return x * (x > 0)

    for funct in ["trig", "pos"]:
        ax2.plot(
            config.s,
            mean(results, funct == "pos", config.s, funct),
            marker=".",
            label=funct,
        )
        ax2.fill_between(
            config.s,
            relu(
                mean(results, funct == "pos", config.s, funct)
                - std(results, funct == "pos", config.s, funct)
            ),
            mean(results, funct == "pos", config.s, funct)
            + std(results, funct == "pos", config.s, funct),
            alpha=0.5,
            linewidth=0,
        )
    ax2.set_title("Trigonometric vs Positive", fontweight="bold", pad=10)
    ax2.set_xlabel("Sampled features", fontsize=14)
    ax2.set_ylabel(y_label, fontsize=14)
    ax2.margins(x=0.05)
    ax2.legend(ncol=2, loc="upper right")
    plt.savefig(
        file_name,
        bbox_inches="tight",
    )


if __name__ == "__main__":
    main()
