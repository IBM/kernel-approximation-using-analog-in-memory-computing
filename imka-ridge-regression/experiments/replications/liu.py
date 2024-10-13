import pickle
import time
from collections import defaultdict

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib.gridspec import SubplotSpec

from hwkeap.kernels import RBF, ArcCosine
from hwkeap.kernels.approximations import (
    OrthogonalRandomFeatures,
    RandomFourierFeatures,
    StructuredOrthogonalFeatures,
)
from hwkeap.utils.dataload import load_data_tensors
from hwkeap.utils.f import accuracy, approx_loss
from hwkeap.utils.misc import device, fix_random, join_datasets
from hwkeap.utils.plot import plt

CONFIG_PATH = "experiments/replications/config_liu.yml"


def main():
    config = edict(yaml.safe_load(open(CONFIG_PATH)))
    config.device = device

    trainset, valset, testset = load_data_tensors(config)
    testset = join_datasets(valset, testset)
    d = trainset[0].shape[-1]

    kernels = [RBF(gamma=0.05), ArcCosine(n=0), ArcCosine(n=1)]

    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    elp = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    err = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

    for kernel in kernels:
        print(f"Experiments on {kernel} kernel...")
        for s in config.hidden_features:
            for seed in config.seeds:
                for name in config.rf:
                    fix_random(seed)
                    rf = __init_rf(name, kernel, s, d, logscale=True, device=device)
                    eacc, eelp, eerr = __experiment(
                        rf, kernel, trainset, testset, device
                    )
                    acc[name][str(kernel)][s].append(eacc)
                    elp[name][str(kernel)][s].append(eelp)
                    err[name][str(kernel)][s].append(eerr)

    __plot_result(acc, elp, err, kernels, config)


def __approx_error(X, rff, kernel):
    X_sub = X[:1000, :]
    K = kernel(X_sub, X_sub)
    start = time.time()
    X = rff(X_sub)
    K_approx = X @ X.T
    elapsed = time.time() - start
    return approx_loss(K, K_approx), elapsed


def __init_rf(name, kernel, s, d, device, logscale=True):
    features = 2**s * d if logscale else s * d
    if name == "rff":
        return RandomFourierFeatures(kernel, features, d, device)
    elif name == "orf":
        return OrthogonalRandomFeatures(kernel, features, d, device)
    elif name == "sorf":
        return StructuredOrthogonalFeatures(kernel, features, d, device)


def __experiment(approx, kernel, trainset, testset, device):
    in_feat = approx.s * 2 if str(kernel) == "rbf" else approx.s
    clf = RidgeClassifier(alpha=0.5, in_features=in_feat + 1, out_features=1).to(device)
    clf.fit(approx(trainset[0]), trainset[1])
    inference_model = torch.nn.Sequential(approx, clf)
    acc = accuracy(inference_model(testset[0]), testset[1])
    app_error, elapsed = __approx_error(testset[0], approx.forward, kernel)
    return (acc, elapsed, app_error)


def __plot_result(acc, elp, err, kernels, config):
    def to_dict(d):
        if isinstance(d, defaultdict):
            return dict((k, to_dict(v)) for k, v in d.items())
        return d

    with open("experiments/hardware/resources/acc.pkl", "wb") as handle:
        pickle.dump(to_dict(acc), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("experiments/hardware/resources/elp.pkl", "wb") as handle:
        pickle.dump(to_dict(elp), handle, protocol=pickle.HIGHEST_PROTOCOL)
    with open("experiments/hardware/resources/err.pkl", "wb") as handle:
        pickle.dump(to_dict(err), handle, protocol=pickle.HIGHEST_PROTOCOL)

    fig, axs = plt.subplots(len(kernels), 2)
    fig.set_size_inches(18, 20)
    for (ax1, ax2), kernel in zip(axs, kernels):
        for technique in ["rff", "orf", "sorf"]:

            def mean(x):
                return np.array(
                    [
                        np.mean([x[technique][str(kernel)][n]])
                        for n in config.hidden_features
                    ]
                )

            def std(x):
                return np.array(
                    [
                        np.std([x[technique][str(kernel)][n]])
                        for n in config.hidden_features
                    ]
                )

            ax1.errorbar(
                x=config.hidden_features,
                y=mean(err),
                yerr=std(err),
                marker=".",
                capsize=6,
            )
            # ax2.errorbar(
            #     x=config.hidden_features,
            #     y=mean(elp),
            #     yerr=std(elp),
            #     marker=".",
            #     capsize=6,
            # )
            ax2.errorbar(
                x=config.hidden_features,
                y=mean(acc),
                yerr=std(acc),
                marker=".",
                capsize=6,
            )

        ax1.set_xlabel("$\log_2(s/d)$", fontsize=12)
        # ax2.set_xlabel("$\log_2(s/d)$", fontsize=12)
        ax2.set_xlabel("$\log_2(s/d)$", fontsize=12)
        ax1.set_ylim(bottom=0)
        # ax2.set_ylim(bottom=0)
        ax1.set_ylabel("approximation error", fontsize=12)
        # ax2.set_ylabel("time cost", fontsize=12)
        ax2.set_ylabel("clf accuracy", fontsize=12)

    def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
        # https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
        ker2title = {
            "rbf": "Radial Basis Function",
            "arccos0": "ArcCosine$_0$",
            "arccos1": "ArcCosine$_1$",
        }
        row = fig.add_subplot(grid)
        row.set_title(f"{ker2title[title]}\n", fontweight="semibold")
        row.set_frame_on(False)
        row.axis("off")

    grid = plt.GridSpec(len(kernels), 3)
    for i in range(len(kernels)):
        create_subtitle(fig, grid[i, ::], str(kernels[i]))
    lgd = fig.legend(
        labels=[
            "Random Fourier Features",
            "Orthogonal Random Features",
            "Structured Orthogonal Random Features",
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, 0),
        ncol=1,
        bbox_transform=fig.transFigure,
    )
    plt.subplots_adjust(hspace=0.34)
    fig.tight_layout()
    plt.savefig(
        "resources/replications/liu.png",
        bbox_extra_artists=(lgd,),
        bbox_inches="tight",
    )


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


if __name__ == "__main__":
    main()
