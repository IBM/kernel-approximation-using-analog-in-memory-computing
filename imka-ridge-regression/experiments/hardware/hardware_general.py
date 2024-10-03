import os
import warnings
from collections import defaultdict
from itertools import product

import dill
import matplotlib.patches as mpatches
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib.gridspec import SubplotSpec
from matplotlib.lines import Line2D
from resources.models import ClassificationModel, RidgeClassifier
from tqdm import tqdm

from hwkeap.kernels.approximations import FavorPlus, OrthogonalRandomFeatures, RandomFourierFeatures, StructuredOrthogonalFeatures
from hwkeap.utils.dataload import load_data_tensors
from hwkeap.utils.f import accuracy, approx_loss
from hwkeap.utils.misc import fix_random, init_kernel, join_datasets
from hwkeap.utils.parsing import parse_args_hardware
from hwkeap.utils.plot import plot_color_cycle, plt



def main():
    # Parse args.
    args = parse_args_hardware()
    # Load experiment configuration.
    config = edict(yaml.safe_load(open(args.config)))
    config.device = torch.device("cpu")
    # Load training, vaidation and test data.
    trainset, valset, testset = load_data_tensors(config)
    # Join validation and test sets for testing.
    testset = join_datasets(valset, testset)
    # Initialize kernels to be tested.
    kernels = [init_kernel(k) for k in config.kernels]
    # Initialize output containers.
    acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    err = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fp_acc = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    fp_err = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    # Kernel sweep.
    for n_sampled_features in tqdm(config.hidden_features):
        for kernel in kernels:
            # Inner hyper-parameters sweep.
            hyp_sweep = product(config.seeds, config.rf)
            for seed, rf_name in hyp_sweep:
                fix_random(seed)
                # Initialize random feature sampler.
                experiment = Experiment(rf_name=rf_name,kernel=kernel,s=n_sampled_features,d=testset[0].shape[-1],config=config,device=config.device,)
                # Run the experiment.
                # eacc, eerr = experiment.run(trainset=trainset, testset=testset)
                facc, ferr = experiment.evaluate_fp(trainset=trainset, testset=testset)
                # Log the results.
                # acc[rf_name][str(kernel)][n_sampled_features].append(eacc)
                # err[rf_name][str(kernel)][n_sampled_features].append(eerr)
                fp_acc[rf_name][str(kernel)][n_sampled_features].append(facc)
                fp_err[rf_name][str(kernel)][n_sampled_features].append(ferr)
            # Plot final results.
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                warnings.simplefilter("ignore", category=UserWarning)
                plot(acc, err, fp_acc, fp_err, kernels, config)
                save(acc, err, fp_acc, fp_err, config)


class Experiment:
    """Experiment class.

    Each instance of this class represents a single experiment on a specific configuration
    of parameters (e.g. number of sampled features or kernel used).
    """

    def __init__(self, rf_name, kernel, s, d, config, device) -> None:
        self.config = config
        self.kernel = kernel
        self.device = device
        self.d = d
        self.s = int(2**s * d) - s
        self.rf = self.__init_rf(rf_name,kernel,s,d,logscale=True,device=device,)

    def __init_rf(self, approximation, kernel, s, d, device, logscale=True):
        """Initial random features approximators.

        Args:
            approximation (str): approximation type.
            kernel (object): original kernel of which we want to compute the approx.
            s (int): number of sampled features.
            d (int): number of input features.
            device (torch.device): torch device.
            logscale (bool, optional): whether s stands for the log(s/d). Defaults to True.

        Returns:
            object: random feature approximation object.
        """
        features = int(2**s * d) - s if logscale else s
        if approximation == "rff": return RandomFourierFeatures(kernel, features, d, device=device)
        elif approximation == "orf": return OrthogonalRandomFeatures(kernel, features, d, device=device)
        elif approximation == "sorf": return StructuredOrthogonalFeatures(kernel, features, d, device=device)
        elif approximation == "favor+": return FavorPlus(kernel, features, d, device, ort=True, funct="pos")

    
    def run(self, trainset, testset):
        # Removed code for hardware experiments
        return None, None

    def __approximation_error(self, model, X, platform=None, subsample_size=1000):
        # Subsample the test dataset.
        X_sub = X[:subsample_size, :]
        # Expand testset for approximated features.
        X_sub_rep = X_sub
        # Compute real kernel O(n^2).
        K = self.kernel(X_sub, X_sub)
        # Compute approximations O(n).
        if platform is not None: X = model(X_sub_rep, platform).squeeze()
        else: X = model(X_sub_rep)
        # Compute approximated kernel matrix.
        K_approx = X @ X.T
        # Log end time.
        return approx_loss(K, K_approx)

    def __accuracy(self, model, X, y):
        return accuracy(model(X), y)

    def evaluate_fp(self, trainset, testset):
        err = self.__approximation_error(self.rf, testset[0])
        clf = RidgeClassifier(alpha=0.5, in_features=self.rf.s + 1, out_features=1).to(self.device)
        clf.fit(self.rf(trainset[0]), trainset[1])
        clf_model = ClassificationModel([self.rf, clf]).eval()
        acc = self.__accuracy(model=clf_model, X=testset[0], y=testset[1])
        return acc, err


def plot(acc, err, facc, ferr, kernels, config):
    """Plot experimental results.

    Args:
        acc (defaultdict): logged downstream accuracy
        err (defaultdict): logged approximation error
        kernels (list): kernels tested
        config (defaultdict): experiment parameters
    """
    fig, axs = plt.subplots(len(kernels), 2)
    fig.set_size_inches(15, 14)

    for (ax1, ax2), kernel in zip(axs, kernels):
        cyc = iter(plot_color_cycle)

        for technique in ["rff", "orf", "sorf"]:
            color = next(cyc)["color"]

            def mean(x):return np.array([np.mean([x[technique][str(kernel)][n]])for n in config.hidden_features])

            def std(x): return np.array([np.std([x[technique][str(kernel)][n]])for n in config.hidden_features])

            # ANALOG
            # ax1.errorbar(
            #     x=config.hidden_features,
            #     y=mean(err),
            #     yerr=std(err),
            #     marker=".",
            #     linestyle="dashed",
            #     capsize=6,
            #     color=color,
            # )
            # ax2.errorbar(
            #     x=config.hidden_features,
            #     y=mean(acc),
            #     yerr=std(acc),
            #     marker=".",
            #     linestyle="dashed",
            #     capsize=6,
            #     color=color,
            # )
            # FP
            if config.fp_reference:
                ax1.errorbar(x=config.hidden_features,y=mean(ferr),yerr=std(ferr),marker=".",capsize=6,color=color,)
                ax2.errorbar(x=config.hidden_features,y=mean(facc),yerr=std(facc),marker=".",capsize=6,color=color,)

        ax1.set_xlabel("$\log_2(s/d)$", fontsize=12)
        ax2.set_xlabel("$\log_2(s/d)$", fontsize=12)
        ax1.set_ylim(bottom=0)
        if config.dataset == "skin": ax2.set_ylim(top=1)
        ax1.set_ylabel("approximation error", fontsize=12)
        ax2.set_ylabel("clf accuracy", fontsize=12)

    def create_subtitle(fig: plt.Figure, grid: SubplotSpec, title: str):
        # https://stackoverflow.com/questions/27426668/row-titles-for-matplotlib-subplot
        row = fig.add_subplot(grid)
        row.set_title(f"{title}\n", fontweight="semibold")
        row.set_frame_on(False)
        row.axis("off")

    grid = plt.GridSpec(len(kernels), 2)
    for i in range(len(kernels)): 
        create_subtitle(fig, grid[i, ::], str(kernels[i]))

    rff_patch = mpatches.Patch(color="#9b59b6", label="Random Fourier Features")
    orf_patch = mpatches.Patch(color="#3498db", label="Orthogonal Random Features")
    sorf_patch = mpatches.Patch(color="#95a5a6", label="Structured Orthogonal Random Features")
    dash = Line2D([0], [0], label="HW", color="k", linestyle="dashed")
    soli = Line2D([0], [0], label="FP", color="k", linestyle="solid")
    lgd = plt.legend(handles=[rff_patch, orf_patch, sorf_patch, dash, soli],loc="lower center",bbox_to_anchor=(0.5, 0),ncol=2,bbox_transform=fig.transFigure,)
    fig.tight_layout()
    mode = "emulated"
    folder = f"resources/hardware/{config.dataset}"
    os.makedirs(folder, exist_ok=True)
    filename = os.path.join(folder, f"{mode}.png")
    plt.savefig(filename,bbox_extra_artists=(lgd,),bbox_inches="tight",)
    plt.close("all")


def save(acc, err, fp_acc, fp_err, config):
    """
    Save experiment results as pickles.
    """
    folder_name = f"resources/hardware/{config.dataset}/"
    os.makedirs(folder_name, exist_ok=True)
    mode = "fp"
    file_name = os.path.join(folder_name, f"{mode}_acc.pkl")
    with open(file_name, "wb") as file: dill.dump(dict(fp_acc), file)
    file_name = os.path.join(folder_name, f"{mode}_err.pkl")
    with open(file_name, "wb") as file: dill.dump(dict(fp_err), file)


if __name__ == "__main__":
    main()
