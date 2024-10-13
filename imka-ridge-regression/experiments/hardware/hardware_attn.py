import logging
import os
import time
from collections import defaultdict
from itertools import product

import dill
import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from matplotlib.lines import Line2D

from hwkeap.kernels import Softmax
from hwkeap.kernels.approximations import FavorPlus
from hwkeap.utils.dataload import load_data_tensors
from hwkeap.utils.f import approx_loss
from hwkeap.utils.misc import device, fix_random, join_datasets
from hwkeap.utils.plot import plot_color_cycle, plt

CONFIG_PATH = "experiments/hardware/config/config_attn.yml"
SAVE_MAT = False


def add_hermes_args(config):
    config.device = torch.device("cpu")
    config.n_batches = -1
    return config


def main():
    fix_random(0)
    # Load experiment configuration.
    config = edict(yaml.safe_load(open(CONFIG_PATH)))
    config = add_hermes_args(config)
    # Load training, vaidation and test data.
    trainset, valset, testset = load_data_tensors(config)
    # Join validation and test sets for testing.
    testset = join_datasets(valset, testset)
    trainset = join_datasets(trainset, testset)
    # Initialize kernels to be tested.
    kernel = Softmax()
    # Initialize output containers.
    err = defaultdict(lambda: defaultdict(list))
    fp_err = defaultdict(lambda: defaultdict(list))

    # Inner hyper-parameters sweep.

    for n_sampled_features in config.hidden_features:
        hyp_sweep = product(config.seeds, config.rf)
        for seed, rf_name in hyp_sweep:
            fix_random(seed)
            # Initialize random feature sampler.
            experiment = Experiment(kernel=kernel,s=n_sampled_features,d=config.d,config=config,device=device,save_mat=SAVE_MAT,)
            # Run the experiment.
            _, eerr, _, ferr = experiment.run(trainset=trainset, testset=testset)
            # Log the results.
            err[rf_name][n_sampled_features].append(eerr)
            fp_err[rf_name][n_sampled_features].append(ferr)
        # Plot final results.
        plot(err, fp_err, config)
        save(err, fp_err, config)


class AnalogFavor(torch.nn.Module):
    def __init__(self, lin, s):
        super().__init__()
        self.lin = lin
        self.s = s

    def forward(self, input):
        # q, k = input[0], input[1]
        proj_q = self.lin(input[0])
        proj_k = self.lin(input[1])

        def phi(x, proj): return 2**-0.5 * torch.exp(-torch.square(x).sum(axis=-1, keepdims=True) / 2) * self.s ** (-0.5) * torch.concatenate( [torch.exp(proj), torch.exp(-proj)],axis=-1,)
        
        q_prime = phi(input[0], proj_q)
        k_prime = phi(input[1], proj_k)
        return q_prime, k_prime


class Experiment:
    """Experiment class.

    Each instance of this class represents a single experiment on a specific configuration
    of parameters (e.g. number of sampled features or kernel used).
    """

    def __init__(self, kernel, s, d, config, device, n_replicate=1, save_mat=False) -> None:
        self.config = config
        self.kernel = kernel
        self.device = device
        self.d = d
        self.s = s * d - 1
        self.rf = FavorPlus(kernel, self.s, d, device, ort=True, fast=True, funct="pos")
        self.n_replicate = n_replicate
        self.save_mat = save_mat

    def run(self, trainset, testset):
        # Get number of samples.
        Q = trainset[0][:, 0, :, :].to(device)
        K = trainset[0][:, 1, :, :].to(device)
        V = trainset[0][:, 2, :, :].to(device)

        # initiliaze analog model, program weights
        fp_err, fp_elp = self.__attention_approximation_error(model=self.rf, X=(Q, K, V),)
        return None, None, fp_elp, fp_err

    def __attention_approximation_error(self, model, X, platform=None, fp=None):
        """Compute attention error and elapsed time."""
        err, elapsed = [], []
        save_mat = dict()

        for i in range(10):
            save_mat[i] = dict()
            # Get query, key and value vectors.
            Q = X[0][i, :, :]
            K = X[1][i, :, :]
            V = X[2][i, :, :]
            # Compute real attention.
            A = self.kernel(Q, K) @ V
            # Compute analog approximations O(n).
            start = time.time()
            if platform is not None:
                Q_prime = None
                K_prime = None
                while K_prime is None:
                    res = model((Q * (64**-0.25), K * (64**-0.25)), platform)
                    if res: Q_prime, K_prime = res[0][0], res[1][0]
                model.pipeline.reset()
            else:
                Q_prime = model(Q * (64**-0.25))
                K_prime = model(K * (64**-0.25))
            elapsed.append(time.time() - start)
            D_inv = torch.diag(1 / (Q_prime @ (K_prime.T @ torch.ones(Q.shape[0], device=device))))
            A_hat = D_inv @ (Q_prime @ (K_prime.T @ V))

            if self.save_mat:
                q_prime_fp = fp(Q * (64**-0.25))
                k_prime_fp = fp(K * (64**-0.25))
                D_inv_fp = torch.diag(1 / (q_prime_fp @ (k_prime_fp.T @ torch.ones(Q.shape[0], device=device))))
                a = (self.kernel(Q, K)).clone().detach().cpu().numpy()
                a_hat = (D_inv @ Q_prime @ K_prime.T).clone().detach().cpu().numpy()
                a_hat_fp = (D_inv_fp @ q_prime_fp @ k_prime_fp.T).clone().detach().cpu().numpy()
                # find embedding tokens, save only the attention w/o padding
                m_diag = np.diag(a)
                embedding = m_diag[-1]
                for j in range(len(m_diag)):
                    if m_diag[j] == embedding:
                        emb_idx = j
                        break
                # store them
                save_mat[i]["a"] = a[:emb_idx, :emb_idx]
                save_mat[i]["a_hat"] = a_hat[:emb_idx, :emb_idx]
                save_mat[i]["a_hat_fp"] = a_hat_fp[:emb_idx, :emb_idx]

            # Log approximation loss.
            err.append(float(approx_loss(A, A_hat)))

        if self.save_mat:
            import pickle

            with open("resources/hardware/attention/attn_mat.pickle", "wb") as handle: pickle.dump(save_mat, handle, protocol=pickle.HIGHEST_PROTOCOL)
            raise ValueError()

        return np.mean(err), np.mean(elapsed)


def plot(err, fp_err, config):
    """Plot experimental results.

    Args:
        acc (defaultdict): logged downstream accuracy
        elp (defaultdict): logged elapsed execution time
        err (defaultdict): logged approximation error
        kernels (list): kernels tested
        config (defaultdict): experiment parameters
    """
    fig, ax = plt.subplots(1, 1)
    fig.set_size_inches(8, 8)
    cyc = iter(plot_color_cycle)

    for technique in config.rf:
        color = next(cyc)["color"]
        def mean(x):return np.array([np.mean([x[technique][n]]) for n in config.hidden_features])
        def std(x): return np.array([np.std([x[technique][n]]) for n in config.hidden_features])

        # # ANALOG
        # ax.errorbar(x=config.hidden_features,y=mean(err),yerr=std(err),marker=".",linestyle="dashed",capsize=6,color=color,) 
        #FP
        ax.errorbar(x=config.hidden_features,y=mean(fp_err),yerr=std(fp_err),marker=".",capsize=6,color=color,)
        mode = "_emulated"

    ax.set_xlabel("$s/d$", fontsize=12)
    ax.set_ylabel("approximation error", fontsize=12)

    dash = Line2D([0], [0], label="HW", color="k", linestyle="dashed")
    soli = Line2D([0], [0], label="FP", color="k", linestyle="solid")
    lgd = plt.legend(handles=[dash, soli],loc="lower center",bbox_to_anchor=(0.5, -0.04),ncol=2,bbox_transform=fig.transFigure,)

    fig.tight_layout()
    mode = "emulated"
    folder_name = "resources/hardware/attention"
    os.makedirs(folder_name, exist_ok=True)
    filename = os.path.join(folder_name, f"{mode}.png")
    plt.savefig(filename,bbox_extra_artists=(lgd,),bbox_inches="tight",)


def save(err, fp_err, config):
    """
    Save experiment results as pickles.
    """
    folder_name = "resources/hardware/attention/"
    os.makedirs(folder_name, exist_ok=True)
    mode = "emulated"
    # file_name = os.path.join(folder_name, f"{mode}_err.pkl")
    # with open(file_name, "wb") as file:
    #     dill.dump(dict(err), file)
    file_name = os.path.join(folder_name, f"{mode}_fp_err.pkl")
    with open(file_name, "wb") as file: dill.dump(dict(fp_err), file)


if __name__ == "__main__":
    main()
