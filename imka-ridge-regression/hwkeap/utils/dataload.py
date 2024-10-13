import bz2
import os
import urllib.request
from pathlib import Path

import numpy as np
import scipy.io
import torch
import torch.utils.data as data_utils
from easydict import EasyDict as edict
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms

from hwkeap.utils.misc import join_datasets, shuffle_and_split


def __load_attention(ipath: str, norm: bool = True):
    """Load Attention dataset.
    Syntetic dataset that contains key, query and value vectors extracted from
    an HuggingFace bert-base-uncased model, fed with sentences from the SMILE
    twitter dataset for sentiment classification.
    The representations were extracted from all the 12 heads of the first
    encoder layer of the model.
    The vectors were extracted from the Transformer available at the following link:
    https://github.com/baotramduong/Twitter-Sentiment-Analysis-with-Deep-Learning-using-BERT.
    This dataset purpose is to study the approximation error for attention
    kernels. Hence, it does not have any label.

    Args:
        opath (str): input path
    """
    file = Path(os.path.join(ipath, "data.pt"))
    data = torch.load(file, map_location="cpu")
    trainset, testset = train_test_split(data, test_size=0.2)
    return trainset, testset


def __download_mnist(opath: str):
    """Download MNIST Dataset data.

    Args:
        opath (str): output path
    """
    train_dataset = datasets.MNIST(opath, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.MNIST(opath, train=False, download=True, transform=transforms.ToTensor())
    return train_dataset, test_dataset


def __download_fashion(opath: str):
    """Download MNIST Dataset data.

    Args:
        opath (str): output path
    """
    train_dataset = datasets.FashionMNIST(opath, train=True, download=True, transform=transforms.ToTensor())
    test_dataset = datasets.FashionMNIST(opath, train=False, download=True, transform=transforms.ToTensor())
    return train_dataset, test_dataset


# review paper datasets


def __download_ijcnn1(opath: str, norm: bool):
    """Download and load IJCNN 2001 Generalization Ability Challenge data.

    Args:
        opath (str): output path

    Returns:
        tuple: returns three sets, train (35000 elements) val (14990 elements) \
                and test (91701 elements)
    """
    files = [Path(os.path.join(opath, "ijcnn1.tr.bz2")),Path(os.path.join(opath, "ijcnn1.val.bz2")),Path(os.path.join(opath, "ijcnn1.t.bz2")),]
    urls = ["https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.tr.bz2","https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.val.bz2","https://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/ijcnn1.t.bz2",]

    # download binaries if not existing
    os.makedirs(opath, exist_ok=True)
    for file, url in zip(files, urls):
        if not file.is_file(): urllib.request.urlretrieve(url, file)

    # read and decompress files
    sets = list()

    def remove_prefix(s):
        """Remove 'a:' in the string 'a:b'"""
        return float(s.split(":")[1])

    def remove_postfix(s):
        """Remove ':b' in the string 'a:b'"""
        return float(s.split(":")[0])

    def decode_line(line):
        """Decode byte dataset line"""
        features = line.decode("utf-8").split(" ")
        pulse = [int(remove_postfix(features[1]))]
        others = list(map(remove_prefix, features[2:]))
        label = int(float(features[0]))
        return pulse + others, label

    for file in files:
        zipfile = bz2.BZ2File(file)
        X, Y = [], []
        for line in zipfile:
            x, y = decode_line(line)
            X.append(x)
            Y.append(y if y == 1 else 0)

        X = torch.FloatTensor(X)
        Y = torch.IntTensor(Y)
        if norm:
            mean, std = torch.mean(X, dim=0), torch.std(X, dim=0)
            X = (X - mean) / std
        sets.append((X, Y))

    return tuple(sets)


def __load_skin_dataset(opath: str, norm: bool):
    # read and decompress files

    def remove_prefix(s):
        """Remove 'a:' in the string 'a:b'"""
        return float(s.split(":")[1])

    def decode_line(line):
        """Decode byte dataset line"""
        features = line.split(" ")
        label = int(float(features[0]))
        features = list(map(remove_prefix, features[1:]))
        return features, label

    with open(os.path.join(opath, "dataset.txt")) as f:
        lines = f.readlines()
        X, Y = [], []
        for line in lines:
            x, y = decode_line(line)
            X.append(x)
            Y.append(y if y == 1 else 0)

        X = torch.FloatTensor(X)
        Y = torch.IntTensor(Y)
        if norm:
            mean, std = torch.mean(X, dim=0), torch.std(X, dim=0)
            X = (X - mean) / std

    return shuffle_and_split(X, Y)


def __load_codrna_dataset(opath: str, norm: bool):
    # read and decompress files

    def remove_prefix(s):
        """Remove 'a:' in the string 'a:b'"""
        return float(s.split(":")[1])

    def decode_line(line):
        """Decode byte dataset line"""
        features = line.split(" ")
        label = int(float(features[0]))
        features = list(map(remove_prefix, features[1:]))
        return features, label

    def decode_file(file, norm):
        with open(file) as f:
            lines = f.readlines()
            X, Y = [], []
            for line in lines:
                x, y = decode_line(line)
                X.append(x)
                Y.append(y if y == 1 else 0)

            X = torch.FloatTensor(X)
            Y = torch.IntTensor(Y)
            if norm:
                mean, std = torch.mean(X, dim=0), torch.std(X, dim=0)
                X = (X - mean) / std
        return X, Y

    X_train, y_train = decode_file(os.path.join(opath, "training.txt"), norm)
    X_test, y_test = decode_file(os.path.join(opath, "validation.txt"), norm)

    X_val, X_test = torch.tensor_split(X_test, 2)
    y_val, y_test = torch.tensor_split(y_test, 2)

    return [(X_train, y_train), (X_val, y_val), (X_test, y_test)]


def __load_letter_dataset(opath: str, norm: bool):
    data = scipy.io.loadmat(os.path.join(opath, "letter.mat"))
    X_train, y_train = (torch.FloatTensor(data["X_train"]),torch.IntTensor(data["Y_train"]).squeeze(),)
    X_test, y_test = (torch.FloatTensor(data["X_test"]),torch.IntTensor(data["Y_test"]).squeeze(),)
    if norm:
        mean, std = torch.mean(X_train, dim=0), torch.std(X_train, dim=0)
        X_train = (X_train - mean) / std
        mean, std = torch.mean(X_test, dim=0), torch.std(X_test, dim=0)
        X_test = (X_test - mean) / std

    y_train[y_train == -1] = 0
    y_test[y_test == -1] = 0

    X_val, X_test = torch.tensor_split(X_test, 2)
    y_val, y_test = torch.tensor_split(y_test, 2)

    return [(X_train, y_train), (X_val, y_val), (X_test, y_test)]


def __load_magic_dataset(opath: str, norm: bool):
    def decode_line(line):
        """Decode byte dataset line"""
        features = line.split(",")
        label = features[-1]
        features = [float(x) for x in features[:-1]]
        return features, label

    with open(os.path.join(opath, "magic04.data")) as f:
        lines = f.readlines()
        X, Y = [], []
        for line in lines:
            x, y = decode_line(line)
            X.append(x)
            Y.append(1 if y == "g\n" else 0)

    X = torch.FloatTensor(X)
    Y = torch.IntTensor(Y)

    if norm:
        mean, std = torch.mean(X, dim=0), torch.std(X, dim=0)
        X = (X - mean) / std

    return shuffle_and_split(X, Y)


def __load_eeg_dataset(opath: str, norm: bool):
    data = scipy.io.arff.loadarff(os.path.join(opath, "eeg.arff"))
    X, Y = [], []
    for a in data[0]:
        byte_arr = np.asarray(a.tolist())
        x = [float(x) for x in byte_arr[:-1]]
        y = int(byte_arr[-1])
        X.append(x)
        Y.append(y)
    X = torch.FloatTensor(X)
    Y = torch.IntTensor(Y)
    if norm:
        mean, std = torch.mean(X, dim=0), torch.std(X, dim=0)
        X = (X - mean) / std

    return shuffle_and_split(X, Y)


def load_data_loaders(args: edict, merge_val_test: bool = False) -> tuple:
    """Load training, validation, and test datasets as torch.Dataloader.

    Args:
        args (edict): command line arguments
        merge_val_test (bool): if true, merges the val and test set in a single set

    Returns:
        tuple: returns a tuple of torch DataLoader, (train_set, val_set, test_set)
               if !merge_val_test, (train_set, test_set) otherwise
    """
    # load data
    trainset, valset, testset = load_data_tensors(args)
    if merge_val_test:testset = join_datasets(valset, testset)
    # create torch data loaders
    train_loader = torch.utils.data.DataLoader(dataset=data_utils.TensorDataset(trainset[0], trainset[1]),batch_size=args.batch_size,shuffle=False,drop_last=False,)
    val_loader = torch.utils.data.DataLoader(dataset=data_utils.TensorDataset(valset[0], valset[1]),batch_size=args.batch_size,shuffle=False,drop_last=False,)
    test_loader = torch.utils.data.DataLoader(dataset=data_utils.TensorDataset(testset[0], testset[1]),batch_size=args.batch_size,shuffle=False,drop_last=False,)
    return (train_loader, test_loader)if merge_val_test else (train_loader, val_loader, test_loader)


def load_data_tensors(args: edict) -> tuple:
    """Load training, validation, and test datasets for sklearn models.

    Args:
        args (edict): command line arguments

    Returns:
        tuple: return a tuple (train_set, val_set, test_set), where the sets are numpy arrays
    """

    def tuple2device(tup, device):
        return tuple(tuple(tt.to(device) for tt in t) for t in tup)

    if args.dataset == "mnist" or args.dataset == "fashion":
        train_dataset, test_dataset = __download_mnist(args.dpath) if args.dataset == "mnist" else __download_fashion(args.dpath)
        X_train, y_train = (torch.flatten(train_dataset.data, start_dim=1).float(),train_dataset.targets,)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=args.val_size, random_state=args.seed)
        X_test, y_test = (torch.flatten(test_dataset.data, start_dim=1).float(),test_dataset.targets,)

        if args.norm:
            mean_train, std_train = torch.mean(X_train, dim=0), torch.std(X_train, dim=0)
            mean_val, std_val = torch.mean(X_val, dim=0), torch.std(X_val, dim=0)
            mean_test, std_test = torch.mean(X_test, dim=0), torch.std(X_test, dim=0)
            X_train = (X_train - mean_train) / std_train
            X_val = (X_val - mean_val) / std_val
            X_test = (X_test - mean_test) / std_test

        sets = tuple(((X_train, y_train), (X_val, y_val), (X_test, y_test)))

    elif args.dataset == "ijcnn1":
        path = os.path.join(args.dpath, "ijcnn1")
        sets = __download_ijcnn1(path, args.norm)

    elif args.dataset == "skin":
        path = os.path.join(args.dpath, "skin")
        sets = __load_skin_dataset(path, args.norm)

    elif args.dataset == "cod-rna":
        path = os.path.join(args.dpath, "cod-rna")
        sets = __load_codrna_dataset(path, args.norm)

    elif args.dataset == "letter":
        path = os.path.join(args.dpath, "letter")
        sets = __load_letter_dataset(path, args.norm)

    elif args.dataset == "magic":
        path = os.path.join(args.dpath, "magic")
        sets = __load_magic_dataset(path, args.norm)

    elif args.dataset == "eeg":
        path = os.path.join(args.dpath, "EEG")
        sets = __load_eeg_dataset(path, args.norm)

    elif args.dataset == "attention":
        attpath = os.path.join(args.dpath, "attention")
        trainset, testset = __load_attention(attpath, args.norm)
        trainset, valset = train_test_split(trainset, test_size=0.25)
        sets = tuple(((trainset,), (valset,), (testset,)))

    elif args.dataset == "attention":
        attpath = os.path.join(args.dpath, "attention")
        trainset, testset = __load_attention(attpath)
        trainset, valset = train_test_split(trainset, test_size=0.25)

        sets = tuple(((trainset,), (valset,), (testset,)))

    return tuple2device(sets, args.device)


if __name__ == "__main__":
    # debug
    args = edict({"dataset": "skin", "norm": True, "dpath": "data", "device": "cpu"})
    sets = load_data_tensors(args)
    args = edict({"dataset": "eeg", "norm": True, "dpath": "data", "device": "cpu"})
    a2 = load_data_tensors(args)
    print(sets)
