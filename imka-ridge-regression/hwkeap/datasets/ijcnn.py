import torch


class IJCNN1(torch.utils.data.Dataset):
    """IJCNN 2001 Generalization Ability Challenge dataset.

    Args:
        data : pair (X, y)
    """

    def __init__(self, data, gt):
        self.X = data
        self.gt = gt

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        x = self.X[idx]
        gt = self.gt[idx]
        return x, gt
