from abc import ABC


class Sampler(ABC):
    """
    Model data sampler.
    """

    def __init__(self, device) -> None:
        super().__init__()
        self.device = device

    def sample(self):
        x = next(self.loader)
        return x


class BatchedTensorIterator:
    def __init__(self, tensor):
        self.tensor = tensor
        self.current_idx = -1

    def __iter__(self):
        return self

    def __next__(self):
        self.current_idx += 1
        if self.current_idx >= self.tensor.shape[0]:
            self.current_idx = 0

        return self.tensor[self.current_idx]


class SamplerTensor(Sampler):
    """
    Data sampler that works with tensors.
    """

    def __init__(self, tensor, device):
        super().__init__(device)
        self.loader = BatchedTensorIterator(tensor)


class SamplerLoader(Sampler):
    """
    Data sampler that works with data loaders.
    """

    def __init__(self, loader, device):
        super().__init__(device)
        self.loader = iter(loader)
