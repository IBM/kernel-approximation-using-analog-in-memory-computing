import torch
from fairseq.utils import move_to_cuda
from tqdm import tqdm


@torch.no_grad()
def accuracy(model, loader, get_norm_p, device):
    """Compute model accuracy."""
    preds = []
    targets = []
    total = correct = 0.0

    for test_batch in tqdm(loader):
        test_batch = move_to_cuda(test_batch) if device == "cuda" else test_batch
        targets.append(test_batch["target"].view(-1))
        preds.append(model(test_batch))

    for pred, target in zip(preds, targets):
        lprobs = get_norm_p(pred, log_probs=True)
        lprobs = lprobs.view(-1, lprobs.size(-1))
        pred = torch.argmax(lprobs, 1)
        correct += (pred == target).sum().item()
        total += pred.shape[0]

    return correct / total * 100.0
