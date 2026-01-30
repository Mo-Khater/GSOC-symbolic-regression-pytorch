import torch
import math

def calculate_complexity(tree, seen=None):
    """
    Complexity = number of nodes in the tree.
    """
    if seen is None:
        seen = set()
    node_id = id(tree)
    seen.add(node_id)
    total = 1 + sum(calculate_complexity(child, seen) for child in tree.children)
    seen.remove(node_id)
    return total


def calculate_loss_and_cost(complexity,dataset,tree,parsimony_penalty):
    X = dataset.X
    y = dataset.y
    w = getattr(dataset, "weights", None) # for future implementaion

    yhat = tree.forward(X)
    if torch.isnan(yhat).any() or torch.isinf(yhat).any():
        return float("inf"), float("inf")
    err2 = (yhat - y) ** 2

    if w is None:
        loss_t = err2.mean()
    else:
        # weighted mean
        loss_t = (err2 * w).sum() / (w.sum() + 1e-12)

    loss = float(loss_t.detach().cpu().item())
    cost = float(loss + parsimony_penalty * complexity)
    if math.isnan(loss) or math.isinf(loss):
        return float("inf"), float("inf")
    return loss, cost