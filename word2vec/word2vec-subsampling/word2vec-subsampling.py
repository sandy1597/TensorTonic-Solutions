import torch

def subsample_keep_probs(counts: torch.Tensor, t: float = 1e-5) -> torch.Tensor:
    """
    Returns torch.Tensor of shape (vocab_size,) with the keep-probability for each word.
    """
    # YOUR CODE HERE
    total = torch.sum(counts)
    f_w = counts / total
    p_keep = torch.sqrt(t / f_w)
    p_keep = torch.clamp(p_keep, max=1.0)

    return p_keep
