import torch

def skipgram_pairs(token_ids: torch.Tensor, window: int) -> torch.Tensor:
    """
    Returns int64 torch.Tensor of shape (num_pairs, 2).
    """
    pairs = []
    for i in range(len(token_ids)):
        center = token_ids[i]
        start = max(0, i - window)
        end = min(len(token_ids), i + window + 1)
        
        for j in range(start, end):
            if i != j:
                pairs.append([center.item(), token_ids[j].item()])
    if not pairs:
        return torch.zeros((0, 2), dtype=torch.int64)
    return torch.tensor(pairs, dtype=torch.int64)
            
            
