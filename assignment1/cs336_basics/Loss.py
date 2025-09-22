import torch 
from torch import Tensor


def log_softmax(x:Tensor,dim:int):
    x_max = x.max(dim = dim,keepdim=True).values
    sum_exp = torch.exp(x - x_max).sum(dim=dim,keepdim=True)
    return x - x_max - torch.log(sum_exp)


def cross_entropy(inputs:Tensor, targets:Tensor) -> Tensor:

    probs = log_softmax(inputs,-1)
    targets = targets.to(probs.device).long()
    target_probs = probs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    
    return -target_probs.mean()

def perplexity(inputs:Tensor, targets:Tensor) -> Tensor:

    ce = cross_entropy(inputs,targets)
    return torch.exp(ce)

