import torch
import torch.nn as nn

def gradient_clipping(params, max_l2_norm):
    # 1. 计算所有梯度的L2范数平方和
    total_norm_sq = 0.0
    for param in params:
        if param.grad is not None: 
            param_norm = param.grad.data.norm(2)  
            total_norm_sq += param_norm.item() **2  
    
    total_norm = total_norm_sq** 0.5  
    
    clip_coef = max_l2_norm / (total_norm + 1e-6)  
    if clip_coef < 1:  
        for param in params:
            if param.grad is not None:
                param.grad.data.mul_(clip_coef)
    