import torch
import torch.nn.functional as F

def top_p_sampling(probabilities, top_p=0.9):
    """
    Top-p 核采样
    probabilities: [vocab_size]
    """
    sorted_probs, sorted_indices = torch.sort(probabilities, descending=True)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

    # 创建mask，累积概率超过top_p的token置零
    mask = cumulative_probs > top_p
    mask[..., 1:] = mask[..., :-1].clone()  # 向右移动，确保至少保留一个token
    mask[..., 0] = 0
    sorted_probs[mask] = 0.0

    # 重新归一化
    sorted_probs /= sorted_probs.sum()

    # 采样
    next_token_in_sorted = torch.multinomial(sorted_probs, 1)
    next_token = sorted_indices[next_token_in_sorted]
    return next_token

def temperature_scaling(logits, temperature=1.0):
    """
    温度缩放 + softmax
    logits: [vocab_size]
    """
    return F.softmax(logits / temperature, dim=-1)

def decode_token(input_tokens, model, max_tokens_to_generate, top_p=0.9, temperature=1.0, eos_token_id=None):
    """
    解码推理
    input_tokens: list[int] 或 tensor([seq_len])
    """
    model.eval()
    input_tokens = torch.tensor(input_tokens).unsqueeze(0)  # [1, seq_len]

    with torch.no_grad():
        for _ in range(max_tokens_to_generate):
            logits = model(input_tokens)  # [1, seq_len, vocab_size]
            logits_last = logits[:, -1, :]  # 取最后一个位置
            probs = temperature_scaling(logits_last.squeeze(0), temperature)
            next_token_idx = top_p_sampling(probs, top_p)
            input_tokens = torch.cat([input_tokens, next_token_idx.unsqueeze(0)], dim=-1)

            # 如果下一个token是终止token，停止生成
            if eos_token_id is not None and next_token_idx.item() == eos_token_id:
                break

    return input_tokens.squeeze(0).tolist()  # 返回 list[int]

def generate(model, tokenizer, input_tokens, max_tokens_to_generate=50, top_p=0.9, temperature=1.0, eos_token_id=None):
    """
    生成文本
    """
    output_ids = decode_token(input_tokens, model, max_tokens_to_generate, top_p, temperature, eos_token_id)
    return tokenizer.decode(output_ids)
