from cs336_basics.BPETokenizer import BPETokenizer
from cs336_basics.Loss import cross_entropy
from typing import Any, BinaryIO, IO, Iterable, Optional, Union
from cs336_basics.optimizer import Adamw, get_lr_cosine_schedule
from cs336_basics.gradient_clip import gradient_clipping
from cs336_basics.check_point import save_checkpoint, load_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.train_bpe import train_bpe
import torch
import os
import time
import json
import pickle
from torch import nn
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
class Args:
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    epochs = 40
    train_steps = 5000
    batch_size = 32
args = Args()

device = args.device
epochs = args.epochs
train_steps = args.train_steps
batch_size = args.batch_size

timestamp = time.strftime("%Y%m%d_%H%M%S")
config = {
    # 实验配置
    "experiment_name": f"tinystories_17M_{timestamp}",
    "total_tokens_processed": 40000000,
    
    # 数据配置
    "train_data_path": "../data/TinyStoriesV2-GPT4-train.txt",
    "valid_data_path": "../data/TinyStoriesV2-GPT4-valid.txt",
    "vocab_path": "../out/owt_train-vocab.txt",
    "merges_path": "../out/owt_train-merges.txt",

    # 模型配置
    "vocab_size": 10000,
    "context_length": 256,
    "d_model": 512,
    "d_ff": 1344,
    "n_layers": 4,
    "n_heads": 16,
    "rope_theta": 10000.0,

    # 训练配置
    "batch_size": batch_size,
    "initial_lr": 3e-5,
    "max_learning_rate": 3e-5,
    "min_learning_rate": 1e-5,
    "lr_warmup_steps": 2000,
    "cosine_cycle_iters": 10000,
    "weight_decay": 0.1,
    "adam_beta1": 0.9,
    "adam_beta2": 0.95,
    "eps": 1e-8,
    "grad_clip": 1.0,
    "epochs": epochs,
    "train_steps": train_steps,
    
    # 日志和检查点配置
    "log_interval": 20,
    "val_interval": 20,
    "checkpoint_interval": 60,
    "checkpoint_dir": "checkpoints",
}
os.makedirs("../config", exist_ok=True)
with open(f"../config/config.json", "w") as f:
    json.dump(config, f, indent=4)
device = torch.device(device if torch.cuda.is_available() else "cpu")

# 训练BPE分词器
special_tokens = ["<|endoftext|>"]
data_path = config["train_data_path"]
vocab_size = config["vocab_size"]
# vocab, merges = train_bpe(data_path, vocab_size, special_tokens,merges_outpath="../out/merges.pkl",vocab_outpath="../out/vocab.pkl")
# print("已经训练好BPE分词器")

# 从 vocab.pkl 加载词汇表
with open("../out/vocab.pkl", "rb") as f:
    # pickle.load 会自动恢复字典，并且值是 bytes 类型
    vocab = pickle.load(f)

# 从 merges.pkl 加载合并规则
with open("../out/merges.pkl", "rb") as f:
    # pickle.load 会自动恢复列表，并且元组里的元素是 bytes 类型
    merges = pickle.load(f)
special_tokens = ["<|endoftext|>"]  

tokenizer = BPETokenizer(vocab, merges, special_tokens)
import pickle
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from tqdm import tqdm
import os
import time
import multiprocessing as mp
import hashlib

def get_cache_filename(file_path, vocab_size):
    """生成缓存文件名"""
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    return f"../out/cache_{file_hash}_{vocab_size}.pkl"

def encode_chunk_worker(args):
    """多进程编码工作函数"""
    chunk_text, vocab, merges, special_tokens = args
    tokenizer = BPETokenizer(vocab, merges, special_tokens)
    return tokenizer.encode(chunk_text)

def fast_encode_file(tokenizer, file_path, output_path, max_workers=None, chunk_size = 1 * 1024 * 1024):
    print(f"📁 Loading {file_path}...")
    
    # 检查输出文件缓存
    if os.path.exists(output_path):
        print("📦 Loading from cache...")
        start = time.time()
        with open(output_path, 'rb') as f:
            data = pickle.load(f)
        print(f"✅ Cached data loaded in {time.time() - start:.2f}s ({len(data):,} tokens)")
        return data
    
    # 读取文件
    print("📖 Reading file...")
    start = time.time()
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return []
    
    print(f"File read in {time.time() - start:.2f}s ({len(content):,} chars)")
    
    # 分块处理
    chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
    print(f"Split into {len(chunks)} chunks")
    
    # 并行编码
    print("🚀 Encoding chunks...")
    start = time.time()
    all_tokens = []
    max_workers = max_workers or min(4, mp.cpu_count())
    
    chunk_args = [(chunk, tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens) 
                    for chunk in chunks]
    
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        results = list(tqdm(
            executor.map(encode_chunk_worker, chunk_args),
            total=len(chunks),
            desc="Encoding"
        ))
    
    for tokens in results:
        all_tokens.extend(tokens)
    
    encode_time = time.time() - start
    print(f"✅ Encoded {len(all_tokens):,} tokens in {encode_time:.2f}s "
          f"({len(content)/encode_time:.0f} chars/sec)")
    
    # 保存缓存
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(all_tokens, f)
        print(f"💾 Saved cache to {output_path}")
    except Exception as e:
        print(f"⚠️ Could not save cache: {e}")
    
    return all_tokens

train_ids = fast_encode_file(
        tokenizer,
        config["train_data_path"],
        "../out/encoded_ids_train.pkl",
        max_workers=4
    )

# 编码验证数据
print("="*50)
valid_ids = fast_encode_file(
    tokenizer,
    config["valid_data_path"], 
    "../out/encoded_ids_valid.pkl",
    max_workers=4
)

print(f"\n🎉 完成! 训练:{len(train_ids):,} 验证:{len(valid_ids):,}")