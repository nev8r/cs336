import os
import sys
import torch
import numpy as np
from tqdm import tqdm
import time
import pickle
import hashlib
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

from cs336_basics.BPETokenizer import BPETokenizer
from cs336_basics.model import TransformerLM
from cs336_basics.Loss import cross_entropy, perplexity
from cs336_basics.optimizer import get_adamw_cls, get_lr_cosine_schedule
from cs336_basics.gradient_clip import gradient_clipping
from cs336_basics.check_point import save_checkpoint, load_checkpoint
from cs336_basics.data import get_batch


def set_seed(seed: int = 42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_cache_filename(file_path: str, vocab_size: int) -> str:
    """生成缓存文件名"""
    file_hash = hashlib.md5(file_path.encode()).hexdigest()[:8]
    return f"{file_path}_{file_hash}_{vocab_size}.npy"


def encode_chunk(args):
    """并行处理单个文本块"""
    chunk_text, tokenizer = args
    return tokenizer.encode(chunk_text)


class FastDataLoader:
    """超快数据加载器 - 多进程 + 缓存 + 分块"""
    
    def __init__(self, tokenizer: BPETokenizer, max_workers: int = None):
        self.tokenizer = tokenizer
        self.max_workers = max_workers or min(4, mp.cpu_count())
        
    def load_and_encode(self, file_path: str, max_tokens: int = None) -> np.ndarray:
        """加载并编码文件"""
        print(f"Loading {file_path}...")
        
        # 检查缓存
        cache_path = get_cache_filename(file_path, len(self.tokenizer.vocab))
        if os.path.exists(cache_path):
            print("📦 Loading from cache...")
            start = time.time()
            data = np.load(cache_path)
            print(f"✅ Cached data loaded in {time.time() - start:.2f}s ({len(data):,} tokens)")
            return data[:max_tokens] if max_tokens else data
        
        # 读取文件
        print("📖 Reading file...")
        start = time.time()
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"❌ Error reading file: {e}")
            return np.array([], dtype=np.int64)
        
        print(f"File read in {time.time() - start:.2f}s ({len(content):,} chars)")
        
        # 分块处理 - 避免单次编码太大的文本
        chunk_size = 50000  # 5万字符一块，平衡速度和内存
        chunks = [content[i:i+chunk_size] for i in range(0, len(content), chunk_size)]
        print(f"Split into {len(chunks)} chunks")
        
        # 并行编码
        print("🚀 Encoding chunks...")
        start = time.time()
        
        all_tokens = []
        if len(chunks) == 1 or self.max_workers == 1:
            # 单线程处理
            for chunk in tqdm(chunks, desc="Encoding"):
                tokens = self.tokenizer.encode(chunk)
                all_tokens.extend(tokens)
        else:
            # 多进程处理
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                chunk_args = [(chunk, self.tokenizer) for chunk in chunks]
                results = list(tqdm(
                    executor.map(encode_chunk, chunk_args),
                    total=len(chunks),
                    desc="Encoding"
                ))
                for tokens in results:
                    all_tokens.extend(tokens)
        
        encode_time = time.time() - start
        print(f"✅ Encoded {len(all_tokens):,} tokens in {encode_time:.2f}s "
              f"({len(content)/encode_time:.0f} chars/sec)")
        
        # 转换并限制长度
        data = np.array(all_tokens, dtype=np.int64)
        if max_tokens and len(data) > max_tokens:
            data = data[:max_tokens]
            print(f"Truncated to {max_tokens:,} tokens")
        
        # 保存缓存
        try:
            np.save(cache_path, data)
            print(f"💾 Saved cache to {cache_path}")
        except Exception as e:
            print(f"⚠️ Could not save cache: {e}")
        
        return data


def generate_sample(model, tokenizer, device, context_length, max_length=100, temperature=0.8):
    """生成文本样本"""
    model.eval()
    with torch.no_grad():
        # 简单的起始
        start_ids = [1, 2, 3]
        context = torch.tensor(start_ids, dtype=torch.long, device=device).unsqueeze(0)
        generated = start_ids.copy()
        
        for _ in range(max_length):
            if context.size(1) > context_length:
                context = context[:, -context_length:]
            
            logits = model(context)[:, -1, :] / temperature
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            generated.append(next_token.item())
            context = torch.cat([context, next_token], dim=1)
        
        try:
            text = tokenizer.decode(generated)
            print(f"📝 Generated: {text[:150]}{'...' if len(text) > 150 else ''}")
        except:
            print("📝 Generated: [decode error]")
    
    model.train()


def train_model(model, train_data, val_data, tokenizer, device, config):
    """简化高效的训练循环"""
    
    print(f"🚀 Training on {device}")
    print(f"📊 Model: {sum(p.numel() for p in model.parameters()):,} parameters")
    print(f"📊 Data: {len(train_data):,} train tokens, {len(val_data):,} val tokens")
    
    # 优化器
    optimizer = get_adamw_cls()(
        model.parameters(), 
        lr=config['max_lr'], 
        weight_decay=config.get('weight_decay', 0.1)
    )
    
    # 训练状态
    best_val_ppl = float('inf')
    os.makedirs(config['checkpoint_dir'], exist_ok=True)
    
    # 训练循环
    train_start = time.time()
    
    for step in tqdm(range(config['max_iters']), desc="Training"):
        # 学习率调度
        lr = get_lr_cosine_schedule(
            step, config['max_lr'], config['min_lr'],
            config['warmup_steps'], config['max_iters']
        )
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        
        # 前向传播
        x, y = get_batch(train_data, config['batch_size'], config['context_length'], device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, config['vocab_size']), y.view(-1))
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        gradient_clipping(model.parameters(), config['max_grad_norm'])
        optimizer.step()
        
        # 验证和日志
        if step % config['eval_interval'] == 0 or step == config['max_iters'] - 1:
            # 验证
            model.eval()
            val_losses = []
            val_ppls = []
            
            with torch.no_grad():
                for _ in range(config['eval_batches']):
                    x_val, y_val = get_batch(val_data, config['batch_size'], 
                                           config['context_length'], device)
                    logits_val = model(x_val)
                    val_loss = cross_entropy(logits_val.view(-1, config['vocab_size']), 
                                           y_val.view(-1))
                    val_ppl = perplexity(logits_val.view(-1, config['vocab_size']), 
                                       y_val.view(-1))
                    val_losses.append(val_loss.item())
                    val_ppls.append(val_ppl.item())
            
            val_loss_avg = np.mean(val_losses)
            val_ppl_avg = np.mean(val_ppls)
            
            # 保存最佳模型
            if val_ppl_avg < best_val_ppl:
                best_val_ppl = val_ppl_avg
                save_path = os.path.join(config['checkpoint_dir'], 'best_model.pt')
                save_checkpoint(model, optimizer, step, save_path)
                print(f"\n💾 Best model saved! PPL: {best_val_ppl:.4f}")
            
            # 打印状态
            elapsed = time.time() - train_start
            print(f"\n📈 Step {step:4d} | Loss: {loss.item():.4f} | "
                  f"Val Loss: {val_loss_avg:.4f} | Val PPL: {val_ppl_avg:.4f} | "
                  f"LR: {lr:.2e} | Time: {elapsed/60:.1f}min")
            
            # 生成样本
            if config.get('generate_samples', True):
                generate_sample(model, tokenizer, device, config['context_length'])
            
            model.train()
        
        # 定期保存
        if step > 0 and step % config.get('save_interval', 500) == 0:
            save_path = os.path.join(config['checkpoint_dir'], f'step_{step}.pt')
            save_checkpoint(model, optimizer, step, save_path)
    
    return best_val_ppl


def main():
    set_seed(42)
    
    # 配置 - 针对快速训练优化
    config = {
        # 数据
        'max_tokens': 2_000_000,  # 限制200万token，避免内存爆炸
        'max_workers': 2,  # 并行进程数
        
        # 训练
        'batch_size': 16,  # 较小batch size，适合较小GPU
        'context_length': 256,
        'max_iters': 2000,
        'eval_interval': 100,
        'eval_batches': 5,
        'save_interval': 500,
        'checkpoint_dir': './fast_checkpoints',
        
        # 优化器
        'max_lr': 3e-4,
        'min_lr': 3e-5,
        'warmup_steps': 100,
        'weight_decay': 0.1,
        'max_grad_norm': 1.0,
        
        # 生成
        'generate_samples': True,
        
        # 模型 - 中等大小
        'vocab_size': 50257,
        'd_model': 512,
        'num_layers': 8,
        'num_heads': 8,
        'd_ff': 2048,
        'rope_theta': 10000.0,
    }
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🖥️ Using device: {device}")
    
    # 加载tokenizer
    print("🔧 Loading tokenizer...")
    try:
        tokenizer = BPETokenizer.load_from_file(
            "../out/owt_valid-vocab.txt",
            "../out/owt_valid-merges.txt",
            ["<|endoftext|>"]
        )
        config['vocab_size'] = len(tokenizer.vocab)
        print(f"✅ Tokenizer loaded: {config['vocab_size']:,} vocab size")
    except Exception as e:
        print(f"❌ Tokenizer error: {e}")
        print("Creating minimal test setup...")
        # 这里可以创建一个最小化的测试环境
        return
    
    # 加载数据 - 使用超快加载器
    data_loader = FastDataLoader(tokenizer, max_workers=config['max_workers'])
    data_path = "../data/owt_valid.txt"
    
    if os.path.exists(data_path):
        dataset = data_loader.load_and_encode(data_path, config['max_tokens'])
    else:
        print("❌ Data file not found, creating test data...")
        test_text = ("Hello world! This is a test sentence for training. " * 100 +
                    "The quick brown fox jumps over the lazy dog. " * 100)
        tokens = tokenizer.encode(test_text)
        dataset = np.array(tokens, dtype=np.int64)
        print(f"📝 Created test dataset: {len(dataset)} tokens")
    
    if len(dataset) < 1000:
        print("❌ Dataset too small!")
        return
    
    # 转换为tensors
    train_data = torch.from_numpy(dataset)
    val_data = dataset  # 简单起见，用同样的数据做验证
    
    print(f"✅ Data ready: {len(dataset):,} tokens")
    
    # 初始化模型
    print("🧠 Initializing model...")
    model = TransformerLM(
        vocab_size=config['vocab_size'],
        context_length=config['context_length'],
        d_model=config['d_model'],
        num_layers=config['num_layers'],
        num_heads=config['num_heads'],
        d_ff=config['d_ff'],
        rope_theta=config['rope_theta']
    ).to(device)
    
    # 开始训练
    print("🎯 Starting training...")
    best_ppl = train_model(model, train_data, val_data, tokenizer, device, config)
    
    print(f"\n🎉 Training completed!")
    print(f"🏆 Best validation perplexity: {best_ppl:.4f}")


if __name__ == "__main__":
    main()