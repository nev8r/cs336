import os
import heapq
from typing import BinaryIO
import regex as re
import collections
import multiprocessing as mp
import time
import pickle
from functools import reduce
from .pretokenization_example import find_chunk_boundaries
from collections import defaultdict

PAT = re.compile(r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


class Reverse_Pair:
    def __init__(self, pair: tuple[bytes, bytes]):
        self.pair = pair

    def __lt__(self, other: "Reverse_Pair") -> bool:
        # 反转正常顺序：如果 self > other，则 self < other
        return self.pair > other.pair

    def __eq__(self, other: "Reverse_Pair") -> bool:
        return self.pair == other.pair


def pretokenize_chunk(chunk: str, special_pattern: re.Pattern | None) -> dict[tuple[bytes], int]:
    freqs: dict[tuple[bytes], int] = {}

    sub_chunks = special_pattern.split(chunk) if special_pattern else [chunk]

    for sub_chunk in sub_chunks:
        for match in PAT.finditer(sub_chunk):
            match_bytes = tuple(bytes([b]) for b in match.group().encode("utf-8"))
            freqs[match_bytes] = freqs.get(match_bytes,0) + 1

    return freqs

def merge_freqs(dict1:dict[tuple[bytes], int],dict2:dict[tuple[bytes], int]) -> dict[tuple[bytes], int]:
    result = dict1.copy()
    for key, value in dict2.items():
        result[key] = result.get(key, 0) + value
    return result

def pretokenize(input_path:str,special_tokens:list[str]) -> dict[tuple[bytes], int]:
    num_processes = mp.cpu_count()

    pool = mp.Pool(processes=num_processes)

    chunk_freqs = []
    special_pattern = re.compile(
        "|".join(re.escape(token) for token in special_tokens)
    )

    with open(input_path,"rb") as f:
        boundaries = find_chunk_boundaries(f,num_processes,b"<|endoftext|>")

        for st,end in zip(boundaries[:-1],boundaries[1:]):
            f.seek(st)
            chunk = f.read(end - st).decode("utf-8",errors = "ignore")
            chunk_freqs.append(pool.apply_async(pretokenize_chunk,(chunk,special_pattern)))

    pool.close()
    pool.join()

    freq_dicts = [res.get() for res in chunk_freqs]
    freqs = reduce(merge_freqs,freq_dicts,{})
    return freqs

def get_pair_freqs(
    freqs: dict[tuple[bytes], int]
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    pair_freqs = defaultdict(int)
    pair_to_keys = defaultdict(set)

    for token,freq in freqs.items():
        for b1,b2 in zip(token[:-1],token[1:]):
            pair = (b1,b2)
            pair_freqs[pair] += freq
            pair_to_keys[pair].add(token)

    return pair_freqs,pair_to_keys

def build_new_repr(old_repr: tuple[bytes], pair: tuple[bytes, bytes]) -> tuple[bytes]:
    """将 old_repr 中每个 pair=(x,y) 替换为合并符号 x+y"""
    new_symbols = []
    i = 0
    while i < len(old_repr):
        if i < len(old_repr) - 1 and old_repr[i] == pair[0] and old_repr[i + 1] == pair[1]:
            new_symbols.append(old_repr[i] + old_repr[i + 1])  # 合并，例如 b'A' + b'B' => b'AB'
            i += 2
        else:
            new_symbols.append(old_repr[i])
            i += 1
    return tuple(new_symbols)

def merge(
    freqs: dict[tuple[bytes], int],
    pair_freqs: dict[tuple[bytes, bytes], int],
    pairs_to_keys: dict[tuple[bytes, bytes], set[tuple[bytes]]],
    pair: tuple[bytes, bytes],
) -> set[tuple[bytes, bytes]]:
    
    changed_pairs = set()
    keys_need_modify = pairs_to_keys[pair].copy()

    for old_key in keys_need_modify:
        old_freq = freqs.pop(old_key)
        new_key = build_new_repr(old_key, pair)

        for i in range(len(old_key) - 1):
            left, right = old_key[i],old_key[i + 1]
            pair_freqs[left,right] -= old_freq
            changed_pairs.add((left,right))
            if pair_freqs[left, right] <= 0:
                del pair_freqs[left, right]
            pairs_to_keys[left,right].discard(old_key)
        
        for i in range(len(new_key) - 1):
            left, right = new_key[i],new_key[i + 1]
            pair_freqs[left,right] += old_freq
            changed_pairs.add((left,right))
            pairs_to_keys[left,right].add(new_key)

        freqs[new_key] = freqs.get(new_key, 0) + old_freq
        
    pairs_to_keys[pair] = set()
    return changed_pairs

def write_merges(merges, outpath):
    """将 merges 列表序列化并写入二进制文件"""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(merges, f)
    print(f"已保存 {len(merges)} 个合并到 {outpath}")


def write_vocab(vocab, outpath):
    """将 vocab 字典序列化并写入二进制文件"""
    os.makedirs(os.path.dirname(outpath), exist_ok=True)
    with open(outpath, "wb") as f:
        pickle.dump(vocab, f)
    print(f"已保存 {len(vocab)} 个词到 {outpath}")


def train_bpe(
    input_path: str,
    vocab_size: int,
    special_tokens: list[str],
    merges_outpath: str = None,
    vocab_outpath: str = None,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    
    train_start_time = time.time()

    # initialize vocab
    _tokens = [
        token.encode("utf-8") for token in special_tokens
    ] + [bytes([b]) for b in range(256)]

    vocab = {
        i:token for i,token in enumerate(_tokens)
    }
    merges = []

    print("分词预处理: 开始")
    start_time = time.time()
    freqs = pretokenize(input_path, special_tokens)
    print(f"分词预处理: 完成，耗时 {time.time() - start_time:.2f}s")


    print("计算初始对频率: 开始")
    start_time = time.time()
    pair_freqs, pairs_to_keys = get_pair_freqs(freqs)

    pair_heap = []
    for pair,freq in pair_freqs.items():
        heapq.heappush(pair_heap,(-freq,Reverse_Pair(pair),pair))

    print(f"计算初始对频率: 完成，耗时 {time.time() - start_time:.2f}s")

    print("合并训练: 开始")
    start_time = time.time()

    st = len(_tokens)

    for i in range(st,vocab_size):
        if not pair_heap:break

        while pair_heap:
            neg_freq,_,top_pair = heapq.heappop(pair_heap)
            freq = -neg_freq
            if pair_freqs.get(top_pair,0) == freq:
                pair = top_pair
                break
            # if top_pair in pair_freqs and pair_freqs[top_pair] > 0: # lazy_update
            #     heapq.heappush(pair_heap, (-pair_freqs[top_pair], Reverse_Pair(top_pair), top_pair))
        else:
            break

        if pair_freqs.get(pair, 0) <= 0:
            break

        vocab[i] = pair[0] + pair[1]

        merges.append(pair)

        changed_pairs = merge(freqs,pair_freqs,pairs_to_keys,pair)

        for cp in changed_pairs:
            if cp in pair_freqs and pair_freqs[cp] > 0:
                heapq.heappush(pair_heap, (-pair_freqs[cp], Reverse_Pair(cp), cp))

                # 每 100 次合并或最后一次打印进度
        if ((i > st) and ((i - st + 1) % 100 == 0)) or (
            i == vocab_size - 1
        ):
            print(
                f"{i - st + 1}/{vocab_size - st} 次合并完成 (合并耗时: {time.time() - start_time:.2f}s)"
            )

    print(f"合并完成，耗时 {time.time() - start_time:.2f}s")
    print(f"训练完成，总耗时 {time.time() - train_start_time:.2f}s")

            # 可选保存 merges 和 vocab
    if merges_outpath:
        write_merges(merges, merges_outpath)
    if vocab_outpath:
        write_vocab(vocab, vocab_outpath)

    return vocab, merges
if __name__ == "__main__":
    (vocab, merges) = train_bpe(
        input_path="./data/test.txt",
        vocab_size=10000,
        special_tokens=["<|endoftext|>"],
        # merges_outpath="./out/ts-train-merges-2.txt",
        # vocab_outpath="./out/ts-train-vocab-2.txt",
    )

    print(vocab,merges)
