import torch
from torch import Tensor,nn
from .pretokenization_example import find_chunk_boundaries
import regex as re
import os
import multiprocessing as mp
from collections import defaultdict
import json

GPT2_PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

def split_by_special_tokens(text:str,special_tokens:list[str] | None = None):
    
    # <spe><spe> > <spe>
    special_tokens = sorted(special_tokens,key=lambda x:-len(x))

    pattern = "(" + "|".join(re.escape(token) for token in special_tokens) + ")"

    parts = re.split(pattern,text)

    return  [part for part in parts if part]
    

def pretokenize(text:str,special_tokens:list[str],ingore_special_tokens=True) -> list[bytes]:
    

    # split by special tokens
    pretokens = split_by_special_tokens(text,special_tokens)

    tokens = []

    for token in pretokens:
        if token in special_tokens:
            if ingore_special_tokens == False:
                tokens.append(token.encode('utf-8'))
        else:
            token_list = re.findall(GPT2_PAT,token)
            tokens.extend([token_.encode("utf-8") for token_ in token_list])
    
    return tokens

def worker(text:str,special_tokens,q:mp.Queue):

    tokens = pretokenize(text,special_tokens)
    q.put(tokens)
def merge(counts: dict[tuple[int, int], int], index_dict: dict[tuple[int, int],set[int]], pretokens: list[list[int]], max_pair: tuple[int, int],new_idx:int):

    # find the token which need renew
    idx_set = index_dict[max_pair] # 只处理需要处理的部分

    for idx in idx_set:
        pretoken = pretokens[idx]

        pos_list = []
        new_pretoken = []
        j = 0
        while j < len(pretoken):
            if j < len(pretoken) - 1 and (pretoken[j],pretoken[j + 1]) == max_pair:
                new_pretoken.append(new_idx)
                pos_list.append(len(new_pretoken) - 1)
                j += 2
            else:
                new_pretoken.append(pretoken[j])
                j += 1

        for pos in pos_list:
            counts[max_pair] -= 1
            if pos > 0:
                if new_pretoken[pos - 1] == new_idx:
                    if (max_pair[1],max_pair[0]) in counts:
                        counts[max_pair[1],max_pair[0]] -= 1
                else:
                    if (new_pretoken[pos - 1],max_pair[0]) in counts:
                        counts[new_pretoken[pos - 1],max_pair[0]] -= 1
                counts[new_pretoken[pos - 1],new_pretoken[pos]] += 1
                index_dict[new_pretoken[pos - 1],new_pretoken[pos]].add(idx)
            if pos < len(new_pretoken) - 1:
                if (max_pair[1],new_pretoken[pos + 1]) in counts:
                    counts[max_pair[1],new_pretoken[pos + 1]] -= 1

                counts[new_pretoken[pos],new_pretoken[pos + 1]] += 1
                index_dict[new_pretoken[pos],new_pretoken[pos + 1]].add(idx)

        pretokens[idx] = new_pretoken   
        

def train_bpe(
    input_path: str | os.PathLike,
    vocab_size: int,
    special_tokens: list[str],
    **kwargs,
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    # initialize vocab and merges
    
    merges = []

    special_tokens = special_tokens or None

    vocab = {i:bytes([i]) for i in range(256)}

    for i in special_tokens:
        vocab[len(vocab)] = i.encode("utf-8")

    # load tokens list
    chunks = []
    with open(input_path,"rb") as f:
        num_processes = 4
        boundaries = find_chunk_boundaries(f, num_processes, b"<|endoftext|>")

        for st,end in zip(boundaries[:-1],boundaries[1:]):
            f.seek(st)
            chunk = f.read(end - st).decode("utf-8",errors="ignore")
            chunks.append(chunk)
    
    # pretoken 
    # split , RE
    q = mp.Queue()
    processes = [mp.Process(target=worker,args=(chunk,special_tokens,q)) for chunk in chunks]
    for p in processes:p.start()
    pretokens = [q.get() for _ in processes]
    for p in processes:p.join()

    pretokens = [b for token in pretokens for b in token ] # list[list[bytes]]
    # initialize count dict and pos
    counts = defaultdict(int)
    index_dict = defaultdict(set)

    for idx,token in enumerate(pretokens):
        for i in range(len(token) - 1):
            counts[token[i],token[i + 1]] += 1
            index_dict[token[i],token[i + 1]].add(idx)
    
    # print(counts,index_dict)

    # vocab train

    for i in range(256 + len(special_tokens),vocab_size):
        # find the max pair in counts
        max_pair = max(
            counts.items(),
            key=lambda x:(
            x[1],# compare the count
            vocab[x[0][0]], 
            vocab[x[0][1]]# 比较字典序
            )
        )[0]


        # renew vocab

        vocab[i] = vocab[max_pair[0]] + vocab[max_pair[1]]
        merges.append((vocab[max_pair[0]],vocab[max_pair[1]]))
        merge(counts,index_dict,pretokens,max_pair,i)
        # renew counts,index_dict
    return vocab,merges

# Tokenizer
class BPETokenizer:
    def __init__(self,vocab:tuple[dict[int, bytes]], merges:list[tuple[bytes, bytes]]):
        self.vocab = vocab
        self.merges = merges
    
    @classmethod
    def load_from_files(cls,file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    
        merges = [tuple(pair) for pair in data["merges"]]  # list → tuple
        # vocab 的 key 可能被 JSON 强制变成字符串，需要转回 int
        vocab = {int(k): v for k, v in data["vocab"].items()}
        cls(vocab,merges)
    
    def save_files(self,save_path = "./pre/bpe.json"):
        data = {
        "merges": [list(pair) for pair in self.merges],  # tuple 转 list
        "vocab": self.vocab
        }
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
from memory_profiler import profile
@profile
def main():
    special_tokens =  ["<|endoftext|>","<|endoftext|><|endoftext|>"]
    vocab,merges = train_bpe(
        input_path="./data/TinyStoriesV2-GPT4-train.txt",
        vocab_size=1000,
        special_tokens=["<|endoftext|>"],
    )
    tokenizer = BPETokenizer(vocab,merges)
    tokenizer.save_files()
if __name__ == "__main__":
    main()
    