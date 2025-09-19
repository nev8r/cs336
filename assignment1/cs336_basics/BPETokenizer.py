import pickle
from cs336_basics.train_bpe import *
import regex as re
import heapq
from collections.abc import Iterable, Iterator



class BPETokenizer:

    def __init__(self,vocab:dict[int, bytes],merges:list[tuple[bytes,bytes]],special_tokens:list[str]|None):

        self.vocab = vocab
        self.merges = merges
        self.special_tokens = sorted(special_tokens,key = lambda x:-len(x)) if special_tokens is not None else None

        self.special_pattern = re.compile(
            "(" + "|".join(re.escape(tok) for tok in self.special_tokens) + ")"
        ) if special_tokens is not None else None
        self.merge_idx = {
             pair:i for i,pair in enumerate(merges)
        }
        self.token_to_idx = {
            tok:i for i,tok in vocab.items()
        }

        self.token_encode_cache = {}
        if special_tokens:
            for spe in special_tokens:
                spe = spe.encode("utf-8")
                if spe not in self.token_to_idx:
                    self.token_to_idx[spe] = len(self.vocab)
                    self.vocab[len(vocab)] = spe

    @classmethod
    def load_from_file(cls,
        vocab_input_path:str,
        merges_input_path:str,
        special_tokens:list[str]|None = None
    ):
        with open(vocab_input_path,"rb") as f:
            vocab = pickle.load(f)

        with open(merges_input_path,"rb") as f:
            merges = pickle.load(f)

        return cls(vocab,merges,special_tokens)
    
    def _pre_tokenize(self,text:str) -> list[str]:
        sub_chunks = self.special_pattern.split(text) if text and self.special_tokens else [text]
        tokens = []
        for sub_chunk in sub_chunks:
            if self.special_tokens and sub_chunk in self.special_tokens:
                tokens.append(sub_chunk)
            else:
                tokens.extend([match.group() for match in PAT.finditer(sub_chunk)])
        return tokens
                
    
    def encode_token(self,tok:list[bytes]) -> list[int]:
        otk = str(tok).encode("utf-8")
        while True:
            best_rank = float('inf')
            best_idx = None
            
            for i in range(len(tok) - 1):
                pair = (tok[i],tok[i + 1])
                rank = self.merge_idx.get(pair)

                if rank is not None and rank < best_rank:
                    best_rank = rank
                    best_idx = i 
                
            if best_idx == None:
                break
            
            tok = tok[:best_idx] + [tok[best_idx] + tok[best_idx + 1]] + tok[best_idx+2:]

        ids = [self.token_to_idx.get(rep) for rep in tok]
        self.token_encode_cache[otk] = ids
        return ids

    def encode(
        self,
        text:str
    ) -> list[int]:
        # pre-tokenize
        pretokens = self._pre_tokenize(text)
        # apply the merges，按照merges中的优先级进行 merge
        token_ids = []
        for tok in pretokens:
            if self.special_tokens and tok in self.special_tokens:
                token_ids.append(self.token_to_idx.get(tok.encode("utf-8")))
            elif tok in self.token_encode_cache:
                token_ids.extend(self.token_encode_cache.get(tok.encode("utf-8")))
            else:
                tok = [bytes([b]) for b in tok.encode("utf-8")]
                token_ids.extend(self.encode_token(tok))

        return token_ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            yield from self.encode(text)

    def decode(self,token_ids:list[int]):
        text = b"".join(self.vocab[id] for id in token_ids)
        return text.decode("utf-8", errors="replace")

def main():
    special_tokens = ["<|endoftext|>","<|endoftext|><|endoftext|>"]
    bpe = BPETokenizer.load_from_file(
        "./out/ts-valid-vocab-2.txt",
        "./out/ts-valid-merges-2.txt",
        special_tokens
    )

    ids = bpe.encode("Hello, how <|endoftext|><|endoftext|> are you?<|endoftext|>")

    print(ids)
    tokenized_string = [bpe.decode([x]) for x in ids]
    print(tokenized_string)
if __name__ == "__main__":
        main()