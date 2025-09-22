import torch
from torch import nn,Tensor
from jaxtyping import Bool, Float, Int
import math

def softmax(x:Tensor,dim:int=-1):
    x_exp = torch.exp(x - x.max())
    return x_exp / torch.sum(x_exp,dim=-1,keepdim=True)

class Linear(nn.Module):

    def __init__(self, in_features, out_features, device=None, dtype=None):

        super().__init__()

        std = math.sqrt(2 / (in_features + out_features))
        mean = 0
        w = torch.empty((out_features,in_features),device=device,dtype=dtype)

        torch.nn.init.trunc_normal_(w,mean,std,-3 * std,3 * std)

        self.weights = nn.Parameter(w)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weights.T


class Embedding(nn.Module):

    def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings # vocab_size
        self.embedding_dim = embedding_dim # d_model

        self.device = device
        self.dtype =dtype

        w = torch.empty((num_embeddings,embedding_dim),device=device,dtype=dtype)

        nn.init.trunc_normal_(w,0,1,-3,3)

        self.weights = nn.Parameter(w)

    def forward(self,token_ids:Tensor) -> Tensor:
        return self.weights[token_ids.int()]
        
class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.d_model = d_model
        self.weights = nn.Parameter(torch.ones(d_model))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype

        rms_x = torch.sqrt((x * x).mean(dim=-1,keepdim=True) + self.eps)
        res =  x  * self.weights / rms_x

        return res.to(in_dtype)
    

def silu(x:Tensor):
    return x  * torch.sigmoid(x)

class SwiGLU(nn.Module):
    def __init__(self,d_ff:int,d_model:int):
        super().__init__()
        self.d_ff = d_ff
        self.d_model = d_model
        self.w1 = Linear(d_model,d_ff)
        self.w2 = Linear(d_ff,d_model)
        self.w3 = Linear(d_model,d_ff)
    def forward(self,x:Tensor):
        SiLU = silu(self.w1(x))
        GLU =  self.w2(SiLU * self.w3(x))
        return GLU
    

class ROPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        freqs = torch.arange(0,d_k,2)

        inv_freqs = torch.exp(-math.log(theta) * freqs / d_k)

        positions = torch.arange(max_seq_len, device=device)
        angles = torch.outer(positions,inv_freqs)

        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)


    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        
        if token_positions is None:
            token_positions = torch.arange(0,x.size(-2))
        
        cos = self.cos[token_positions] # (seq_len, d_k/2)
        sin = self.sin[token_positions] # (seq_len, d_k/2)

        x_even = x[...,::2] # (batch,seq_len,d_k/2)
        x_odd = x[...,1::2] # (batch,seq_len,d_k/2)

        x_rotated_even = x_even * cos - x_odd * sin # (batch,seq_len,d_k/2)
        x_rotated_odd = x_even * sin + x_odd * cos  # (batch,seq_len,d_k/2)                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            

        x_rotated = torch.stack((x_rotated_even,x_rotated_odd),dim=-1).flatten(-2) # (batch,seq_len,d_k)
        

        return x_rotated


class Attention(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self,Q,K,V,mask:Tensor|None=None) -> Tensor:
        d_k = Q.size(-1)
        scores = (Q @ K.transpose(-2,-1) ) / math.sqrt(d_k)
        if mask is not None:
            scores = scores.masked_fill(mask.to(scores.device)==False,float('-inf'))
        attn = softmax(scores,dim=-1)
        out = attn @ V
        return out
    
class MultiHeadSelfAttention(nn.Module):
    
    def __init__(self,d_model:int,num_heads:int,use_rope=False,theta=10000,max_seq_len:int|None=None):
        super().__init__()
        assert d_model % num_heads == 0,"d_model % num_heads != 0"
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.w_q = Linear(d_model,d_model)
        self.w_k = Linear(d_model,d_model)
        self.w_v = Linear(d_model,d_model)
        self.w_o = Linear(d_model,d_model)
        self.use_rope = use_rope
        self.attn = Attention()

        if self.use_rope:
            self.rope = ROPE(theta,self.head_dim,max_seq_len)
    
    def forward(self,x:Tensor):
        Q,K,V = self.w_q(x),self.w_k(x),self.w_v(x) # (batch,seq_len,d_model)

        batch_size,seq_len = x.size(-3),x.size(-2)

        

        Q = Q.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2) 
        K = K.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)
        V = V.view(batch_size,seq_len,self.num_heads,self.head_dim).transpose(1,2)

        mask = ~torch.triu(torch.ones((seq_len,seq_len),dtype=torch.bool),diagonal=1)

        if self.use_rope:
            Q = self.rope(Q)
            K = self.rope(K)

        scores = self.attn(Q,K,V,mask)

        scores = scores.transpose(1,2).contiguous().view(batch_size,seq_len,self.d_model)
        
        out = self.w_o(scores)

        return out


class TransfomerBlock(nn.Module):

    def __init__(self,
                d_model: int,
                num_heads: int,
                d_ff: int,
                max_seq_len: int,
                theta: float
                 ):
        super().__init__()
        self.rmsnorm1 = RMSNorm(d_model)
        self.rmsnorm2 = RMSNorm(d_model)
        self.mha = MultiHeadSelfAttention(d_model,num_heads,True,theta,max_seq_len)
        self.swiglu = SwiGLU(d_ff,d_model)

    def forward(self,x:Tensor):
        score = self.mha(self.rmsnorm1(x))

        x = score + x

        return x + self.swiglu(self.rmsnorm2(x))
    

class TransformerLM(nn.Module):

    def __init__(
            self,
            vocab_size: int,
            context_length: int,
            d_model: int,
            num_layers: int,
            num_heads: int,
            d_ff: int,
            rope_theta: float,
    ):
        super().__init__()
        self.emb = Embedding(vocab_size,d_model)
        self.tfbs = nn.ModuleList(
            [TransfomerBlock(d_model,num_heads,d_ff,context_length,rope_theta) for _ in range(num_layers)]
        )

        self.rmsnorm = RMSNorm(d_model)
        self.lm_head = Linear(d_model,vocab_size)

    def forward(self,x:Tensor):
        x = self.emb(x)

        for tfb in self.tfbs:
            x = tfb(x)
            
        x = self.rmsnorm(x)

        return self.lm_head(x)
