from typing import Tuple
import gin
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.modules.transformer import Transformer
# try:
#     from flash_attn_interface import flash_attn_func  # type: ignore[import]
# except ImportError:
#     # Fallback to FlashAttention 2
#     from flash_attn import flash_attn_func  # type: ignore[import]

from custom.hrm_utils.common import trunc_normal_init_


CosSin = Tuple[torch.Tensor, torch.Tensor]


def _find_multiple(a, b):
    return (-(a // -b)) * b


def rotate_half(x: torch.Tensor):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_pos_emb(q: torch.Tensor, k: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    # q, k: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = q.dtype
    q = q.to(cos.dtype)
    k = k.to(cos.dtype)

    q_embed = (q * cos.unsqueeze(-2)) + (rotate_half(q) * sin.unsqueeze(-2))
    k_embed = (k * cos.unsqueeze(-2)) + (rotate_half(k) * sin.unsqueeze(-2))

    return q_embed.to(orig_dtype), k_embed.to(orig_dtype)


class CastedLinear(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 bias: bool=False):
        super().__init__()
        # Truncated LeCun normal init
        self.weight = nn.Parameter(
            trunc_normal_init_(torch.empty((out_features, in_features)), std=1.0 / (in_features ** 0.5))
        )
        self.bias = None
        if bias:
            # Zero init bias
            self.bias = nn.Parameter(torch.zeros((out_features, )))

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight.to(input.dtype), bias=self.bias.to(input.dtype) if self.bias is not None else None)


class CastedEmbedding(nn.Module):
    def __init__(self,
                 num_embeddings: int,
                 embedding_dim: int,
                 init_std: float,
                 cast_to: torch.dtype):
        super().__init__()
        self.cast_to = cast_to

        # Truncated LeCun normal init
        self.embedding_weight = nn.Parameter(
            trunc_normal_init_(torch.empty((num_embeddings, embedding_dim)), std=init_std)
        )
        
    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.embedding(input, self.embedding_weight.to(self.cast_to))

@gin.configurable
class RotaryEmbedding(nn.Module):
    def __init__(self, hidden_dim: int= 128, num_heads: int=4,  max_position_embeddings=10, base=1e5, device=None):
        super().__init__()

        # RoPE
        dim = ((hidden_dim)//num_heads)
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)

        # Different from paper, but it uses a different permutation in order to obtain the same calculation
        emb = torch.cat((freqs, freqs), dim=-1)
        self.cos_cached = nn.Buffer(emb.cos(), persistent=False)
        self.sin_cached = nn.Buffer(emb.sin(), persistent=False)

    def forward(self):
        return self.cos_cached, self.sin_cached

class Attention(nn.Module):
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        qkv = self.qkv_proj(hidden_states)

        # Split head
        qkv = qkv.view(batch_size, seq_len, self.num_heads + 2 * self.num_key_value_heads, self.head_dim)
        query = qkv[:, :, :self.num_heads]
        key = qkv[:, :, self.num_heads: self.num_heads + self.num_key_value_heads]
        value = qkv[:, :, self.num_heads + self.num_key_value_heads:]

        # RoPE 
        if cos_sin is not None:
            seq_len = query.shape[1]
            cos, sin = cos_sin
            query, key = apply_rotary_pos_emb(query, key, cos[:seq_len], sin[:seq_len])

        # flash attn
        # attn_output = flash_attn_func(q=query, k=key, v=value, causal=self.causal)

        # flash attn
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=self.causal)

        if isinstance(attn_output, tuple):  # fa2 and fa3 compatibility
            attn_output = attn_output[0]

        # attn_output: [batch_size, num_heads, seq_len, head_dim]
        attn_output = attn_output.transpose(1, 2)    # [batch_size, seq_len, num_heads, head_dim]
        attn_output = attn_output.reshape(batch_size, seq_len, self.output_size)  # type: ignore
        return self.o_proj(attn_output)


class SwiGLU(nn.Module):
    def __init__(self, hidden_size: int, expansion: float):
        super().__init__()
        inter = _find_multiple(round(expansion * hidden_size * 2 / 3), 256)

        self.gate_up_proj = CastedLinear(hidden_size, inter * 2, bias=False)
        self.down_proj    = CastedLinear(inter, hidden_size, bias=False)

    def forward(self, x):
        gate, up = self.gate_up_proj(x).chunk(2, dim=-1)
        return self.down_proj(F.silu(gate) * up)


def rms_norm(hidden_states: torch.Tensor, variance_epsilon: float=1e-7) -> torch.Tensor:
    input_dtype = hidden_states.dtype
    hidden_states = hidden_states.to(torch.float32)

    variance = hidden_states.square().mean(-1, keepdim=True)
    hidden_states = hidden_states * torch.rsqrt(variance + variance_epsilon)
    return hidden_states.to(input_dtype)


def apply_rotary_pos_emb_single(t: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """Applies rotary positional embedding to a single tensor."""
    # t: [bs, seq_len, num_heads, head_dim]
    # cos, sin: [seq_len, head_dim]
    orig_dtype = t.dtype
    t = t.to(cos.dtype)
    t_embed = (t * cos.unsqueeze(-2)) + (rotate_half(t) * sin.unsqueeze(-2))
    return t_embed.to(orig_dtype)


class CrossAttBlock(nn.Module):
    def __init__(self,
                 dim_zH: int,
                 dim_x: int,
                 num_heads: int = 4,
                 num_key_value_heads: int = 1,
                 ffn_expansion: float = 2.0,
                 dropout: float = 0.1,
                 eps: float = 1e-6):
        super().__init__()
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_key_value_groups = num_heads // num_key_value_heads
        self.head_dim = dim_zH // num_heads
        self.variance_epsilon = eps
        # Attention projections using CastedLinear
        self.q_proj = CastedLinear(dim_zH, num_heads * self.head_dim, bias=False)
        self.k_proj = CastedLinear(dim_x, num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = CastedLinear(dim_x, num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = CastedLinear(num_heads * self.head_dim, dim_zH, bias=False)

        # SwiGLU Feed-forward network
        self.ffn = SwiGLU(hidden_size=dim_zH, expansion=ffn_expansion)
        self.dropout = nn.Dropout(dropout)

    def forward(self, zH: torch.Tensor, x: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        # zH: [B, T1, dim_zH] (Query source)
        # x:  [B, T2, dim_x] (Key/Value source)
        # cos_sin: Tuple of cos and sin tensors for RoPE

        residual = zH
        zH_norm = rms_norm(zH, self.variance_epsilon)
        B, T1, _ = zH.shape
        _, T2, _ = x.shape
        q = self.q_proj(zH_norm).view(B, T1, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(B, T2, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(x).view(B, T2, self.num_key_value_heads, self.head_dim)
        
        # Apply Rotary Positional Embeddings to q and k independently
        cos, sin = cos_sin
        q = apply_rotary_pos_emb_single(q, cos[:T1], sin[:T1])
        k = apply_rotary_pos_emb_single(k, cos[:T2], sin[:T2])
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)

        if self.num_key_value_groups > 1:
            k = k.repeat_interleave(self.num_key_value_groups, dim=1)
            v = v.repeat_interleave(self.num_key_value_groups, dim=1)

        context = F.scaled_dot_product_attention(q, k, v)
        context = context.transpose(1, 2).contiguous().view(B, T1, -1)
        attention_output = self.out_proj(context)

        residual =residual+ self.dropout(attention_output)

        zH_norm = rms_norm(residual, self.variance_epsilon)
        ffn_output = self.ffn(zH_norm)
        
        zH_updated = residual + self.dropout(ffn_output)
        
        return zH_updated
    
    
@gin.configurable
class ChunkRepresentativeAttention(nn.Module):
    def __init__(self, reasoning_tokens: int, in_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        self.input_dim = reasoning_tokens * in_dim
        self.adapter = nn.Linear(self.input_dim, self.input_dim)
        
        # is_causal=True automatically applies the causal mask (PyTorch >= 1.12)
        self.attention = nn.MultiheadAttention(
            embed_dim=self.input_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True    # because our input is [B, N, ...]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, Z, D = x.shape
        x_flat = x.view(B, N, -1)

        x_adapted = self.adapter(x_flat)
        x_norm = rms_norm(x_adapted)
        causal_mask = Transformer.generate_square_subsequent_mask(N).to(x.device)
        # --- Causal Self-Attention ---
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask, 
            is_causal=True,
            need_weights=False # We usually don't need the weights for the forward pass
        )
        contextualized_reps_flat = x_adapted + attn_output
        return contextualized_reps_flat.view(B, N, Z, D)