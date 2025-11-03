from typing import Tuple, List,Optional
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

def init_scaled_linear(module: nn.Module, scale_factor: float):
    """
    Initializes a linear layer's weights with a scaled normal distribution.

    This is ideal for output/projection layers in residual blocks of a
    Pre-LN Transformer to ensure stability.

    Args:
        module (nn.Module): The linear layer to be initialized.
        scale_factor (float): The value to scale the standard deviation by.
                              Typically `1.0 / math.sqrt(2 * num_layers)`.
    """
    if hasattr(module, 'weight'):
        # Use a baseline std of 0.02, a common practice for transformers
        std = 0.02 * scale_factor
        nn.init.normal_(module.weight, mean=0.0, std=std)
    
    # Initialize bias to zero, if it exists
    if hasattr(module, 'bias') and module.bias is not None:
        nn.init.zeros_(module.bias)


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


class LayerScale(nn.Module):
    def __init__(self, dim: int, init_values: float = 5e-3):
        super().__init__()
        # Create a learnable parameter vector of size `dim`
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Element-wise multiplication
        return x * self.gamma

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
    def __init__(self, hidden_dim: int= 128, num_heads: int=4,  
                 max_position_embeddings=10, base=1e5, device=None):
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
    def __init__(self, hidden_size, head_dim, num_heads, num_key_value_heads, weight_init_scale, causal=False):
        super().__init__()

        self.hidden_size = hidden_size
        self.head_dim = head_dim
        self.output_size = head_dim * num_heads
        self.num_heads = num_heads
        self.num_key_value_heads = num_key_value_heads
        self.causal = causal

        self.qkv_proj = CastedLinear(self.hidden_size, (self.num_heads + 2 * self.num_key_value_heads) * self.head_dim, bias=False)
        self.o_proj = CastedLinear(self.output_size, self.hidden_size, bias=False)
        init_scaled_linear(self.o_proj, weight_init_scale)
        self.pre_attn_norm = nn.LayerNorm(hidden_size)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = hidden_states.shape

        # hidden_states: [bs, seq_len, num_heads, head_dim]
        hidden_states=self.pre_attn_norm(hidden_states)
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
            
        attn_output = F.scaled_dot_product_attention(query, key, value, is_causal=self.causal)

        if isinstance(attn_output, tuple):  
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
                 weight_init_scale: float=1,
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
        init_scaled_linear(self.out_proj, weight_init_scale)
        
        # SwiGLU Feed-forward network
        self.ffn = SwiGLU(hidden_size=dim_zH, expansion=ffn_expansion)
        init_scaled_linear(self.ffn.down_proj, weight_init_scale)
        self.dropout = nn.Dropout(dropout)
        self.pre_attn_norm = nn.LayerNorm(dim_zH, eps=eps)
        self.pre_mlp_norm = nn.LayerNorm(dim_zH, eps=eps)
        
        self.gamma_1 = LayerScale(dim_zH)
        self.gamma_2 = LayerScale(dim_zH)

    def forward(self, zH: torch.Tensor, x: torch.Tensor, cos_sin: CosSin) -> torch.Tensor:
        # zH: [B, T1, dim_zH] (Query source)
        # x:  [B, T2, dim_x] (Key/Value source)
        # cos_sin: Tuple of cos and sin tensors for RoPE

        residual_attn  = zH
        zH_norm = self.pre_attn_norm(zH)
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
        zH = residual_attn + self.dropout(self.gamma_1(attention_output))
        
        zH_norm = self.pre_mlp_norm(zH)
        residual_ffn = zH
        ffn_output = self.ffn(zH_norm)
        zH_updated = residual_ffn + self.dropout(self.gamma_2(ffn_output))
        
        return zH_updated
    

class Conv1DNormPoolBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)        
        self.norm = nn.LayerNorm(out_channels) 
        self.prelu = nn.PReLU()
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        
    def forward(self, x: torch.Tensor):
        x = self.conv(x)        
        x = x.transpose(1, 2)  # (B, Z', D) -> (B, D, Z')
        x = self.norm(x)  # Apply LayerNorm over Z' (The Channel dimension)
        x = x.transpose(1, 2)  # (B, D, Z') -> (B, Z', D)
        x = self.prelu(x)
        x = self.pool(x) 
        return x

class ChunkRepresentativeAttentionBlock(nn.Module):
    def __init__(self, in_dim: int, 
                 num_heads: int, dropout: float = 0.1):
        super().__init__()        
        # is_causal=True automatically applies the causal mask (PyTorch >= 1.12)
        self.attention = nn.MultiheadAttention(
            embed_dim=in_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True    # because our input is [B, N, ...]
        )
        self.ffn = SwiGLU(hidden_size=in_dim, expansion=1.2)
        self.pre_attn_norm = nn.LayerNorm(in_dim)
        self.pre_mlp_norm = nn.LayerNorm(in_dim)

    def forward(self, x: torch.Tensor,  causal_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x_norm = self.pre_attn_norm(x)
        # --- Causal Self-Attention ---
        attn_output, _ = self.attention(
            query=x_norm,
            key=x_norm,
            value=x_norm,
            attn_mask=causal_mask, 
            is_causal=True,
            need_weights=False # We usually don't need the weights for the forward pass
        )
        contextualized_reps_flat = x + attn_output
        
        residual_ffn = contextualized_reps_flat
        ffn_input_norm = self.pre_mlp_norm(residual_ffn)
        ffn_output = self.ffn(ffn_input_norm)
        final_output_flat = residual_ffn + ffn_output

        return final_output_flat
    
@gin.configurable()
class ChunkRepresentativeAttention(nn.Module):
    def __init__(self, reasoning_tokens: int, in_dim: int, num_heads: int, channel_dims: List[int] ,
                 num_layers: int=4, dropout: float = 0.1):
        super().__init__()
        self.num_layers = num_layers
        self.N = None

        # Atten based temporal fusion layers
        self.layers = nn.ModuleList([
            ChunkRepresentativeAttentionBlock(
                in_dim=in_dim, 
                num_heads=num_heads, 
                dropout=dropout
            )
            for _ in range(num_layers)
        ])
        
        # dim reduction (merging : [B, N, Z, D] -> [B, N, D'])
        dims = [reasoning_tokens] + channel_dims 
        reduction_layers = []
        assert in_dim==channel_dims[-1] , "The last channel dim must be equal to in_dim"
        for i in range(len(dims) - 1):
            reduction_layers.append(Conv1DNormPoolBlock(dims[i], dims[i + 1]))
        reduction_layers.append(nn.AdaptiveAvgPool1d(1))
        self.dim_reduction = nn.Sequential(*reduction_layers)
      
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, Z, D = x.shape
        original_x = x
        x_flat = x.view(B * N, Z, D) 
        x_reduced = self.dim_reduction(x_flat).squeeze(-1) # Shape (B*N, Z')
        x_reduced = x_reduced.view(B, N, D)    # we asserted D==channel_dims[-1]
        
        x=x_reduced
        # Generate the causal mask once for all layers
        # The mask will be a [N, N] tensor
        causal_mask = Transformer.generate_square_subsequent_mask(N).to(x.device)

        # Pass the input through all stacked attention layers
        for layer in self.layers:
            x = layer(x, causal_mask=causal_mask)

        x= torch.cat([original_x, x.unsqueeze(2)], dim=2) # (B, N, Z+1, D)

        return x 

    
    
class ResidualBlock(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        # Ensure the input and output features are the same for the residual connection
        if in_features != out_features:
            raise ValueError("In a standard ResidualBlock, in_features must equal out_features.")

        self.block = nn.Sequential(
            nn.Linear(in_features, out_features),
            nn.ReLU(),
            nn.Linear(out_features, out_features)
        )
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(x + self.block(x))

@gin.configurable()
class MultiGammaValueEstimator(nn.Module):
    """
    A PyTorch module using a Residual Network (ResNet) architecture to estimate 
    the value function for multiple discount factors (gammas).

    The use of residual blocks makes the network more robust and easier to train 
    when using deeper architectures.

    Input Shapes:
    - g_: (B, dim1)
    - vae_input: (B, dim2)

    Output Shape:
    - value_estimates: (B, G), where G = num_gammas
    """

    def __init__(self, subgoal_dim: int, state_dim: int, num_gammas: int, 
                 hidden_dim: int = 256, num_blocks: int = 3):
        """
        Initializes the value estimator network using Residual Blocks.

        Args:
            subgoal_dim (int): Dimension of the first state component (g_).
            state_dim (int): Dimension of the second state component (vae_input).
            num_gammas (int): The number of discount factors (G) for which to estimate the value.
            hidden_dim (int): The consistent dimension used within all hidden layers and blocks.
            num_blocks (int): The number of Residual Blocks to stack.
        """
        super().__init__()
        self.input_dim = state_dim+ subgoal_dim
        self.num_gammas = num_gammas
        self.hidden_dim = hidden_dim

        # 1. Input Projection Layer
        # Maps the concatenated state to the uniform hidden_dim for the residual blocks
        self.input_layer = nn.Sequential(
            nn.Linear(self.input_dim, hidden_dim),
            nn.ReLU()
        )

        # 2. Stack of Residual Blocks
        # All residual blocks maintain the hidden_dim size
        residual_blocks = []
        for _ in range(num_blocks):
            residual_blocks.append(ResidualBlock(hidden_dim, hidden_dim))
        
        self.residual_stack = nn.Sequential(*residual_blocks)

        # 3. Output Layer
        # Maps the final hidden state to the required number of gamma value estimates (G)
        self.output_layer = nn.Linear(hidden_dim, num_gammas)

        # Initialize weights for stability
        self._initialize_weights()

    def _initialize_weights(self):
        """Custom weight initialization for better training start."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Using Kaiming Uniform for layers followed by ReLU
                nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            # The ResidualBlock already handles internal initialization

    def forward(self, g_: torch.Tensor, vae_input: torch.Tensor) -> torch.Tensor:
        """
        Passes the state components through the ResNet structure.

        Args:
            g_ (torch.Tensor): Tensor representing the first state component, shape (B, dim1).
            vae_input (torch.Tensor): Tensor representing the second state component, shape (B, dim2).

        Returns:
            torch.Tensor: Value estimates for all gammas, shape (B, G).
        """
        if g_.shape[0] != vae_input.shape[0]:
            raise ValueError(
                f"Batch sizes must match. Got g_: {g_.shape[0]}, vae_input: {vae_input.shape[0]}"
            )
        # 1. Combine the state components
        vae_input= torch.mean(vae_input, dim=1) # (B, scratch-token, dim)
        state_combined = torch.cat([g_, vae_input], dim=-1) # (B, dim1 + dim2)

        # 2. Input Projection
        h = self.input_layer(state_combined) # (B, hidden_dim)

        # 3. Pass through Residual Blocks
        h = self.residual_stack(h) # (B, hidden_dim)

        # 4. Final Value Prediction
        value_estimates = self.output_layer(h) # (B, G)
        return value_estimates



class ResidualBlockV2(nn.Module):
    """
    A residual block featuring Layer Normalization, LeakyReLU activation, and Dropout.
    """
    def __init__(self, hidden_dim: int, dropout_rate: float = 0.1, leaky_relu_slope: float = 0.01):
        super().__init__()
        self.main_path = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        self.final_activation = nn.LeakyReLU(leaky_relu_slope, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        processed_x = self.main_path(x)
        return self.final_activation(x + processed_x)

@gin.configurable()
class MultiGammaValueEstimatorV3(nn.Module):
    """
    An improved PyTorch module to estimate the value function for multiple gammas.

    This version uses a convolutional front-end to process a sequential input,
    extracting features before feeding them into a residual MLP.

    Improvements:
    1.  **Convolutional Feature Extractor**: Processes a sequence input via a 1D CNN
        to create a rich, fixed-size feature vector.
    2.  **Layer/Batch Normalization**: Stabilizes training.
    3.  **LeakyReLU Activation**: Mitigates the "dying ReLU" problem.
    4.  **Dropout Regularization**: Prevents overfitting.
    """
    def __init__(
        self,
        subgoal_dim: int,
        sequence_input_shape: Tuple[int, int], # (num_tokens, token_dim)
        num_gammas: int,
        hidden_dim: int = 256,
        num_blocks: int = 4,
        dropout_rate: float = 0.1,
        leaky_relu_slope: float = 0.01,
        conv_channels: List[int] = [64, 128],
        conv_kernel_size: int = 3
    ):
        
        """
        Args:
            subgoal_dim (int): Dimension of the g_ state component.
            sequence_input_shape (Tuple[int, int]): Shape of the sequential input (num_tokens, token_dim).
            num_gammas (int): The number of discount factors (G) to estimate.
            hidden_dim (int): Dimension for the main MLP and residual blocks.
            num_blocks (int): The number of Residual Blocks to stack.
            dropout_rate (float): Dropout probability.
            leaky_relu_slope (float): Negative slope for LeakyReLU.
            conv_channels (List[int]): Output channels for the 1D conv layers.
            conv_kernel_size (int): Kernel size for the 1D conv layers.
        """
        super().__init__()
        num_tokens, _ = sequence_input_shape

        # 1. Convolutional Processor for the Sequence Input
        self.sequence_processor = self._build_conv_processor(
            num_tokens, conv_channels, conv_kernel_size, leaky_relu_slope
        )
        
        # The input to the main MLP is the concatenation of g_ and the processed sequence
        self.combined_input_dim = subgoal_dim + conv_channels[-1]

        # 2. Input Projection Layer
        self.input_layer = nn.Sequential(
            nn.Linear(self.combined_input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.LeakyReLU(leaky_relu_slope, inplace=True)
        )

        # 3. Stack of Residual Blocks
        self.residual_stack = nn.Sequential(
            *[ResidualBlockV2(hidden_dim, dropout_rate, leaky_relu_slope) for _ in range(num_blocks)]
        )

        # 4. Output Layer
        self.output_layer = nn.Linear(hidden_dim, num_gammas)

        self._initialize_weights(leaky_relu_slope)

    def _build_conv_processor(self, num_tokens, channels, kernel_size, leaky_slope):
        """Helper to create the 1D convolutional feature extractor."""
        layers = []
        # Conv1d expects (Batch, Channels, Length). For us, `num_tokens` is the channel dim
        # and `token_dim` is the length. We must permute the input tensor in `forward`.
        in_channels = num_tokens
        
        for out_channels in channels:
            layers.append(nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, stride=1, padding='same', bias=False),
                nn.BatchNorm1d(out_channels),
                nn.LeakyReLU(leaky_slope, inplace=True),
                nn.MaxPool1d(kernel_size=2, stride=2) # Downsample the length (token_dim)
            ))
            in_channels = out_channels
        
        layers.append(nn.AdaptiveAvgPool1d(1)) # Pool over the length dimension
        layers.append(nn.Flatten())
        return nn.Sequential(*layers)

    def _initialize_weights(self, leaky_relu_slope: float):
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                nn.init.kaiming_uniform_(module.weight, a=leaky_relu_slope, nonlinearity='leaky_relu')
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, g_: torch.Tensor, sequence_input: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        Args:
            g_ (torch.Tensor): The subgoal state component. Shape (B, subgoal_dim).
            sequence_input (torch.Tensor): The sequential state component. Shape (B, num_tokens, token_dim).

        Returns:
            torch.Tensor: Value estimates for all gammas, shape (B, num_gammas).
        """
        # 1. Process the sequence input
        # Conv1d expects (Batch, Channels, Length). We treat tokens as channels.
        # So, we permute (B, num_tokens, token_dim) -> (B, num_tokens, token_dim)
        processed_sequence = self.sequence_processor(sequence_input) # -> (B, conv_channels[-1])
        
        # 2. Combine the state components
        state_combined = torch.cat([g_, processed_sequence], dim=-1)
        
        # 3. Pass through the main ResNet MLP
        h = self.input_layer(state_combined)
        h = self.residual_stack(h)
        value_estimates = self.output_layer(h)
        
        return value_estimates