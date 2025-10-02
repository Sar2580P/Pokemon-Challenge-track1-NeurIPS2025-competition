from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple
from amago.nets.traj_encoders import TrajEncoder
import torch
from torch import nn
import gin
from enum import Enum
from typing import List, Dict,  Type
from amago.nets import ff, transformer
from amago.utils import amago_warning
from dataclasses import dataclass
from custom.hrm_utils.layers import (Attention, SwiGLU, CosSin, rms_norm, 
                                     CrossAttBlock, RotaryEmbedding, CastedLinear,
                                     ChunkRepresentativeAttention)
import math
from custom.hrm_utils.common import trunc_normal_init_
from amago.loading import Batch, MAGIC_PAD_VAL
import numpy as np 

class ModelType(str, Enum):
    REASONING="latent reasoning model"
    INPUT="input model for encoding text"
    OUTPUT="the output model to make predictions"
    CRITIC="the model to evaluate the quality of action using zH"
    MIXER="the model to combine zH, x in sophisticated manner to use it while updating the zL"
    
@gin.configurable
class HierarchicalReasoningModel_ACTV1Block(nn.Module):
    def __init__(self, 
                hidden_size: int=128, 
                num_heads: int=4, 
                expansion: int=2, 
                rms_norm_eps: float=1e-5
                ) -> None:
        super().__init__()

        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            expansion=expansion,
        )
        self.norm_eps = rms_norm_eps

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Post Norm
        # Self Attention
        atten_output=self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = rms_norm(hidden_states + atten_output, variance_epsilon=self.norm_eps)
        # Fully Connected
        hidden_states = rms_norm(hidden_states + self.mlp(hidden_states), variance_epsilon=self.norm_eps)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block]):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)

    @torch.compile()
    def forward(self, hidden_states: torch.Tensor, input_injection: torch.Tensor, **kwargs) -> torch.Tensor:
        # Input injection (add)
        hidden_states = hidden_states + input_injection
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, **kwargs)
        return hidden_states

@gin.configurable
class Mixer(nn.Module):
    def __init__(self, num_layers,
                 dim_zH, dim_x, n_heads=4, 
                 ffn_expansion=2, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([
            CrossAttBlock(dim_zH=dim_zH,
                 dim_x=dim_x,
                 num_heads=n_heads,
                 ffn_expansion=ffn_expansion,
                 dropout=dropout)
            for _ in range(num_layers)
        ])
        
    @torch.compile()
    def forward(self, zH, x, cos_sin=CosSin):
        for layer in self.layers: zH = layer(zH=zH, x=x, cos_sin=cos_sin)
        return zH

@gin.configurable
class SinusoidalTimeEmbedder(nn.Module):
    """
    A non-learnable time embedder using sinusoidal positional encoding.

    Args:
        embedding_dim (int): The desired dimension of the output time vector.
        max_timesteps (int): The maximum number of timesteps to pre-compute.
    """
    def __init__(self, embedding_dim: int, max_timesteps: int = 5000):
        super().__init__()
        self.embedding_dim = embedding_dim
        pe = torch.zeros(max_timesteps, embedding_dim)
        position = torch.arange(0, max_timesteps, dtype=torch.float).unsqueeze(1)        
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, time_idxs: torch.Tensor) -> torch.Tensor:
        return self.pe[time_idxs]
    
@dataclass
class State:
    zH: torch.Tensor
    zL: torch.Tensor   

    @classmethod
    def create(cls, zH_shape, zL_shape, device, dtype=torch.float32):      
        zH = trunc_normal_init_(torch.empty(zH_shape, dtype=dtype, device=device), std=1)
        zL = trunc_normal_init_(torch.empty(zL_shape, dtype=dtype, device=device), std=1)
        return cls(zH=zH, zL=zL)
    
    def reset(self, dones: np.ndarray):
        """
        Resets the states for the given indices.

        Args:
            dones (np.ndarray): A NumPy array of integer indices corresponding
                                to the batch items that need to be reset.
        """
        if dones.size == 0: return
        num_to_reset = len(dones)
        device, dtype = self.zH.device, self.zH.dtype

        new_zH_shape = (num_to_reset,) + self.zH.shape[1:]
        new_zL_shape = (num_to_reset,) + self.zL.shape[1:]

        new_zH = trunc_normal_init_(torch.empty(new_zH_shape, dtype=dtype, device=device), std=1)
        new_zL = trunc_normal_init_(torch.empty(new_zL_shape, dtype=dtype, device=device), std=1)
        self.zH[dones] = new_zH
        self.zL[dones] = new_zL

@gin.configurable
class HRMTrajEncoder(TrajEncoder):
    def __init__(
        self,
        tstep_dim: int,
        max_seq_len: int,
        H_block_type: Type[HierarchicalReasoningModel_ACTV1Block], 
        L_block_type: Type[HierarchicalReasoningModel_ACTV1Block], 
        M_block_type: Type[Mixer], 
        pos_encoding_type: Type[RotaryEmbedding],
        chunk_atten_type: Type[ChunkRepresentativeAttention] ,
        d_model: int=256,
        H_cycles: int=4, 
        L_cycles: int=4, 
        reasoning_tokens: int=4,
        H_layers: int=5, 
        L_layers: int=5,  
        chunk_len:int=20, 
        
    ):
        super().__init__(tstep_dim, max_seq_len)
        self.device="cuda" if torch.cuda.is_available() else "cpu"
        self.H_cycles, self.L_cycles=H_cycles, L_cycles
        self.d_model, self.tstep_dim = d_model, tstep_dim
        self.reasoning_tokens=reasoning_tokens
        self.chunk_len=chunk_len

        self.chunk_attn= chunk_atten_type(reasoning_tokens=self.reasoning_tokens, 
                                                      in_dim=self.d_model)
        self.H_model = self.get_model(
            model_type=ModelType.REASONING,
            use_compile=True,
            layers=H_layers,
            instance=H_block_type()
        ).to(self.device)
        self.L_model = self.get_model(
            model_type=ModelType.REASONING,
            use_compile=True,
            layers=L_layers,
            instance=L_block_type()
        ).to(self.device)
        self.M_model = self.get_model(
            model_type=ModelType.MIXER,
            use_compile=True,
            instance=M_block_type()
        ).to(self.device)
        self.time_embedder=SinusoidalTimeEmbedder(embedding_dim=d_model).to(self.device)
        self.rotary_emb = pos_encoding_type(device=self.device)
        
        if self.tstep_dim!=self.d_model:
            self.inp = nn.Sequential(
                        CastedLinear(in_features=tstep_dim, out_features=d_model),
                        nn.PReLU()
                    ).to(self.device)

    def get_model(self, model_type, **kwargs: Any):
        use_compile = kwargs.pop('use_compile', False)
        match model_type:
            case ModelType.REASONING:
                model = HierarchicalReasoningModel_ACTV1ReasoningModule(
                    layers=[kwargs['instance'] for _ in range(kwargs['layers'])]
                )
            case ModelType.MIXER: model = kwargs['instance']
            case _: raise ValueError('Invalid model type provided for HRM')

        # if use_compile and model_type in [ModelType.REASONING, ModelType.MIXER]:
        #     # Using max-autotune for best runtime performance
        #     compilation_mode = "max-autotune"
        #     print(f"⚡ Compiling {model.__class__.__name__} with {compilation_mode}...")
        #     return torch.compile(model, mode=compilation_mode)
        
        
        return model           

    def init_hidden_state(self, batch_size: int, device: torch.device) -> Optional[Any]:
        size= (batch_size, self.reasoning_tokens, self.d_model)
        return State.create(zH_shape=size, zL_shape=size, device=device)
    
    # @torch.compile(disable=True)
    def inner_forward(self, seq_chunk:torch.Tensor, time_idxs: torch.Tensor, 
                      hidden_state:State, intermediate_reasoning: bool=False):
        B,L, _ , D =seq_chunk.shape   # shape : [Batch, L_gamestep, L_feat, dim]
        assert D== self.d_model
        dtype, device= seq_chunk.dtype, seq_chunk.device
        if hidden_state is None:
            hidden_state=self.init_hidden_state(batch_size=B, device=device)
        zL, zH=hidden_state.zL.to(dtype=dtype), hidden_state.zH.to(dtype=dtype)
        time_embeddings = self.time_embedder(time_idxs).unsqueeze(2)
        if intermediate_reasoning:
            temporal_reasoning=torch.zeros(size=(B, L, self.reasoning_tokens, D), 
                                           device=seq_chunk.device, dtype=seq_chunk.dtype)
        else: 
            temporal_reasoning=torch.zeros(size=(B, L, D), 
                                           device=seq_chunk.device, dtype=seq_chunk.dtype)
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        # with torch._dynamo.disable():
        for t in range(L):   # at inference L=1 ... but during training L=# total gamesteps
            x, time_step_embedding=seq_chunk[:, t, :, :], time_embeddings[:, t,:, :]

            x = torch.cat([x, time_step_embedding], dim=1)
            with torch.no_grad():
                for _H_step in range(self.H_cycles):
                    zH_ = self.M_model(zH=zH, x=x, **seq_info)
                    for _L_step in range(self.L_cycles): 
                        if not ((_H_step == self.H_cycles - 1) and (_L_step == self.L_cycles - 1)):
                            zL=self.L_model(hidden_states=zL, input_injection=zH_, **seq_info)
                    if not (_H_step == self.H_cycles - 1):
                        zH=self.H_model(hidden_states=zH, input_injection=zL, **seq_info)
            # 1-step grad
            zH_ = self.M_model(zH=zH, x=x, **seq_info)
            zL = self.L_model(hidden_states=zL, input_injection=zH_, **seq_info)
            zH = self.H_model(hidden_states=zH, input_injection=zL, **seq_info)
            if intermediate_reasoning: temporal_reasoning[:, t, :, :] = zH 
            else: temporal_reasoning[:, t, :] = torch.sum(zH, dim=-2)  
        return temporal_reasoning, hidden_state
    
    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: State,
        log_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[transformer.TformerHiddenState]]:
        assert time_idxs is not None
        B,L, Z , D =seq.shape   # shape : [Batch, L_gamestep, L_feat, dim]
        if hidden_state is not None:
            # inference phase, assume L=1
            return self.inner_forward(
                seq_chunk=seq , time_idxs=time_idxs, hidden_state=hidden_state)

        extra_pad_len = self.chunk_len - (L%self.chunk_len)  if L%self.chunk_len>0 else 0
        if extra_pad_len>0 :
            if not hasattr(self, "pad_embedding"):
                pad_embedding = torch.mean(seq[:, -1, :, :], dim=0).unsqueeze(0)
                self.pad_embedding=pad_embedding.expand(B, -1, -1).unsqueeze(1)
            
            pad_embedding=self.pad_embedding.expand(-1, extra_pad_len, -1, -1)
            seq=torch.cat([seq, pad_embedding], dim=1)
            t= torch.Tensor([MAGIC_PAD_VAL]*extra_pad_len).unsqueeze(0)
            t=t.expand(B, -1).unsqueeze(2)
            time_idxs=torch.cat([time_idxs, t.to(device=time_idxs.device)], dim=1).long()

        if hasattr(self, "inp"): seq= self.inp(seq)

        # STEP-1: chunk the game steps...  
        D=self.d_model
        num_chunks = (L+ extra_pad_len)//self.chunk_len
        seq_chunked = seq.view(B, num_chunks, self.chunk_len, Z, D)
        seq_chunked = seq_chunked.reshape(-1, self.chunk_len, Z, D)
        
        time_idxs_chunked = time_idxs.view(B, num_chunks, self.chunk_len)
        time_idxs_chunked = time_idxs_chunked.reshape(-1, self.chunk_len)
        
        if hidden_state is not None:
            raise NotImplementedError("Need to implement logic for inference")

        chunked_reasoning_local, _ = self.inner_forward(seq_chunk=seq_chunked, time_idxs=time_idxs_chunked,
                                                               hidden_state=hidden_state, intermediate_reasoning=True)
        
        # STEP-2: Get the representative zH from each chunk
        representatives = chunked_reasoning_local[:, -1, :, :]    # shape: [B*num_chunks, chunk_len, reasoning_tokens, D]
        representatives = representatives.view(B, num_chunks, self.reasoning_tokens, self.d_model) 
        
        # STEP-3: Apply causal self-attention to these representatives
        temporal_summary = self.chunk_attn(representatives)
        
        summary = temporal_summary.view(-1, 1, self.reasoning_tokens, self.d_model)    
        summary = summary.expand(-1, self.chunk_len, -1, -1)   
        
        # STEP-4: Use this final_summary as new input-seq
        seq_chunked = torch.cat([seq_chunked, summary], dim=-2)
        final_reasoning , _ = self.inner_forward(seq_chunk=seq_chunked, time_idxs=time_idxs_chunked, 
                                            hidden_state=hidden_state)
        
        # STEP-5: Merge chunks, aggregate over reasoning dimension...
        final_reasoning = final_reasoning.view(B, -1, self.chunk_len , self.d_model)
        final_reasoning = final_reasoning.reshape(B, -1, self.d_model)
        
        # STEP-6: Removing the pad embedding created...
        final_reasoning=final_reasoning[:, :-extra_pad_len, :] if extra_pad_len>0 else final_reasoning
        return final_reasoning , hidden_state
        
    @property
    def emb_dim(self) -> int:
        return self.d_model
    

    def reset_hidden_state(
        self, hidden_state: Optional[State], dones: np.ndarray
    ) -> Optional[State]:
        if hidden_state is None: return None
        assert isinstance(hidden_state, State)
        hidden_state.reset(idxs=dones)
        return hidden_state