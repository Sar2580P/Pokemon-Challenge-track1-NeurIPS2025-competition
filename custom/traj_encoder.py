from abc import ABC, abstractmethod
from typing import Optional, Any, Tuple
from amago.nets.traj_encoders import TrajEncoder
import torch
import torch.nn.functional as F
from torch import nn
import gin
from enum import Enum
import einops
from typing import List, Dict,  Type
from amago.nets import transformer
from dataclasses import dataclass
from einops import repeat, rearrange, reduce
from custom.hrm_utils.layers import (Attention, SwiGLU, CosSin,  init_scaled_linear, 
                                     CrossAttBlock, RotaryEmbedding, CastedLinear,
                                     ChunkRepresentativeAttention, LayerScale)
import math, os
from custom.hrm_utils.common import trunc_normal_init_
from amago.loading import MAGIC_PAD_VAL
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
                num_layers : int=1,
                ) -> None:
        super().__init__()
        weight_init_scale= 1./math.sqrt(2*num_layers)
        self.self_attn = Attention(
            hidden_size=hidden_size,
            head_dim=hidden_size // num_heads,
            num_heads=num_heads,
            num_key_value_heads=num_heads,
            causal=False, 
            weight_init_scale=weight_init_scale,
        )
        self.mlp = SwiGLU(
            hidden_size=hidden_size,
            expansion=expansion,
        )
        init_scaled_linear(self.mlp.down_proj, weight_init_scale)
        self.pre_mlp_norm= nn.LayerNorm(hidden_size)
        self.gamma_1 = LayerScale(hidden_size)
        self.gamma_2 = LayerScale(hidden_size)

    def forward(self, cos_sin: CosSin, hidden_states: torch.Tensor) -> torch.Tensor:
        # Self Attention
        atten_output=self.self_attn(cos_sin=cos_sin, hidden_states=hidden_states)
        hidden_states = hidden_states + self.gamma_1(atten_output)
        
        # 2. FFN Sub-layer (Pre-Norm)
        residual = hidden_states # The intermediate residual path
        hidden_states_norm = self.pre_mlp_norm(hidden_states)
        mlp_output = self.mlp(hidden_states_norm)
        
        hidden_states = residual + self.gamma_2(mlp_output)
        return hidden_states


class HierarchicalReasoningModel_ACTV1ReasoningModule(nn.Module):
    def __init__(self, layers: List[HierarchicalReasoningModel_ACTV1Block], 
                 hidden_size):
        super().__init__()

        self.layers = torch.nn.ModuleList(layers)
        self.gamma_1 = LayerScale(hidden_size, init_values=5e-4)

    @torch.compile()
    def forward(self, hidden_states: torch.Tensor, 
                input_injection: torch.Tensor, cos_sin=CosSin) -> torch.Tensor:
        hidden_states = hidden_states + self.gamma_1(input_injection)
        for layer in self.layers:
            hidden_states = layer(hidden_states=hidden_states, cos_sin=cos_sin)
        return hidden_states

@gin.configurable
class Mixer(nn.Module):
    def __init__(self, num_layers,
                 dim_zH, dim_x, n_heads=4, 
                 ffn_expansion=2, dropout=0.1):
        super().__init__()
        weight_init_scale=1./math.sqrt(2*num_layers)
        self.n_heads=n_heads
        self.layers = nn.ModuleList([
            CrossAttBlock(dim_zH=dim_zH,
                 dim_x=dim_x,
                 num_heads=n_heads,
                 ffn_expansion=ffn_expansion,
                 dropout=dropout, 
                 weight_init_scale=weight_init_scale)
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
    temporal_buffer: torch.Tensor = None  # only for inference-mode
    curr_inference_step: torch.Tensor = None   # only for inference-mode
    chunk_counter:torch.Tensor=None
    @classmethod
    def create(cls, zH_shape, zL_shape, device, dtype=torch.float32, for_inference:bool=True):      
        zH = trunc_normal_init_(torch.empty(zH_shape, dtype=dtype, device=device), std=1)
        zL = trunc_normal_init_(torch.empty(zL_shape, dtype=dtype, device=device), std=1)
        
        if for_inference:
            batch_size= zH_shape[0]
            reasoning_tokens=zH_shape[1]
            d_model=zH_shape[2]
            total_buffer_len=50
            temporal_buffer=torch.zeros(size=(batch_size, total_buffer_len,  reasoning_tokens, d_model), 
                                       device=device, dtype=dtype)
            curr_inference_step=torch.zeros(size=(batch_size,), device=device, dtype=torch.int8)
            chunk_counter=torch.zeros(size=(batch_size,), device=device, dtype=torch.int8)
            return cls(zH=zH, zL=zL, 
                       temporal_buffer=temporal_buffer, 
                       curr_inference_step=curr_inference_step, 
                       chunk_counter=chunk_counter)
        return cls(zH=zH, zL=zL)
    
    def reset(self, dones: np.ndarray):
        """
        Resets the states for the given indices.

        Args:
            dones (np.ndarray): A NumPy array of integer indices corresponding
                                to the batch items that need to be reset.
        """
        num_to_reset = dones.sum()
        if num_to_reset==0: return 
        device, dtype = self.zH.device, self.zH.dtype

        # Now num_to_reset correctly matches the number of items that will be selected.
        new_zH_shape = (num_to_reset,) + self.zH.shape[1:]
        new_zL_shape = (num_to_reset,) + self.zL.shape[1:]

        new_zH = trunc_normal_init_(torch.empty(new_zH_shape, dtype=dtype, device=device), std=1)
        new_zL = trunc_normal_init_(torch.empty(new_zL_shape, dtype=dtype, device=device), std=1)
        
        self.zH[dones] = new_zH
        self.zL[dones] = new_zL      
        

        # reset the temporal buffer to zeros for these indices
        self.temporal_buffer[dones] = torch.zeros_like(self.temporal_buffer[dones]) 
        self.curr_inference_step[dones] = 0 
        self.chunk_counter[dones]=0

class ResidualBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int = 3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, kernel_size, padding=kernel_size // 2, bias=False),
            nn.BatchNorm1d(channels)
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.activation(x + self.block(x))

@gin.configurable()
class VAE_Prior(nn.Module):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        latent_dim: int,
        kernel_size:int,
        channel_dims: List[int] = [128, 256, 512], 
        # --- Self-contained training arguments ---
        learning_rate: float = 1e-4,
        gradient_accumulation_steps: int = 1,
        
        # === KL Annealing Parameters ===
        kl_anneal_start_step: int = 1000, # Start annealing after 1000 steps
        kl_anneal_end_step: int = 5000,   # Beta reaches 1.0 at step 5000
        kl_anneal_max_beta: float = 1.0,
        # ==============================
    ):
        super().__init__()
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.latent_dim= latent_dim

        # === 📉 ENCODER with Residuals ===
        encoder_layers = []
        in_c = num_tokens
        
        # Initial convolution to get to the first channel dimension
        encoder_layers.append(nn.Sequential(
            nn.Conv1d(in_c, channel_dims[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.GroupNorm(1, channel_dims[0]),
            nn.PReLU()
        ))
        in_c = channel_dims[0]

        # Downsampling pyramid with residual blocks
        for out_c in channel_dims[1:]:
            encoder_layers.append(ResidualBlock(in_c, kernel_size)) # Add residual block
            encoder_layers.append(nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(1, out_c),
                nn.PReLU()
            ))
            in_c = out_c
            
        encoder_layers.append(ResidualBlock(in_c, kernel_size)) # Final residual block
        encoder_layers.append(nn.AdaptiveAvgPool1d(1))
        self.encoder = nn.Sequential(*encoder_layers)

        # === LATENT SPACE (Unchanged) ===
        self.fc_mu = nn.Linear(channel_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(channel_dims[-1], latent_dim)
        self.fc_decode = nn.Linear(latent_dim, channel_dims[-1])
        self.final_conv_channel = channel_dims[-1]

        # === 📈 DECODER with Residuals ===
        decoder_layers = []
        in_c = channel_dims[-1]
        
        # Upsampling pyramid with residual blocks
        reversed_channels = channel_dims[::-1]
        for i in range(len(reversed_channels) - 1):
            out_c = reversed_channels[i+1]
            decoder_layers.append(ResidualBlock(in_c, kernel_size)) # Add residual block
            decoder_layers.append(nn.Sequential(
                nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.GroupNorm(1, out_c),
                nn.PReLU()
            ))
            in_c = out_c
        
        # Final block and output convolution
        decoder_layers.append(ResidualBlock(in_c, kernel_size)) # Add residual block
        decoder_layers.append(
             nn.ConvTranspose1d(channel_dims[0], num_tokens, kernel_size=4, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(*decoder_layers)

        # --- Training and state management ---
        self.target_beta = kl_anneal_max_beta # Renamed for clarity
        self.kl_anneal_start_step = kl_anneal_start_step
        self.kl_anneal_end_step = kl_anneal_end_step
        self.gradient_accumulation_steps = gradient_accumulation_steps
        self.global_step = 0
        self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)

    @torch.compile()
    def encode(self, x):
        h = self.encoder(x) # -> (B, C_last, 1)
        h = h.squeeze(-1)   # -> (B, C_last)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    @torch.compile()
    def decode(self, z):
        h = self.fc_decode(z)       # -> (B, C_last)
        h = h.unsqueeze(-1)         # -> (B, C_last, 1)
        x_recon_pre = self.decoder(h)

        # Ensure output size is exactly token_dim
        x_recon = F.interpolate(x_recon_pre, size=self.token_dim, mode='linear', align_corners=False)
        return x_recon

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

    def get_current_beta(self):
        """Calculates the current beta value based on the global step and annealing schedule."""
        if self.global_step < self.kl_anneal_start_step:
            current_beta = 0.0
        elif self.global_step >= self.kl_anneal_end_step:
            current_beta = self.target_beta
        else:
            # Linear ramp-up
            step_progress = max(1, (self.global_step - self.kl_anneal_start_step) / \
                            (self.kl_anneal_end_step - self.kl_anneal_start_step))
            current_beta = self.target_beta * step_progress
        return current_beta

    def compute_loss(self, reconstructed_o, o_, mu, logvar, vae_mask):
        B = vae_mask.shape[0]
        current_beta = self.get_current_beta()
        recon_loss_unmasked = F.mse_loss(reconstructed_o.reshape(B, -1), o_.reshape(B, -1), reduction="none")
        recon_loss = (recon_loss_unmasked * vae_mask.view(B, -1)).sum() / vae_mask.sum()
        kl_div = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        total_loss = recon_loss + current_beta * kl_div
        return {"total": total_loss, "recon": recon_loss, "kl": kl_div}

    def train_step(self, o_, vae_mask, save_every_n_steps: int, save_dir:str):
        reconstructed_o, mu, logvar = self.forward(o_)
        loss_dict = self.compute_loss(reconstructed_o, o_, mu, logvar, vae_mask)
        total_loss = loss_dict["total"]
        scaled_loss = total_loss / self.gradient_accumulation_steps
        scaled_loss.backward()
        if (self.global_step + 1) % self.gradient_accumulation_steps == 0:
            self.optimizer.step()
            self.optimizer.zero_grad()
        if (self.global_step + 1) % save_every_n_steps == 0:
            self.save_model(step=self.global_step + 1, save_dir=save_dir)
        self.global_step += 1
        return {k: v.detach().item() for k, v in loss_dict.items()}

    def save_model(self, save_dir: str, step: int = None):
        filename = f"vae_prior_step_{step}.pt"
        save_location = os.path.join(save_dir, filename)
        torch.save(self.state_dict(), save_location)
        print(f"✅ VAE model saved to {save_location}")

    def load_model(self, file_path: str, device: str = 'cpu'):
        self.load_state_dict(torch.load(file_path, map_location=device))
        print(f"✅ VAE model loaded from {file_path}")


@gin.configurable()
class CVAE(VAE_Prior):
    def __init__(
        self,
        num_tokens: int,
        token_dim: int,
        latent_dim: int,
        channel_dims: List[int] = [512, 256],
        kernel_size: int = 3,
    ):  
        super().__init__(
            num_tokens=num_tokens,
            token_dim=token_dim,
            latent_dim=latent_dim,
            channel_dims=channel_dims,
            kernel_size=kernel_size
        )
        self.num_tokens = num_tokens
        self.token_dim = token_dim
        self.latent_dim= latent_dim

        # === 📉 ENCODER with Residuals ===
        encoder_layers = []
        in_c = num_tokens
        
        # Initial convolution to get to the first channel dimension
        encoder_layers.append(nn.Sequential(
            nn.Conv1d(in_c, channel_dims[0], kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm1d(channel_dims[0]),
            nn.ReLU(inplace=True)
        ))
        in_c = channel_dims[0]

        # Downsampling pyramid with residual blocks
        for out_c in channel_dims[1:]:
            encoder_layers.append(ResidualBlock(in_c, kernel_size)) # Add residual block
            encoder_layers.append(nn.Sequential(
                nn.Conv1d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True)
            ))
            in_c = out_c
            
        encoder_layers.append(ResidualBlock(in_c, kernel_size)) # Final residual block
        encoder_layers.append(nn.AdaptiveAvgPool1d(1))
        self.encoder = nn.Sequential(*encoder_layers)

        # === LATENT SPACE (Unchanged) ===
        self.fc_mu = nn.Linear(channel_dims[-1], latent_dim)
        self.fc_logvar = nn.Linear(channel_dims[-1], latent_dim)
        self.fc_decode = nn.Linear(latent_dim, channel_dims[-1])
        self.final_conv_channel = channel_dims[-1]

        # === 📈 DECODER with Residuals ===
        decoder_layers = []
        in_c = channel_dims[-1]
        
        # Upsampling pyramid with residual blocks
        reversed_channels = channel_dims[::-1]
        for i in range(len(reversed_channels) - 1):
            out_c = reversed_channels[i+1]
            decoder_layers.append(ResidualBlock(in_c, kernel_size)) # Add residual block
            decoder_layers.append(nn.Sequential(
                nn.ConvTranspose1d(in_c, out_c, kernel_size=4, stride=2, padding=1, bias=False),
                nn.BatchNorm1d(out_c),
                nn.ReLU(inplace=True)
            ))
            in_c = out_c
        
        # Final block and output convolution
        decoder_layers.append(ResidualBlock(in_c, kernel_size)) # Add residual block
        decoder_layers.append(
             nn.ConvTranspose1d(channel_dims[0], num_tokens, kernel_size=4, stride=2, padding=1)
        )
        self.decoder = nn.Sequential(*decoder_layers)

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
        CVAE_type: Type[CVAE]=None, 
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
        
        ####################################
        ## BTS => before temporal summary ##
        ## ATS => after temporal summary  ##
        ####################################
        
        self.M_model_BTS = self.get_model(
            model_type=ModelType.MIXER,
            use_compile=True,
            instance=M_block_type()
        ).to(self.device)
        self.H_model_BTS = self.get_model(
            model_type=ModelType.REASONING,
            use_compile=True,
            layers=H_layers,
            instance=H_block_type(),
            
            # information injector params-->
            hidden_size=d_model
        ).to(self.device)
        self.L_model_BTS = self.get_model(
            model_type=ModelType.REASONING,
            use_compile=True,
            layers=L_layers,
            instance=L_block_type(),
            
            # information injector params-->
            hidden_size=d_model
        ).to(self.device)

        self.M_model_ATS = self.get_model(
            model_type=ModelType.MIXER,
            use_compile=True,
            instance=M_block_type()
        ).to(self.device)
        self.H_model_ATS = self.get_model(
            model_type=ModelType.REASONING,
            use_compile=True,
            layers=H_layers,
            instance=H_block_type(),
            
            # information injector params-->
            hidden_size=d_model
        ).to(self.device)
        self.L_model_ATS = self.get_model(
            model_type=ModelType.REASONING,
            use_compile=True,
            layers=L_layers,
            instance=L_block_type(),
            
            # information injector params-->
            hidden_size=d_model
        ).to(self.device)
        
        
        self.time_embedder=SinusoidalTimeEmbedder(embedding_dim=d_model).to(self.device)
        self.rotary_emb = pos_encoding_type(device=self.device)
        
        if CVAE_type is not None: 
            self.cvae=CVAE_type()
            dim=self.cvae.latent_dim
            self.proj = CastedLinear(in_features=dim , out_features=self.d_model)   # TODO: look for right shapes        
        else: self.cvae=None
                
        # normalization layers
        self.inp_norm = nn.LayerNorm(self.d_model).to(self.device)
        self.temporal_summary_norm= nn.LayerNorm(self.d_model).to(self.device)
        
        if self.tstep_dim!=self.d_model:
            self.inp = nn.Sequential(
                        CastedLinear(in_features=tstep_dim, out_features=d_model),
                        nn.PReLU(),
                    ).to(self.device)

    def get_model(self, model_type, **kwargs: Any):
        match model_type:
            case ModelType.REASONING:
                model = HierarchicalReasoningModel_ACTV1ReasoningModule(
                    layers=[kwargs['instance'] for _ in range(kwargs['layers'])], 
                    hidden_size=kwargs['hidden_size']
                )
            case ModelType.MIXER: model = kwargs['instance']
            case _: raise ValueError('Invalid model type provided for HRM')
        return model           

    def init_hidden_state(self, batch_size: int, device: torch.device, for_inference:bool=True) -> Optional[Any]:
        size= (batch_size, self.reasoning_tokens, self.d_model)
        return State.create(zH_shape=size, zL_shape=size, device=device, for_inference=for_inference)
    
    def inner_forward_temporal(self, seq_chunk:torch.Tensor, time_idxs: torch.Tensor, 
                                hidden_state:State=None):
        B,L, _ , D =seq_chunk.shape   # shape : [Batch, L_gamestep, L_feat, dim]
        # assert D== self.d_model
        dtype, device= seq_chunk.dtype, seq_chunk.device
        
        if hidden_state is None:
            hidden_state=self.init_hidden_state(batch_size=B, device=device, for_inference=False)
        zL, zH=hidden_state.zL.to(dtype=dtype), hidden_state.zH.to(dtype=dtype)
        time_embeddings = self.time_embedder(time_idxs).unsqueeze(2)
        temporal_reasoning=torch.zeros(size=(B, L, self.reasoning_tokens, D), 
                                        device=seq_chunk.device, dtype=seq_chunk.dtype)
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        
        m_model, h_model, l_model =self.M_model_BTS, self.H_model_BTS ,self.L_model_BTS
        # with torch._dynamo.disable():
        for t in range(L):   # at inference L=1 ... but during training L=# total gamesteps
            x, time_step_embedding=seq_chunk[:, t, :, :], time_embeddings[:, t,:, :]
            x = torch.cat([x, time_step_embedding], dim=1)
            with torch.no_grad():
                for _H_step in range(self.H_cycles):
                    zH_ = m_model(zH=zH, x=x, **seq_info)
                    for _L_step in range(self.L_cycles): 
                        if not ((_H_step == self.H_cycles - 1) and (_L_step == self.L_cycles - 1)):
                            zL=l_model(hidden_states=zL, input_injection=zH_, **seq_info)
                    if not (_H_step == self.H_cycles - 1):
                        zH=h_model(hidden_states=zH, input_injection=zL, **seq_info)
            
            # 1-step gradient update
            zH_ = m_model(zH=zH, x=x, **seq_info)
            zL = l_model(hidden_states=zL, input_injection=zH_, **seq_info)
            zH = h_model(hidden_states=zH, input_injection=zL, **seq_info)

            temporal_reasoning[:, t, :, :] = zH 
            zH, zL = zH.detach(), zL.detach()
        return temporal_reasoning, hidden_state
    
    

    def inner_forward(self, seq_chunk:torch.Tensor, time_idxs: torch.Tensor, 
                      hidden_state:State=None):
        B,R, D =seq_chunk.shape   # shape : [Batch, L_gamestep, L_feat, dim]
        assert D== self.d_model
        dtype, device= seq_chunk.dtype, seq_chunk.device
        
        if hidden_state is None:
            hidden_state=self.init_hidden_state(batch_size=B, device=device, for_inference=False)
        zL, zH=hidden_state.zL.to(dtype=dtype), hidden_state.zH.to(dtype=dtype)
        time_embeddings = self.time_embedder(time_idxs).unsqueeze(1)
        seq_info = dict(
            cos_sin=self.rotary_emb() if hasattr(self, "rotary_emb") else None,
        )
        x = torch.cat([seq_chunk, time_embeddings], dim=1)
        
        m_model, h_model, l_model =self.M_model_ATS, self.H_model_ATS , self.L_model_ATS
        with torch.no_grad():
            for _H_step in range(self.H_cycles):
                zH_ = m_model(zH=zH, x=x, **seq_info)
                for _L_step in range(self.L_cycles): 
                    if not ((_H_step == self.H_cycles - 1) and (_L_step == self.L_cycles - 1)):
                        zL=l_model(hidden_states=zL, input_injection=zH_, **seq_info)
                if not (_H_step == self.H_cycles - 1):
                    zH=h_model(hidden_states=zH, input_injection=zL, **seq_info)
        
        # 1-step gradient update
        zH_ = m_model(zH=zH, x=x, **seq_info)
        zL = l_model(hidden_states=zL, input_injection=zH_, **seq_info)
        zH = h_model(hidden_states=zH, input_injection=zL, **seq_info)

   
        zH, zL = zH.detach(), zL.detach()
        return zH.mean(dim=1), hidden_state
    
    def forward(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: State,
        log_dict: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, Optional[transformer.TformerHiddenState]]:
        assert time_idxs is not None
        B,L, Z , D =seq.shape   # shape : [Batch, L_gamestep, L_feat, dim]
        if hasattr(self, "inp"): seq= self.inp(seq)
        
        seq= self.inp_norm(seq)
                
        if hidden_state is not None:   # INFERENCE
            
            '''
            STEP-1: Pass through 'inner_forward_temporal', with hidden state, if representative -> save in cache
            STEP-2: Representative Chunk Attn
                - get the cached representatives with padding (on right)
                - send it to chunk-atten module
                - use batch indexing to pick the past summary causally
            STEP-3: Pass through 'inner_forward' with all concatenated like : summary, seq, etc...  
            STEP-4: Store the current chunk representative (if it is) and update chunk_counter        
            '''
            
            time_idxs=time_idxs[:, :, -1]   # [batch, 1, 1] -> [batch, 1]
            representative_interval=5
            temporal_buffer= hidden_state.temporal_buffer   
            
            # STEP-1 
            chunked_reasoning_local, hidden_state = self.inner_forward_temporal(seq_chunk=seq, time_idxs=time_idxs,     # temporal_reasoning: [Batch, 1, reasoning_tokens, self.d_model]
                                                                                hidden_state=hidden_state)
            # STEP-2
            temporal_summary = self.chunk_attn(temporal_buffer)
            batch_indices = torch.arange(B, device=temporal_buffer.device, dtype=torch.long)
            recent_representative_idx= (hidden_state.chunk_counter-1).to(torch.long)
            summary = temporal_summary[batch_indices, recent_representative_idx, :, :].unsqueeze(1)    #expand dim for temporal dim

            # there would be some in batch, for which recent_representative_idx=-1
            invalid_mask = (hidden_state.chunk_counter == 0)
            if torch.any(invalid_mask):
                summary[invalid_mask] = 0.0
                        
            updated_reasoning_tokens = summary.shape[2]
            summary = self.temporal_summary_norm(summary)    # as the input seq is normed
            
            # STEP-3
            seq = torch.cat([seq, summary, chunked_reasoning_local], dim=-2)
            seq = einops.rearrange(seq, 'bn c r d -> (bn c) r d')
            time_idxs = einops.rearrange(time_idxs, 'b c -> (b c)')
            final_reasoning , _ = self.inner_forward(seq_chunk=seq, time_idxs=time_idxs)

            final_reasoning= einops.rearrange(final_reasoning, '(b c) d -> b c d', b=B)    # Merge chunks, aggregate over reasoning dimension...
            
            # STEP-4
            hidden_state.curr_inference_step+=1
            mask = hidden_state.curr_inference_step%representative_interval == 0
            if torch.any(mask):
                batch_indices, buffer_indices = torch.where(mask)[0].to(torch.long)  , hidden_state.chunk_counter[mask].to(torch.long)
                representative = chunked_reasoning_local[mask] 
                hidden_state.temporal_buffer[batch_indices, buffer_indices] = representative.squeeze()
                
                hidden_state.chunk_counter[mask] += 1
            return final_reasoning , hidden_state
               
            
        extra_pad_len = self.chunk_len - (L%self.chunk_len)  if L%self.chunk_len>0 else 0
        if extra_pad_len>0 :
            if not hasattr(self, "pad_embedding"):
                # as we are rejecting the temporal summary for very last chunk in full seq, we can simply initialize it as 0
                self.pad_embedding =  torch.zeros(size=(B, 1, Z, self.d_model)).to(seq.device)     #torch.mean(seq[:, -1, :, :], dim=0).unsqueeze(0).detach()
            
            pad_embedding=self.pad_embedding.expand(-1, extra_pad_len, -1, -1)
            seq=torch.cat([seq, pad_embedding], dim=1)
            t= torch.Tensor([MAGIC_PAD_VAL]*extra_pad_len).unsqueeze(0)
            t=t.expand(B, -1).unsqueeze(2)
            time_idxs=torch.cat([time_idxs, t.to(device=time_idxs.device)], dim=1).long()

        # STEP-1: chunk the game steps...  
        D=self.d_model
        num_chunks = (L+ extra_pad_len)//self.chunk_len
        seq_chunked = seq.view(B, num_chunks, self.chunk_len, Z, D)
        seq_chunked = seq_chunked.reshape(-1, self.chunk_len, Z, D)
        
        time_idxs_chunked = time_idxs.view(B, num_chunks, self.chunk_len)
        time_idxs_chunked = time_idxs_chunked.reshape(-1, self.chunk_len)
        chunked_reasoning_local, _ = self.inner_forward_temporal(seq_chunk=seq_chunked, time_idxs=time_idxs_chunked, 
                                                        hidden_state=hidden_state)
        
        # STEP-2: Get the representative zH from each chunk
        representatives = chunked_reasoning_local[:, -1, :, :]    # shape: [B*num_chunks, chunk_len, reasoning_tokens, D]
        representatives = representatives.view(B, num_chunks, self.reasoning_tokens, self.d_model) 
        
        # STEP-3: Apply causal self-attention to these representatives
        temporal_summary = self.chunk_attn(representatives)
        summary_for_prev_chunk = temporal_summary[:, :-1, :, :] # Get summaries for chunks 0 to N-1
        zero_summary = torch.zeros_like(summary_for_prev_chunk[:, :1, :, :]) # Create a placeholder for the first chunk
        shifted_summary = torch.cat([zero_summary, summary_for_prev_chunk], dim=1) # Shape: [B, num_chunks, ...]

        updated_reasoning_tokens = shifted_summary.shape[2]
        # STEP-4a: Broadcast and combine for the second, definitive pass
        summary = shifted_summary.view(-1, 1, updated_reasoning_tokens, self.d_model) # Reshape to [B*num_chunks, ...]
        summary = self.temporal_summary_norm(summary.expand(-1, self.chunk_len, -1, -1))    # as the input seq is normed


        # STEP-4b: Use this final_summary as new input-seq
        seq_chunked = torch.cat([seq_chunked, summary, chunked_reasoning_local], dim=-2)
        
        seq_chunked = einops.rearrange(seq_chunked, 'bn c r d -> (bn c) r d')
        time_idxs_chunked = einops.rearrange(time_idxs_chunked, 'bn c -> (bn c)')

        final_reasoning , _ = self.inner_forward(seq_chunk=seq_chunked, time_idxs=time_idxs_chunked)
        
        # STEP-5: Merge chunks, aggregate over reasoning dimension...
        final_reasoning= einops.rearrange(final_reasoning, '(b n c) d -> b (n c) d', b=B, n=num_chunks)
        
        # STEP-6: Removing the pad embedding created...
        if extra_pad_len>0:
            final_reasoning=final_reasoning[:, :-extra_pad_len, :] if extra_pad_len>0 \
                                                                    else final_reasoning
        return final_reasoning , hidden_state
    
    
    def forward_opponent_modeling(
        self,
        seq: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: State,
        curr_step:int=-1 , total_steps: int=-1 , 
        log_dict: Optional[dict] = None,
        vae_prior_latents:torch.Tensor=None, 
    ) -> Tuple[torch.Tensor, Optional[transformer.TformerHiddenState]]:
        
        assert time_idxs is not None
        B,L, Z , D =seq.shape   # shape : [Batch, L_gamestep, L_feat, dim]
        if hasattr(self, "inp"): seq= self.inp(seq)
        seq= self.inp_norm(seq)
        
        if hidden_state is not None:
            time_idxs=time_idxs[:, :, -1]   # [batch, 1, 1] -> [batch, 1]
            total_buffer_len=hidden_state.temporal_buffer.shape[1]
            representative_interval=5
                        
            temporal_reasoning, hidden_state = self.inner_forward(seq_chunk=seq, time_idxs=time_idxs,     # temporal_reasoning: [Batch, 1, reasoning_tokens, self.d_model]
                                                                hidden_state=hidden_state, 
                                                                intermediate_reasoning=True)
            
            # store the temporal reasoning at idx= curr_inference_step//representative_interval
            temporal_buffer= hidden_state.temporal_buffer    # [B, total_buffer_len, reasoning_tokens, self.d_model]
            # for a specific batch idx, store the temporal reasoning at
            # idx= curr_inference_step//representative_interval
            
            # Change dtype to torch.long for both index tensors
            batch_indices = torch.arange(B, device=temporal_buffer.device, dtype=torch.long)
            seq_indices = torch.clamp(
                hidden_state.curr_inference_step // representative_interval, 
                max=total_buffer_len - 1
            ).to(torch.long) # Use .to(torch.long) here

            # This line should now work
            temporal_buffer[batch_indices, seq_indices, :, :] = temporal_reasoning[:, -1, :, :].to(temporal_buffer.dtype)
            hidden_state.temporal_buffer= temporal_buffer
            hidden_state.curr_inference_step+=1
            
            # Do chunk attention on the entire buffer
            temporal_summary = self.chunk_attn(temporal_buffer)     # shape ==> [B, total_buffer_len, reasoning_tokens, self.d_model]
            summary = temporal_summary[batch_indices, seq_indices, :, :].unsqueeze(1)    # [B, 1, s, d]
            # STEP-4: Use this final_summary as new input-seq
            seq = torch.cat([seq, summary], dim=-2)
            # pass through CVAE
            assert self.cvae is not None, "Idiot!! doing opponent modeling inference without CVAE..."
            temporal_seq_flattened = rearrange( 
                seq, "b l s d -> (b l) s d" 
            )
            mu, logvar = self.cvae.encode(temporal_seq_flattened)
            opponent_latent= self.cvae.reparameterize(mu, logvar)
            opponent_feature = self.proj(opponent_latent).unsqueeze(1).unsqueeze(2)
            # merge with input --> 
            seq = torch.cat([seq, opponent_feature], dim=2) 
            temporal_reasoning, hidden_state = self.inner_forward(seq_chunk=seq, time_idxs=time_idxs, hidden_state=hidden_state)
            return torch.zeros_like(temporal_reasoning), hidden_state            
            
        dev= vae_prior_latents.device
        B_, L_ ,_, D_latent = vae_prior_latents.shape
        assert B==B_ and L==L_ and (time_idxs is not None)

        extra_pad_len = self.chunk_len - (L%self.chunk_len)  if L%self.chunk_len>0 else 0
        if extra_pad_len>0 :
            if not hasattr(self, "pad_embedding"):
                # as we are rejecting the temporal summary for very last chunk in full seq, we can simply initialize it as 0
                self.pad_embedding =  torch.zeros(size=(B, 1, Z, self.d_model)).to(seq.device)     #torch.mean(seq[:, -1, :, :], dim=0).unsqueeze(0).detach()
            
            pad_embedding=self.pad_embedding.expand(-1, extra_pad_len, -1, -1)
            seq=torch.cat([seq, pad_embedding], dim=1)
            t= torch.Tensor([MAGIC_PAD_VAL]*extra_pad_len).unsqueeze(0)
            t=t.expand(B, -1).unsqueeze(2)
            time_idxs=torch.cat([time_idxs, t.to(device=time_idxs.device)], dim=1).long()

            # do padding for vae_prior_latents (batch, seq-len, 1, latent-dim): 
            vae_prior_latents=torch.cat([vae_prior_latents, torch.zeros(size=(B, extra_pad_len, 1, D_latent)).to(dev)], dim=1)

        # STEP-1: chunk the game steps...  
        D=self.d_model
        num_chunks = (L+ extra_pad_len)//self.chunk_len
        seq_chunked = seq.view(B, num_chunks, self.chunk_len, Z, D)
        seq_chunked = seq_chunked.reshape(-1, self.chunk_len, Z, D)
        vae_prior_latents_chunked=vae_prior_latents.view(B, num_chunks, self.chunk_len, 1, D_latent)
        vae_prior_latents_chunked=vae_prior_latents_chunked.reshape(-1, self.chunk_len, 1, D_latent)
        
        time_idxs_chunked = time_idxs.view(B, num_chunks, self.chunk_len)
        time_idxs_chunked = time_idxs_chunked.reshape(-1, self.chunk_len)

        chunked_reasoning_local, _ = self.inner_forward_temporal(seq_chunk=seq_chunked, time_idxs=time_idxs_chunked,
                                                               hidden_state=hidden_state)
        
        # STEP-2: Get the representative zH from each chunk
        representatives = chunked_reasoning_local[:, -1, :, :]    # shape: [B*num_chunks, chunk_len, reasoning_tokens, D]
        representatives = representatives.view(B, num_chunks, self.reasoning_tokens, self.d_model) 
        
        # STEP-3: Apply causal self-attention to these representatives
        temporal_summary = self.chunk_attn(representatives)
        
        summary_for_prev_chunk = temporal_summary[:, :-1, :, :] # Get summaries for chunks 0 to N-1
        zero_summary = torch.zeros_like(summary_for_prev_chunk[:, :1, :, :]) # Create a placeholder for the first chunk
        shifted_summary = torch.cat([zero_summary, summary_for_prev_chunk], dim=1) # Shape: [B, num_chunks, ...]

        # STEP-4: Broadcast and combine for the second, definitive pass
        updated_reasoning_tokens = shifted_summary.shape[2]
        summary = shifted_summary.view(-1, 1, updated_reasoning_tokens, self.d_model) # Reshape to [B*num_chunks, ...]
        summary = self.temporal_summary_norm(summary.expand(-1, self.chunk_len, -1, -1))
        
        # summary = temporal_summary.view(-1, 1, self.reasoning_tokens, self.d_model)    
        # summary = summary.expand(-1, self.chunk_len, -1, -1)   

        # # STEP-4: Use this final_summary as new input-seq
        seq_chunked = torch.cat([seq_chunked, summary, chunked_reasoning_local], dim=-2)
 
        # STEP-5: Get the opponent latent
        if self.cvae is not None:
            seq_chunked_flattened = rearrange( 
                seq_chunked, "b l s d -> (b l) s d" 
            )
            mu, logvar = self.cvae.encode(seq_chunked_flattened)
            opponent_latent= self.cvae.reparameterize(mu, logvar)
            recons_states= self.cvae.decode(opponent_latent)
            # remove the padding part 
            mu= rearrange( 
                mu, "(b n l) d -> b (n l) d" , l=self.chunk_len, n= num_chunks
            )

            
            logvar= rearrange( 
                logvar, "(b n l) d -> b (n l) d" , l=self.chunk_len, n= num_chunks
            )

            if extra_pad_len>0:
                logvar= logvar[:, :-extra_pad_len, :]
                mu= mu[:, :-extra_pad_len, :]
                
            # flatten batch,seq
            mu=rearrange( 
                logvar, "b l d -> (b l) d" 
            )
            logvar=rearrange( 
                logvar, "b l d -> (b l) d" 
            )
            # STEP-6: Compute the recons loss
            recons_loss = F.mse_loss(input=recons_states , target=seq_chunked_flattened, reduction="none")

            recons_loss = reduce(
                recons_loss,
                'b s d -> b',  # Keep 'b', reduce 's' and 'd'
                'mean'         # Specify the reduction operation
            )

            recons_loss = rearrange(
                recons_loss,
                '(b n l) -> b (n l)',
                 l=self.chunk_len, n= num_chunks
                 )
            # 4. Remove the losses corresponding to the padded chunks, if any
            if extra_pad_len > 0:
                recons_loss = recons_loss[:, :-extra_pad_len]

            # 5. Re-flatten the loss tensor to the final correct shape for masking
            # Shape: (b, l_unpadded) -> (b * l_unpadded)
            recons_loss = rearrange(
                recons_loss,
                'b l -> (b l)'
            )
                
            eta = torch.rand(1).item()
            if eta <= self.get_eps(curr_step, total_steps):
                opponent_latent = self.proj(vae_prior_latents_chunked)
            else: 
                opponent_latent = self.proj(opponent_latent)
                opponent_latent = rearrange( 
                    opponent_latent, "(b l) d -> b l d" , l=self.chunk_len 
                ).unsqueeze(2)

            # STEP-6: Attach this opponent modeling to seq_chunked... 
            seq_chunked = torch.cat([seq_chunked , opponent_latent], dim=-2)
        else: 
            recons_loss, mu, logvar =None, None, None     
            
        seq_chunked = einops.rearrange(seq_chunked, 'bn c r d -> (bn c) r d')
        time_idxs_chunked = einops.rearrange(time_idxs_chunked, 'bn c -> (bn c)')
        final_reasoning , _ = self.inner_forward(seq_chunk=seq_chunked, time_idxs=time_idxs_chunked)
                                            
        # STEP-5: Merge chunks, aggregate over reasoning dimension...
        final_reasoning= einops.rearrange(final_reasoning, '(b n c) d -> b (n c) d', b=B, n=num_chunks)
        
        # STEP-6: Removing the pad embedding created...
        if extra_pad_len>0:
            final_reasoning=final_reasoning[:, :-extra_pad_len, :]   
        return final_reasoning , hidden_state ,mu, logvar ,recons_loss
                
    @property
    def emb_dim(self) -> int:
        return self.d_model
    
    def get_eps(self, curr_step, total_steps):
        EPS_VALUES = [0.98, 0.85, 0.6, 0.02]
        DECAY_BOUNDARIES = [0.1, 0.15, 0.25, 0.3]

        progress = curr_step / total_steps
        for boundary, eps_value in zip(DECAY_BOUNDARIES, EPS_VALUES):
            if progress < boundary:
                return eps_value
        return EPS_VALUES[-1]
            

    def reset_hidden_state(
        self, hidden_state: Optional[State], dones: np.ndarray
    ) -> Optional[State]:
        if hidden_state is None: return None
        assert isinstance(hidden_state, State)
        hidden_state.reset(dones=dones)
        return hidden_state



from amago.nets.transformer import VanillaFlexAttention

@gin.configurable()
class CustomVanillaFlexAttention(VanillaFlexAttention):
    def __init__(self, causal: bool, dropout: float):
        super().__init__(
            causal=causal,
            dropout=dropout,
        )