from amago.agent import binary_filter, Multigammas
import gin
import torch 
import itertools
from amago.loading import Batch, MAGIC_PAD_VAL
from typing import Type, Optional, Tuple, Any, List, Iterable, Dict
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
from metamon.rl.metamon_to_amago import MetamonTstepEncoder
import amago.nets.actor_critic as actor_critic
from custom.traj_encoder import HRMTrajEncoder, VAE_Prior
from custom.hrm_utils.layers import MultiGammaValueEstimator
import gymnasium as gym
import torch.nn as nn
import enum
import os
from amago.agent import MultiTaskAgent
from amago.nets.traj_encoders import TformerTrajEncoder
from metamon import  METAMON_CACHE_DIR
import itertools
from typing import Type, Optional, Tuple, Any, List, Iterable
import torch
import torch.distributions as pyd
from einops import rearrange, repeat
import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
import numpy as np
import wandb
import gin
import gymnasium as gym
from torch.distributions import Distribution

from amago.loading import Batch, MAGIC_PAD_VAL
from amago.nets import actor_critic
from amago.nets.policy_dists import DiscreteLikeContinuous
from amago import utils

class AgentComponent(str, enum.Enum):
    TStepEncoder = "tstep_encoder"
    TrajEncoder = "traj_encoder"
    Target_Actor = "target_actor"
    Actor = "actor"
    Target_Critic = "target_critics"
    Critic = "critics"
    Popart="popart"
    Maximized_Critic="maximized_critics"
    VAE_Prior = ""
    VAE_VALUE_ESTIMATOR="vae value estimator"
    TEACHER_TRAJ_ENCODER="traj_encoder.tformer"
    TEACHER_ACTOR="teacher_actor"
    TEACHER_CRITIC="teacher_critics"
    
    
@gin.configurable
class InitComponent:
    def __init__(self, 
                 ckpt_path:str=None,
                 component_name=None,
                 do_init_with_ckpt:bool=False,
                 is_trainable:bool=True, 
                 from_hf:bool=False):
        self.ckpt_path = ckpt_path
        self.component_name = component_name
        self.do_init_with_ckpt = do_init_with_ckpt
        self.is_trainable = is_trainable
        self.from_hf= from_hf


def _nucleus_sample(probs: torch.Tensor, p: float) -> torch.Tensor:
    """
    Performs nucleus (top-p) sampling on a probability distribution.
    """
    if p >= 1.0 or p <= 0.0:
        return torch.distributions.Categorical(probs=probs).sample()

    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
    
    sorted_indices_to_remove = cumulative_probs > p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = False

    indices_to_remove = torch.zeros_like(probs, dtype=torch.bool).scatter_(
        dim=-1, index=sorted_indices, src=sorted_indices_to_remove
    )
    masked_probs = probs.masked_fill(indices_to_remove, 0.0)
    renormalized_probs = masked_probs / torch.sum(masked_probs, dim=-1, keepdim=True)
    
    return torch.distributions.Categorical(probs=renormalized_probs).sample()

def get_q_dist_for_all_actions(
    critic:actor_critic.NCriticsTwoHot, # The critic network object (e.g., an instance of NCriticsTwoHot)
    state_rep: torch.Tensor,
    num_actions: int,
    num_gammas: int,
) -> pyd.Categorical:
    """
    Efficiently computes Q-value distributions for ALL discrete actions for a given state.

    This function uses a "batching trick" by repurposing the critic's `K` dimension
    (originally for multiple action samples) to query all possible discrete actions
    in a single, parallel forward pass.

    Args:
        critic: The critic network object, which must have a `critic_network_forward` method.
        state_rep: The state representation tensor from the TrajEncoder.
                   Shape: [Batch, Length, State_Dim].
        num_actions: The number of discrete actions in the environment.
        num_gammas: The number of gammas the critic was trained with.

    Returns:
        A categorical distribution object where the logits have a shape of
        [Batch, Length, Num_Actions, Num_Critics, Num_Gammas, Bins].
    """
    # Get batch and length dimensions from the state representation
    B, L, _ = state_rep.shape

    # 1. Create a batch of all possible actions as one-hot vectors.
    # Shape: [num_actions, D_action], where D_action is also num_actions.
    all_actions_one_hot = torch.eye(num_actions, device=state_rep.device)

    # 2. Use einops to repeat this list of actions for every state in the batch
    # and for every gamma, creating a tensor that matches the critic's expected
    # input format (K, B, L, G, D), where we set K = num_actions.
    actions_batch = repeat(
        all_actions_one_hot,
        "k d -> k b l g d",
        b=B, l=L, g=num_gammas
    )

    # 3. Call the critic's core forward pass with the batched actions.
    # The state will be automatically repeated inside this method to match.
    q_dist_all_actions = critic.critic_network_forward(state_rep, actions_batch)

    # 4. Reshape the output. The 'K' dimension of the output now corresponds to actions.
    # We rearrange it to a more intuitive order for downstream processing.
    # Original logits shape: [K, B, L, C, G, O] where K = num_actions
    logits = rearrange(
        q_dist_all_actions.logits,
        "k b l c g o -> b l k c g o",
        k=num_actions
    )

    # 5. Return a new distribution with the reshaped logits.
    return pyd.Categorical(logits=logits)


import torch
import torch.nn.functional as F
from einops import rearrange

def get_refined_policy_distribution(
    critics,
    state_rep: torch.Tensor,
    action_dists,
    num_actions: int,
    num_gammas: int,
    T:float=1,
    # --- Method selection ---
    final_ensemble_method: str = 'dynamic_advantage'
) -> torch.Tensor:
    """
    Creates a refined action policy using a standalone, two-step process.

    This function is independent and does not use `self`.

    1.  **Per-Gamma Refinement**: For EACH of the `G` gamma policies from the actor,
        it is independently refined. This is done via "soft pruning," where the
        policy is modulated by sigmoid weights derived from its corresponding
        critic's Advantage scores.

    2.  **Final Ensemble**: The `G` resulting refined policies are then combined into
        a single, final distribution using a selectable dynamic weighting strategy.

    Args:
        critics: The critic network object (e.g., NCriticsTwoHot instance).
        state_rep: The state representation tensor from the TrajEncoder.
        action_dists: The raw distribution object from the actor.
        num_actions: The number of discrete actions in the environment.
        num_gammas: The number of gammas the models were trained with.
        final_ensemble_method (str): The strategy to combine the refined policies.
            Options: 'mean', 'dynamic_advantage', 'dynamic_entropy'.

    Returns:
        A final, refined probability distribution tensor of shape [B, L, Num_Actions].
    """
    
    # --- Preliminary Step: Get Full Critic Evaluation & Actor Probabilities ---
    q_dist = get_q_dist_for_all_actions(
        critic=critics,
        state_rep=state_rep ,
        num_actions=num_actions,
        num_gammas=num_gammas
    )
    q_values_per_action = critics.bin_dist_to_raw_vals(q_dist)
    q_values_stable = q_values_per_action.mean(dim=-3).squeeze(-1) # Shape: [B, L, A, G]
    actor_probs = action_dists.probs # Shape: [B, L, G, A]
        
    # --- Step 1: Per-Gamma Refinement (As you specified) ---
    
    # Calculate Advantage A(s,a) for EACH Gamma separately
    q_values_aligned = q_values_stable.transpose(-1, -2) # Shape: [B, L, G, A]
    state_value_v_per_gamma = torch.sum(actor_probs * q_values_aligned, dim=-1)
    state_value_v_unsqueezed = state_value_v_per_gamma.unsqueeze(-1)
    advantage_values_per_gamma = q_values_aligned - state_value_v_unsqueezed # Shape: [B, L, G, A]
    
    # # Convert per-gamma advantage scores into soft refinement weights (0 to 1) via sigmoid
    # soft_refinement_weights_per_gamma = torch.sigmoid(advantage_values_per_gamma)
    
    soft_refinement_weights_per_gamma = F.softmax(advantage_values_per_gamma / T, dim=-1)
    refined_probs_per_gamma = actor_probs * soft_refinement_weights_per_gamma
    final_refined_probs_per_gamma = refined_probs_per_gamma / (refined_probs_per_gamma.sum(dim=-1, keepdim=True) + 1e-9)
    
    # # Apply the soft-pruning to each gamma's actor policy
    # refined_probs_per_gamma = actor_probs * soft_refinement_weights_per_gamma
    
    # # Re-normalize EACH of the G distributions so they are valid policies
    # final_refined_probs_per_gamma = refined_probs_per_gamma / (
    #     refined_probs_per_gamma.sum(dim=-1, keepdim=True) + 1e-9
    # )
    # # Shape of final_refined_probs_per_gamma: [B, L, G, A]

    # --- Step 2: Combine the Refined Policies into a Final Distribution ---
    
    if final_ensemble_method == 'mean':
        # a) Simple, robust mean of the refined policies
        final_probs = torch.mean(final_refined_probs_per_gamma, dim=-2)
        
    elif final_ensemble_method == 'dynamic_advantage':
        # b) Weighted ensemble based on the "optimism" of each refined policy
        # We calculate the expected advantage of each *refined* policy
        expected_advantage_of_refined = torch.sum(
            final_refined_probs_per_gamma * advantage_values_per_gamma, dim=-1
        )
        # Use softmax on these scores to get the final weights
        final_weights = F.softmax(expected_advantage_of_refined, dim=-1).unsqueeze(-1)
        # Apply weights to get the final ensembled distribution
        final_probs = torch.sum(final_refined_probs_per_gamma * final_weights, dim=-2)
        
    elif final_ensemble_method == 'dynamic_entropy':
        # b) Weighted ensemble based on the "confidence" of each refined policy
        # Calculate the entropy of each refined distribution
        entropy = -torch.sum(
            final_refined_probs_per_gamma * torch.log2(final_refined_probs_per_gamma + 1e-9), dim=-1
        )
        # Low entropy (high confidence) gets a higher weight
        final_weights = F.softmax(-entropy, dim=-1).unsqueeze(-1)
        final_probs = torch.sum(final_refined_probs_per_gamma * final_weights, dim=-2)
    
    else: # Default to simple mean for safety
        final_probs = torch.mean(final_refined_probs_per_gamma, dim=-2)

    return final_probs

    
@gin.configurable
class HRM_MultiTaskAgent(MultiTaskAgent):
    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        tstep_encoder_type: Type[MetamonTstepEncoder],
        traj_encoder_type: Type[HRMTrajEncoder],
        max_seq_len: int,
        num_critics: int = 4,
        num_critics_td: int = 2,
        online_coeff: float = 0.0,
        offline_coeff: float = 1.0,
        gamma: float = 0.999,
        reward_multiplier: float = 10.0,
        tau: float = 0.003,
        use_metamon_models:bool=False, 
        fake_filter: bool = False,
        no_opponent_inference: bool=True,
        num_actions_for_value_in_critic_loss: int = 1,
        num_actions_for_value_in_actor_loss: int = 3,
        fbc_filter_func: callable = binary_filter,
        popart: bool = True,
        use_target_actor: bool = True,
        use_multigamma: bool = True,
        actor_type: Type[actor_critic.BaseActorHead] = actor_critic.Actor,
        critic_type: Type[actor_critic.BaseCriticHead] = actor_critic.NCriticsTwoHot,
        pass_obs_keys_to_actor: Optional[Iterable[str]] = None,
        vae_prior_type: Type[VAE_Prior]=None, 
        vae_value_estimator_type: Type[MultiGammaValueEstimator]=None, 
        # below args for ckpt initialization
        tstep_component_type: Type[InitComponent] = InitComponent,
        traj_component_type: Type[InitComponent] = InitComponent, 
        actor_component_type: Type[InitComponent] = InitComponent, 
        critic_component_type: Type[InitComponent]= InitComponent, 
        maximized_critics_component_type: Type[InitComponent]= InitComponent, 
        target_critic_component_type: Type[InitComponent]= InitComponent, 
        target_actor_component_type: Type[InitComponent]= InitComponent, 
        popart_component_type: Type[InitComponent]= InitComponent, 
        vae_prior_component_type: Type[InitComponent]= None, 
        vae_value_estimator_component_type: Type[InitComponent]= None, 
        
        # Knowledge distillation of traj-encoder
        teacher_traj_encoder_type: Type[TformerTrajEncoder]=None,
        teacher_actor_type: Type[actor_critic.BaseActorHead]=None,
        teacher_critic_type: Type[actor_critic.BaseCriticHead]=None,
        teacher_traj_encoder_component_type: Type[InitComponent]= None,
        teacher_actor_component_type: Type[InitComponent]= None,
        teacher_critic_component_type: Type[InitComponent]= None,
    ):
        super().__init__(
                obs_space=obs_space,
                rl2_space=rl2_space,
                action_space=action_space,
                max_seq_len=max_seq_len,
                tstep_encoder_type=tstep_encoder_type,
                traj_encoder_type=traj_encoder_type,
                num_critics=num_critics,
                num_critics_td=num_critics_td,
                online_coeff=online_coeff,
                offline_coeff=offline_coeff,
                gamma=gamma,
                reward_multiplier=reward_multiplier,
                tau=tau,
                fake_filter=fake_filter,
                num_actions_for_value_in_critic_loss=num_actions_for_value_in_critic_loss,
                num_actions_for_value_in_actor_loss=num_actions_for_value_in_actor_loss,
                fbc_filter_func=fbc_filter_func,
                popart=popart,
                use_target_actor=use_target_actor,
                use_multigamma= use_multigamma,
                actor_type=actor_type,
                critic_type= critic_type,
                pass_obs_keys_to_actor=pass_obs_keys_to_actor,
            )
        self.obs_space = obs_space
        self.rl2_space = rl2_space

        self.action_space = action_space
        self.multibinary = False
        self.discrete = False
        if isinstance(self.action_space, gym.spaces.Discrete):
            self.action_dim = self.action_space.n
            self.discrete = True
        elif isinstance(self.action_space, gym.spaces.MultiBinary):
            self.action_dim = self.action_space.n
            self.multibinary = True
        elif isinstance(self.action_space, gym.spaces.Box):
            self.action_dim = self.action_space.shape[-1]
        else:
            raise ValueError(f"Unsupported action space: `{type(self.action_space)}`")

        self.reward_multiplier = reward_multiplier
        self.pad_val = MAGIC_PAD_VAL
        self.fake_filter = fake_filter
        self.num_actions_for_value_in_critic_loss = num_actions_for_value_in_critic_loss
        self.num_actions_for_value_in_actor_loss = num_actions_for_value_in_actor_loss
        self.fbc_filter_func = fbc_filter_func
        self.offline_coeff = offline_coeff
        self.online_coeff = online_coeff
        self.tau = tau
        self.use_target_actor = use_target_actor
        self.max_seq_len = max_seq_len
        self.no_opponent_inference=no_opponent_inference
        self.use_metamon_models=use_metamon_models
        
        self.tstep_encoder = tstep_encoder_type(
            obs_space=obs_space,
            rl2_space=rl2_space,
        )
        self.scratch_tokens = self.tstep_encoder.scratch_tokens 

        self.traj_encoder = traj_encoder_type(
            tstep_dim=self.tstep_encoder.d_model if not self.use_metamon_models else self.tstep_encoder.d_model*self.scratch_tokens,
            max_seq_len=max_seq_len,
        )
        self.emb_dim = self.traj_encoder.emb_dim
        
        if self.discrete: multigammas = Multigammas().discrete
        else: multigammas = Multigammas().continuous

        # provided hparam `gamma` will stay in the -1 index
        # of gammas, actor, and critic outputs.
        gammas = (multigammas if use_multigamma else []) + [gamma]
        self.gammas = torch.Tensor(gammas).float()
        assert num_critics_td <= num_critics
        self.num_critics = num_critics
        self.num_critics_td = num_critics_td

        self.popart = actor_critic.PopArtLayer(gammas=len(gammas), enabled=popart)

        ac_kwargs = {
            "state_dim": self.traj_encoder.emb_dim ,
            "action_dim": self.action_dim,
            "discrete": self.discrete,
            "gammas": self.gammas,
        }
        self.critics = critic_type(**ac_kwargs, num_critics=num_critics)
        self.target_critics = critic_type(**ac_kwargs, num_critics=num_critics)
        self.maximized_critics = critic_type(**ac_kwargs, num_critics=num_critics)
        self.actor = actor_type(**ac_kwargs)
        self.target_actor = actor_type(**ac_kwargs)

        # opponent modeling components -->
        if vae_prior_type is not None:
            self.vae_prior= vae_prior_type()
            self.vae_value_estimator= vae_value_estimator_type()
        
        # initializing component from ckpts
        self.components = [tstep_component_type(), traj_component_type(), 
                           actor_component_type(), critic_component_type(),
                           target_critic_component_type(), target_actor_component_type(), 
                           popart_component_type() , maximized_critics_component_type(), 
                           ]
        if vae_prior_component_type is not None:
            self.components.append(vae_prior_component_type())
            self.components.append(vae_value_estimator_component_type())
            
        if teacher_traj_encoder_type is not None:
            self.teacher_traj_encoder= teacher_traj_encoder_type(
                            tstep_dim=self.tstep_encoder.d_model*self.scratch_tokens,
                            max_seq_len=max_seq_len,
            )
            teacher_ac_kwargs = {
                "state_dim": self.teacher_traj_encoder.emb_dim,
                "action_dim": self.action_dim,
                "discrete": self.discrete,
                "gammas": self.gammas,
            }
            self.kd_proj = nn.Linear(self.traj_encoder.emb_dim, self.teacher_traj_encoder.emb_dim)
            self.teacher_actor=teacher_actor_type(**teacher_ac_kwargs)
            self.teacher_critics=teacher_critic_type(**teacher_ac_kwargs, num_critics=num_critics)
            self.components.append(teacher_traj_encoder_component_type())
            self.components.append(teacher_actor_component_type())
            self.components.append(teacher_critic_component_type())
        
        for c in self.components:
            match c.component_name:
                case AgentComponent.TStepEncoder: 
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name, 
                                              component=self.tstep_encoder, state_dict_prefix=AgentComponent.TStepEncoder.value, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.tstep_encoder)
                case AgentComponent.TrajEncoder:
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name, 
                                                component=self.traj_encoder, state_dict_prefix=AgentComponent.TrajEncoder.value, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.traj_encoder)
                case AgentComponent.Actor:
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name, 
                                                component=self.actor, state_dict_prefix=AgentComponent.Actor.value, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.actor)
                case AgentComponent.Critic:
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name, 
                                                component=self.critics, state_dict_prefix=AgentComponent.Critic.value, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.critics)   
                case AgentComponent.Target_Actor:
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name, 
                                                component=self.target_actor, state_dict_prefix=AgentComponent.Target_Actor.value, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.target_actor)
                case AgentComponent.Target_Critic:
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name, 
                                                component=self.target_critics, state_dict_prefix=AgentComponent.Target_Critic.value, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.target_critics)
                case AgentComponent.Popart:
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name, 
                                                component=self.popart, state_dict_prefix=AgentComponent.Popart.value, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.popart)
                case AgentComponent.Maximized_Critic:
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name, 
                                                component=self.maximized_critics, state_dict_prefix=AgentComponent.Maximized_Critic.value, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.maximized_critics)
                case AgentComponent.VAE_Prior:
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name, 
                                                component=self.vae_prior, state_dict_prefix=AgentComponent.VAE_Prior.value, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.vae_prior)
                case AgentComponent.VAE_VALUE_ESTIMATOR:
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name, 
                                                component=self.vae_value_estimator, state_dict_prefix=AgentComponent.VAE_VALUE_ESTIMATOR.value, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.vae_value_estimator)
                case AgentComponent.TEACHER_TRAJ_ENCODER:
                    prefix = AgentComponent.TEACHER_TRAJ_ENCODER.value.split('.')[0]
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name,
                                                component=self.teacher_traj_encoder, state_dict_prefix=prefix, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.teacher_traj_encoder)
                case AgentComponent.TEACHER_ACTOR:
                    prefix= AgentComponent.TEACHER_ACTOR.value.split("_")[1]
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name,
                                                component=self.teacher_actor, state_dict_prefix=prefix, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.teacher_actor)
                case AgentComponent.TEACHER_CRITIC:
                    prefix= AgentComponent.TEACHER_CRITIC.value.split("_")[1]
                    if c.do_init_with_ckpt: self.initialize_component(ckpt_path=c.ckpt_path, component_name=c.component_name,
                                                component=self.teacher_critics, state_dict_prefix=prefix, from_hf=c.from_hf)
                    if not c.is_trainable: self.freeze_component(self.teacher_critics)
                    
        # full weight copy to targets
        self.hard_sync_targets()
        self.pass_obs_keys_to_actor = pass_obs_keys_to_actor or []


    @property
    def trainable_params(self):
        """Iterable over all trainable parameters, which should be passed to the optimizer."""
        return itertools.chain(
            # self.tstep_encoder.parameters(),
            # self.traj_encoder.parameters(),
            # self.critics.parameters(),
            # self.actor.parameters(),
            
            *[model.parameters() for model, model_metadata in zip([self.tstep_encoder, self.traj_encoder, self.actor, self.critics], 
                                                                 self.components) if model_metadata.is_trainable and model_metadata.component_name!=AgentComponent.Popart]
        )
        
    def initialize_component(self, component, component_name, state_dict_prefix,  ckpt_path, from_hf:bool=False):
        if from_hf:
            ckpt_path = os.path.join(METAMON_CACHE_DIR, "metamon_models", ckpt_path)
            if not os.path.exists(ckpt_path):
                try:
                    from custom.utils import download_hf_ckpt
                    allow_pattern=ckpt_path[len(f"{METAMON_CACHE_DIR}/metamon_models/"):]
                    download_hf_ckpt(allow_patterns=allow_pattern)
                except Exception as e: 
                    print(f"ERROR: {e}")
                    return
        else: 
            root="/".join(METAMON_CACHE_DIR.split('/')[:-1])
            ckpt_path=os.path.join(root, ckpt_path)
            if not os.path.exists(ckpt_path):
                raise ValueError(f"The ckpt -> ({ckpt_path}) does not exists")
        try:
            ckpt=torch.load(ckpt_path)
            if state_dict_prefix != "":
                state_dict={k.replace(f"{state_dict_prefix}.", '', 1):v for k,v in ckpt.items() if k.startswith(state_dict_prefix)}
            else: state_dict=ckpt 
            component.load_state_dict(state_dict)
            print(f"✅ SUCCESS: {component_name} initialized from ckpt ({ckpt_path})")
        except Exception as e: 
            print(f"😥 FAILED: {component_name} initialization from ckpt ({ckpt_path}).\nERROR: {e}")
    
    def freeze_component(self, component:nn.Module):
        for param in component.parameters():
            param.requires_grad = False

    def _ensemble_gamma_policies(
        self,
        state_rep:torch.Tensor, 
        action_dists,
        method: str,
        T:float=1,
        entropy_fallback_threshold: float = 1
    ) -> torch.Tensor:
        """
        Combines multiple gamma-specific policies into a single probability distribution.
        """
        if method == 'entropy':
            probs = action_dists.probs + 1e-9
            entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
            
            all_uncertain = torch.all(entropy > entropy_fallback_threshold, dim=-1, keepdim=True)
            default_probs = probs[..., -1, :]
            
            confidence_weights = F.softmax(-entropy, dim=-1).unsqueeze(-1)
            ensembled_probs = torch.sum(probs * confidence_weights, dim=-2)
            
            return torch.where(all_uncertain, default_probs, ensembled_probs)
        
        elif method=="q_value":
            return get_refined_policy_distribution(
                critics=self.critics,
                state_rep=state_rep ,
                action_dists=action_dists,
                num_gammas=len(self.gammas),
                num_actions=self.action_dim,
                T=T
            )
        
        elif method == 'mean':
            return torch.mean(action_dists.probs, dim=-2)
            
        elif method == 'none' or method == 'default':
            return action_dists.probs[..., -1, :]
            
        else:
            raise ValueError(f"Unknown ensembling method: {method}")

    def _select_action_from_probs(
        self,
        final_probs: torch.Tensor,
        sample: bool,
        method: str,
        nucleus_p: float = 0.9,
    ) -> torch.Tensor:
        """
        Selects an action from a probability distribution using the specified method.
        """
        if not sample:
            return torch.argmax(final_probs, dim=-1)
        
        # If sampling is enabled, choose the method
        if method == 'nucleus':
            return _nucleus_sample(final_probs, p=nucleus_p)
        elif method == 'standard':
            return torch.distributions.Categorical(probs=final_probs).sample()
        else:
            raise ValueError(f"Unknown sampling method: {method}")
         

    def log_actor_diagnostics(
        self, 
        action_dists: Distribution, 
        time_idxs: torch.Tensor
    ):
        """
        Logs diagnostic information about the actor's raw output distributions.
        This is intended for debugging and analysis during inference.
        
        It prints:
        1. The entropy of each gamma-specific policy.
        2. The max confidence (probability) and the corresponding action index.
        """
        # We'll log info for the first item in the batch (b=0) 
        # and the latest item in the sequence (l=-1).
        
        with torch.no_grad():
            # Get probabilities: [B, L, G, A]
            probs = action_dists.probs
            
            # Get entropy: [B, L, G]
            entropies = action_dists.entropy()
            
            # Select the first batch and last timestep
            try:
                # Assuming L=1 during inference, [0, -1] selects [b=0, l=0]
                probs_to_log = probs[0, -1, :, :]      # Shape: [G, A]
                entropies_to_log = entropies[0, -1, :]  # Shape: [G]
            except IndexError:
                print("Warning: Could not log actor diagnostics. Tensor shape unexpected.")
                return

            # Get the current time step (assuming it's a 3D tensor [B, L, 1])
            try:
                # --- FIXED TIME_STEP LOGIC ---
                # Index to the same batch (0) and time step (-1) as probs/entropy
                # The final [0] selects the value from the last dim of size 1
                time_step = time_idxs[0, -1, 0].item()
            except Exception as e:
                print(f"Warning: Could not get time_step. Error: {e}")
                time_step = "N/A" # Fallback

            # Calculate max confidence and corresponding actions
            # max_probs_to_log: [G]
            # max_actions_to_log: [G]
            max_probs_to_log, max_actions_to_log = torch.max(probs_to_log, dim=-1)

            # --- Start Logging ---
            print(f"\n--- Actor Diagnostics (Step {time_step}) ---")
            
            # --- FIXED RUNTIMEERROR ---
            # Changed `or not self.gammas` (ambiguous for a tensor)
            # to `or self.gammas is None` (unambiguous check)
            if not hasattr(self, 'gammas') or self.gammas is None:
                print("  Warning: 'self.gammas' not found or is None. Cannot label gammas.")
                return
            # --- END FIX ---

            num_gammas_to_log = len(self.gammas)
            
            # Safety check in case probs shape doesn't match gammas list
            if num_gammas_to_log != entropies_to_log.shape[0]:
                print(f"  Warning: Mismatch between self.gammas ({num_gammas_to_log}) and"
                    f" policy gammas ({entropies_to_log.shape[0]}). Truncating log.")
                num_gammas_to_log = min(num_gammas_to_log, entropies_to_log.shape[0])


            for i in range(num_gammas_to_log):
                gamma = self.gammas[i]
                # Handle if gamma is a tensor
                gamma_val = gamma.item() if isinstance(gamma, torch.Tensor) else gamma
                
                entropy_val = entropies_to_log[i].item()
                max_prob_val = max_probs_to_log[i].item()
                max_action_idx = max_actions_to_log[i].item()
                
                # Print formatted log line
                print(f"  [Gamma = {gamma_val:.2f}]: "
                    f"Entropy = {entropy_val:<6.3f} | "
                    f"Max Confidence = {max_prob_val:<5.2f} "
                    f"(for action {max_action_idx})")
            
            print("----------------------------------------")

        
    def get_actions(
        self,
        obs: Dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Any = None,
        sample: bool = True,

        ensembling_method: str = 'none', # Options: 'entropy', 'mean', 'none', 'q_value'
        sampling_method: str = 'nucleus',   # Options: 'nucleus', 'standard'
        nucleus_p: float = 0.9,
        entropy_fallback_threshold: float = 1.5,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Get rollout actions from the current policy with advanced sampling and ensembling.
        """
        # Step 1: Get the sequence embedding from the core models
        tstep_emb = self.tstep_encoder(obs=obs, rl2s=rl2s)
        if not self.use_metamon_models:
            tstep_emb_ = rearrange(
                tstep_emb, "b l (s d) -> b l s d", s=self.tstep_encoder.scratch_tokens
            )
        else: tstep_emb_= tstep_emb
        
        func = self.traj_encoder.forward if self.no_opponent_inference else self.traj_encoder.forward_opponent_modeling
        traj_emb_t, hidden_state = func(tstep_emb_, time_idxs=time_idxs, hidden_state=hidden_state)

        # Step 2: Get the raw action distributions from the actor
        action_dists = self.actor(
            traj_emb_t,
            straight_from_obs={k: obs[k] for k in self.pass_obs_keys_to_actor},
        )
        # self.log_actor_diagnostics(action_dists, time_idxs)
        # Step 3: Use a helper function to ensemble the gamma policies
        T=3 # temperature
        final_probs = self._ensemble_gamma_policies(
            state_rep=traj_emb_t,
            action_dists=action_dists,
            method=ensembling_method,
            entropy_fallback_threshold=entropy_fallback_threshold, 
            T=T
        )
        # Step 4: Use a helper function to select an action from the final distribution
        actions = self._select_action_from_probs(
            final_probs,
            sample=sample,
            method=sampling_method if self.discrete else 'mean', # Fallback for continuous
            nucleus_p=nucleus_p
        )

        # Final Step: Format the output
        actions = actions.unsqueeze(-1)
        dtype = torch.uint8 if (self.discrete or self.multibinary) else torch.float32
        return actions.to(dtype=dtype), hidden_state
        
        
    # def get_actions(
    #     self,
    #     obs: dict[str, torch.Tensor],
    #     rl2s: torch.Tensor,
    #     time_idxs: torch.Tensor,
    #     hidden_state=None,
    #     sample: bool = True,
    # ) -> Tuple[torch.Tensor, Any]:
    #     """Get rollout actions from the current policy.

    #     Note the standard torch `forward` implements the training step, while `get_actions`
    #     is the inference step. Most of the arguments here are easily gathered from the
    #     AMAGOEnv gymnasium wrapper. See `amago.experiment.Experiment.interact` for an example.

    #     Args:
    #         obs: Dictionary of (batched) observation tensors. AMAGOEnv makes all
    #             observations into dicts.
    #         rl2s: Batched Tensor of previous action and reward. AMAGOEnv makes these.
    #         time_idxs: Batched Tensor indicating the global timestep of the episode.
    #             Mainly used for position embeddings when the sequence length is much shorter
    #             than the episode length.
    #         hidden_state: Hidden state of the TrajEncoder. Defaults to None.
    #         sample: Whether to sample from the action distribution or take the argmax
    #             (discrete) or mean (continuous). Defaults to True.

    #     Returns:
    #         tuple:
    #             - Batched Tensor of actions to take in each parallel env *for the primary
    #               ("test-time") discount factor* `Agent.gamma`.
    #             - Updated hidden state of the TrajEncoder.
    #     """
        
    #     tstep_emb = self.tstep_encoder(obs=obs, rl2s=rl2s)
    #     # sequence model embedding [batch, length, d_emb]
    #     tstep_emb_ = rearrange( 
    #             tstep_emb, "b l (s d) -> b l s d", s=self.tstep_encoder.scratch_tokens
    #         )
    #     func= self.traj_encoder.forward if self.no_opponent_inference else self.traj_encoder.forward_opponent_modeling
    #     traj_emb_t, hidden_state = func(tstep_emb_, time_idxs=time_idxs, hidden_state=hidden_state)
        
    #     # generate action distribution [batch, length, len(self.gammas), d_action]
    #     action_dists = self.actor(
    #         traj_emb_t,
    #         straight_from_obs={k: obs[k] for k in self.pass_obs_keys_to_actor},
    #     )
    #     if sample: actions = action_dists.sample()
    #     else:
    #         if self.discrete: actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
    #         else: actions = action_dists.mean
    #     # get intended gamma distribution (always in -1 idx)
    #     actions = actions[..., -1, :]
    #     dtype = torch.uint8 if (self.discrete or self.multibinary) else torch.float32
    #     return actions.to(dtype=dtype), hidden_state


    def forward(self, batch: Batch, log_step: bool):
        # fmt: off
        self.update_info = {}  # holds wandb stats

        ##################
        ## Timestep Emb ##
        ##################
        active_log_dict = self.update_info if log_step else None
        with torch.no_grad():
            o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict)
        straight_from_obs = {k : batch.obs[k] for k in self.pass_obs_keys_to_actor}

        ###################
        ## Get Organized ##
        ###################
        B, L, D_o = o.shape
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1.0, 1.0)
        _B, _L, D_action = a.shape
        assert _L == L - 1
        G = len(self.gammas)
        K_c = self.num_actions_for_value_in_critic_loss
        a_buffer = F.pad(a, (0, 0, 0, 1), "replicate")
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")
        C = len(self.critics)
        assert batch.rews.shape == (B, L - 1, 1)
        assert batch.dones.shape == (B, L - 1, 1)
        r = repeat((self.reward_multiplier * batch.rews).float(), f"b l r -> b l 1 {G} r")
        d = repeat(batch.dones.float(), f"b l d -> b l 1 {G} d")
        gamma = self.gammas.to(r.device).unsqueeze(-1)
        D_emb = self.traj_encoder.emb_dim
        Bins = self.critics.num_bins
        state_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).float()[:, 1:, ...]
        actor_mask = F.pad(state_mask, (0, 0, 0, 1), "constant", 0.0)
        actor_mask = repeat(actor_mask, f"b l 1 -> b l {G} 1")
        critic_mask = repeat(state_mask, f"b l 1 -> b l {C} {G} 1")

        ########################
        ## Sequence Embedding ##
        ########################
        o_ = rearrange( 
                o, "b l (s d) -> b l s d", s=self.scratch_tokens
            )
        s_rep, _ = self.traj_encoder(seq=o_, time_idxs=batch.time_idxs, 
                                                hidden_state=None, log_dict=active_log_dict)
        assert s_rep.shape == (B, L, D_emb)
        

        ################
        ## a ~ \pi(s) ##
        ################
        a_dist = self.actor(s_rep, log_dict=active_log_dict, straight_from_obs=straight_from_obs)
        
        if self.discrete:
            a_dist = DiscreteLikeContinuous(a_dist)
        if log_step:
            policy_stats = self._policy_stats(actor_mask, a_dist)
            self.update_info.update(policy_stats)
            policy_entropy = a_dist.entropy().mean()
            self.update_info["actor/policy_entropy"] = policy_entropy
                        
        critic_loss = None
        if not self.fake_filter or self.online_coeff > 0: # if we use the critic to train the actor
            ################
            ## TD Targets ##
            ################
            with torch.no_grad():
                if self.use_target_actor:
                    a_prime_dist = self.target_actor(s_rep, straight_from_obs=straight_from_obs)
                    if self.discrete:
                        a_prime_dist = DiscreteLikeContinuous(a_prime_dist)
                else:
                    a_prime_dist = a_dist
                ap = a_prime_dist.sample((K_c,)) # a' ~ \pi(s')
                assert ap.shape == (K_c, B, L, G, D_action)
                sp_ap_gp = (s_rep[:, 1:, ...].detach(), ap[:, :, 1:, ...].detach())
                q_targ_sp_ap_gp = self.target_critics(*sp_ap_gp) # Q(s', a')
                assert q_targ_sp_ap_gp.probs.shape == (K_c, B, L - 1, C, G, Bins)
                q_targ_sp_ap_gp = self.target_critics.bin_dist_to_raw_vals(q_targ_sp_ap_gp).mean(0)
                assert q_targ_sp_ap_gp.shape == (B, L - 1, C, G, 1)
                # y = r + gamma * (1.0 - d) * Q(s', a')
                ensemble_td_target = r + gamma * (1.0 - d) * q_targ_sp_ap_gp
                assert ensemble_td_target.shape == (B, L - 1, C, G, 1)
                td_target = self._critic_ensemble_to_td_target(ensemble_td_target)
                assert td_target.shape == (B, L - 1, 1, G, 1)
                self.popart.update_stats(
                    td_target, mask=critic_mask.all(2, keepdim=True)
                )
                assert td_target.shape == (B, L - 1, 1, G, 1)
                td_target_labels = self.target_critics.raw_vals_to_labels(td_target)
                td_target_labels = repeat(
                    td_target_labels, f"b l 1 g bins -> b l {C} g bins"
                )
                assert td_target_labels.shape == (B, L - 1, C, G, Bins)

            #################
            ## Critic Loss ##
            #################
            s_a_g = (s_rep, a_buffer.unsqueeze(0))
            q_s_a_g = self.critics(*s_a_g, log_dict=active_log_dict) # Q(s, a)
            assert q_s_a_g.probs.shape == (1, B, L, C, G, Bins)
            # mean squared bellman error --> cross entropy w/ bin classification labels
            critic_loss = F.cross_entropy(
                rearrange(q_s_a_g.logits[0, :, :-1, ...], "b l c g u -> (b l c g) u"),
                rearrange(td_target_labels, "b l c g u -> (b l c g) u"),
                reduction="none",
            )
            critic_loss = rearrange(
                critic_loss, "(b l c g) -> b l c g 1", b=B, l=L - 1, c=C, g=G
            )
            assert critic_loss.shape == (B, L - 1, C, G, 1)
            scalar_q_s_a_g = self.critics.bin_dist_to_raw_vals(q_s_a_g).squeeze(0)
            if log_step:
                td_stats = self._td_stats(
                    mask=critic_mask,
                    raw_q_s_a_g=self.popart.normalize_values(scalar_q_s_a_g)[:, :-1, ...],
                    q_s_a_g=scalar_q_s_a_g[:, :-1, ...],
                    r=r,
                    d=d,
                    td_target=td_target,
                    raw_q_bins=q_s_a_g.probs[0, :, :-1],
                )
                popart_stats = self._popart_stats()
                self.update_info.update(td_stats | popart_stats)

        actor_loss = 0.0
        K_a = self.num_actions_for_value_in_actor_loss
        if self.offline_coeff > 0:
            #####################################################
            ## "Offline" (Advantage Weighted/Filtered BC) Loss ##
            #####################################################
            if not self.fake_filter:
                # f(A(s, a))
                with torch.no_grad():
                    a_agent = a_dist.sample((K_a,))
                    q_s_a_agent = self.critics(s_rep.detach(), a_agent)
                    assert q_s_a_agent.probs.shape == (K_a, B, L, C, G, Bins)
                    # mean over actions and critic ensemble
                    val_s = self.critics.bin_dist_to_raw_vals(q_s_a_agent)
                    assert val_s.shape == (K_a, B, L, C, G, 1)
                    # A(s, a) = Q(s, a) - V(s) = mean_over_critics(Q(s, a)) - mean_over_critics(mean_over_actions(Q(s, a ~ pi)))
                    advantage_s_a = scalar_q_s_a_g.mean(2) - val_s.mean((0, 3))
                    assert advantage_s_a.shape == (B, L, G, 1)
                    filter_ = self.fbc_filter_func(advantage_s_a)[:, :-1, ...].float()
                    binary_filter_ = binary_filter(advantage_s_a)[:, :-1, ...].float()
            else:
                # Behavior Cloning (f(A(s, a)) = 1)
                filter_ = binary_filter_ = torch.ones(
                    (B, L - 1, G, 1), dtype=torch.float32, device=s_rep.device
                )
            # log pi(a | s)
            if self.discrete:
                logp_a = a_dist.log_prob(a_buffer).unsqueeze(-1)
            elif self.multibinary:
                logp_a = a_dist.log_prob(a_buffer).mean(-1, keepdim=True)
            else:
                logp_a = a_dist.log_prob(a_buffer).sum(-1, keepdim=True)
            # throw away last action that was a duplicate
            logp_a = logp_a[:, :-1, ...]
            actor_loss = actor_loss+  self.offline_coeff * -(filter_.detach() * logp_a)
            if log_step:
                filter_stats = self._filter_stats(actor_mask, logp_a, filter_)
                self.update_info.update(filter_stats)

        if self.online_coeff > 0:
            #########################
            ## "Online" (DPG) Loss ##
            #########################
            # TODO: possible to recycle this q_val for the FBC loss above, as is done in Agent.
            # For now, only call rsample when specifically using online_coeff > 0 (since it's usually turned off)
            assert self.actor.actions_differentiable, "online-style actor loss is not compatible with action distribution"
            a_agent_dpg = torch.stack([a_dist.rsample() for _ in range(K_a)], dim=0)
            q_s_a_agent = self.maximized_critics(s_rep.detach(), a_agent_dpg)
            q_s_a_agent = self.popart.normalize_values(
                # mean over K actions, min over critic ensemble
                self.maximized_critics.bin_dist_to_raw_vals(q_s_a_agent).mean(0).min(2).values
            )
            actor_loss =actor_loss+ self.online_coeff * -(q_s_a_agent[:, :-1, ...])

        return critic_loss, actor_loss

    def forward_opponent_modeling(self, batch: Batch, log_step: bool , curr_step:int , total_steps:int):
        # fmt: off
        self.update_info = {}  # holds wandb stats

        ##################
        ## Timestep Emb ##
        ##################
        active_log_dict = self.update_info if log_step else None
        straight_from_obs = {k : batch.obs[k] for k in self.pass_obs_keys_to_actor}
        with torch.no_grad():
            o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict)
        
        ###################
        ## Get Organized ##
        ###################
        B, L, D_o = o.shape
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1.0, 1.0)
        _B, _L, D_action = a.shape
        assert _L == L - 1
        G = len(self.gammas)
        K_c = self.num_actions_for_value_in_critic_loss
        a_buffer = F.pad(a, (0, 0, 0, 1), "replicate")
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")
        C = len(self.critics)
        assert batch.rews.shape == (B, L - 1, 1)
        assert batch.dones.shape == (B, L - 1, 1)
        r = repeat((self.reward_multiplier * batch.rews).float(), f"b l r -> b l 1 {G} r")
        d = repeat(batch.dones.float(), f"b l d -> b l 1 {G} d")
        gamma = self.gammas.to(r.device).unsqueeze(-1)
        D_emb = self.traj_encoder.emb_dim
        Bins = self.critics.num_bins
        state_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).float()[:, 1:, ...]
        actor_mask = F.pad(state_mask, (0, 0, 0, 1), "constant", 0.0)
        actor_mask = repeat(actor_mask, f"b l 1 -> b l {G} 1")
        critic_mask = repeat(state_mask, f"b l 1 -> b l {C} {G} 1")

        ########################
        ## Sequence Embedding ##
        ########################
        o_ = rearrange( 
                o, "b l (s d) -> b l s d", s=self.scratch_tokens
            )
        with torch.no_grad():
            # pass through VAE prior
            vae_input = rearrange( 
                    o_, "b l s d -> (b l) s d"
                )
            prior_mu,  prior_logvar = self.vae_prior.encode(vae_input)
            g_ = self.vae_prior.reparameterize(mu=prior_mu, logvar=prior_logvar)
        
        # sub-goal selection
        vae_state_values = self.vae_value_estimator(g_ , vae_input)    #TODO sample s_g^t based on values...  expects output shape: (B,L,G)
        vae_state_values= rearrange( 
                    vae_state_values, "(b l) g -> b l g" , b= B
                )
        idx=-1
        selector_values= vae_state_values[:, :, idx]  # shape: [B, L]
        future_idxs = self.find_lookahead_extrema_index(tensor=selector_values , H=3)
        s_g_t=self.apply_lookahead_indices(data_tensor=o_, idx_tensor=future_idxs)
        
        with torch.no_grad():
            # pass through VAE prior
            vae_input = rearrange( 
                    s_g_t, "b l s d -> (b l) s d"
                )
            prior_mu, prior_logvar = self.vae_prior.encode(vae_input)        
        
        
        vae_value_loss=0    # initialized
        g_ = rearrange( 
                g_, "(b l) d -> b l d", b=B
            ).unsqueeze(2)
        
        (s_rep, _, posterior_mu, 
         posterior_logvar ,recons_loss) = self.traj_encoder.forward_opponent_modeling(seq=o_, time_idxs=batch.time_idxs, 
                                                                                    hidden_state=None, log_dict=active_log_dict, 
                                                                                    vae_prior_latents= g_ , curr_step=curr_step , 
                                                                                    total_steps=total_steps)
        assert s_rep.shape == (B, L, D_emb)
        if recons_loss is not None:
            recons_loss=self.calculate_masked_loss(loss_tensor=recons_loss, mask=state_mask)
        
        
        # compute the KL loss between prior and posterior
        KL_loss = self.get_kl_loss(prior_mu=prior_mu, prior_logvar=prior_logvar, 
                                  posterior_mu=posterior_mu, posterior_logvar=posterior_logvar)
        KL_loss=self.calculate_masked_loss(loss_tensor=KL_loss, mask=state_mask)
        
        ################
        ## a ~ \pi(s) ##
        ################
        a_dist = self.actor(s_rep, log_dict=active_log_dict, straight_from_obs=straight_from_obs)
        if self.discrete:
            a_dist = DiscreteLikeContinuous(a_dist)
        if log_step:
            policy_stats = self._policy_stats(actor_mask, a_dist)
            self.update_info.update(policy_stats)

        critic_loss = None
        if not self.fake_filter or self.online_coeff > 0: # if we use the critic to train the actor
            ################
            ## TD Targets ##
            ################
            with torch.no_grad():
                if self.use_target_actor:
                    a_prime_dist = self.target_actor(s_rep, straight_from_obs=straight_from_obs)
                    if self.discrete:
                        a_prime_dist = DiscreteLikeContinuous(a_prime_dist)
                else:
                    a_prime_dist = a_dist
                ap = a_prime_dist.sample((K_c,)) # a' ~ \pi(s')
                assert ap.shape == (K_c, B, L, G, D_action)
                sp_ap_gp = (s_rep[:, 1:, ...].detach(), ap[:, :, 1:, ...].detach())
                q_targ_sp_ap_gp = self.target_critics(*sp_ap_gp) # Q(s', a')
                assert q_targ_sp_ap_gp.probs.shape == (K_c, B, L - 1, C, G, Bins)
                q_targ_sp_ap_gp = self.target_critics.bin_dist_to_raw_vals(q_targ_sp_ap_gp).mean(0)
                assert q_targ_sp_ap_gp.shape == (B, L - 1, C, G, 1)
                # y = r + gamma * (1.0 - d) * Q(s', a')
                ensemble_td_target = r + gamma * (1.0 - d) * q_targ_sp_ap_gp
                assert ensemble_td_target.shape == (B, L - 1, C, G, 1)
                td_target = self._critic_ensemble_to_td_target(ensemble_td_target)
                assert td_target.shape == (B, L - 1, 1, G, 1)
                self.popart.update_stats(
                    td_target, mask=critic_mask.all(2, keepdim=True)
                )
                assert td_target.shape == (B, L - 1, 1, G, 1)
                td_target_labels = self.target_critics.raw_vals_to_labels(td_target)
                td_target_labels = repeat(
                    td_target_labels, f"b l 1 g bins -> b l {C} g bins"
                )
                assert td_target_labels.shape == (B, L - 1, C, G, Bins)
                
            ###################
            ## Selector Loss ##
            ###################
            with torch.no_grad():
                # 1. Get the value distribution from the STABLE target critic for the current state (s) and actions from the buffer (a).
                q_s_a_g_stable = self.target_critics(s_rep, a_buffer.unsqueeze(0))

                # 2. Convert the distribution to raw scalar values.
                stable_target_values = self.target_critics.bin_dist_to_raw_vals(q_s_a_g_stable)
                assert stable_target_values.shape == (1, B, L, C, G, 1)

                # 3. Average over the ensemble of critics to get our final stable target.
                stable_target_values = stable_target_values.mean((0, 3)).squeeze() # Shape: [B, L, G]

            # 4. (Optional but HIGHLY Recommended) Normalize the stable target using PopArt.
            # Your main critic learns from normalized targets, so your VAE-critic should too!
            # Note: popart expects shape (B, L, C, G, 1), so we unsqueeze before normalizing.
            normalized_stable_target = self.popart.normalize_values(
                stable_target_values.unsqueeze(2).unsqueeze(-1)
            ).squeeze()

            # 5. Calculate the loss using this new, stable, and normalized target.
            vae_value_loss = F.mse_loss(normalized_stable_target, vae_state_values, reduction="none")
            vae_value_loss = reduce(
                                    vae_value_loss,
                                    'b l g -> (b l)', # Keep 'b' and 'l' (flattened), reduce 'g'
                                    'mean'            # The reduction operation to apply to 'g'
                                )
            vae_value_loss=self.calculate_masked_loss(loss_tensor=vae_value_loss, mask=state_mask)
            #################
            ## Critic Loss ##
            #################
            s_a_g = (s_rep, a_buffer.unsqueeze(0))
            q_s_a_g = self.critics(*s_a_g, log_dict=active_log_dict) # Q(s, a)
            
            # loss for vae_state_value estimator
            val_s = self.critics.bin_dist_to_raw_vals(q_s_a_g)
            assert val_s.shape == (1, B, L, C, G, 1)
            assert q_s_a_g.probs.shape == (1, B, L, C, G, Bins)
            # mean squared bellman error --> cross entropy w/ bin classification labels
            critic_loss = F.cross_entropy(
                rearrange(q_s_a_g.logits[0, :, :-1, ...], "b l c g u -> (b l c g) u"),
                rearrange(td_target_labels, "b l c g u -> (b l c g) u"),
                reduction="none",
            )
            critic_loss = rearrange(
                critic_loss, "(b l c g) -> b l c g 1", b=B, l=L - 1, c=C, g=G
            )
            assert critic_loss.shape == (B, L - 1, C, G, 1)
            scalar_q_s_a_g = self.critics.bin_dist_to_raw_vals(q_s_a_g).squeeze(0)
            if log_step:
                td_stats = self._td_stats(
                    critic_mask,
                    self.popart.normalize_values(scalar_q_s_a_g)[:, :-1, ...],
                    scalar_q_s_a_g[:, :-1, ...],
                    r=r,
                    d=d,
                    td_target=td_target,
                    raw_q_bins=q_s_a_g.probs[0, :, :-1],
                )
                popart_stats = self._popart_stats()
                self.update_info.update(td_stats | popart_stats)

        actor_loss = 0.0
        K_a = self.num_actions_for_value_in_actor_loss
        if self.offline_coeff > 0:
            #####################################################
            ## "Offline" (Advantage Weighted/Filtered BC) Loss ##
            #####################################################
            if not self.fake_filter:
                # f(A(s, a))
                with torch.no_grad():
                    a_agent = a_dist.sample((K_a,))
                    q_s_a_agent = self.critics(s_rep.detach(), a_agent)
                    assert q_s_a_agent.probs.shape == (K_a, B, L, C, G, Bins)
                    # mean over actions and critic ensemble
                    val_s = self.critics.bin_dist_to_raw_vals(q_s_a_agent)
                    assert val_s.shape == (K_a, B, L, C, G, 1)
                    # A(s, a) = Q(s, a) - V(s) = mean_over_critics(Q(s, a)) - mean_over_critics(mean_over_actions(Q(s, a ~ pi)))
                    advantage_s_a = scalar_q_s_a_g.mean(2) - val_s.mean((0, 3))
                    assert advantage_s_a.shape == (B, L, G, 1)
                    filter_ = self.fbc_filter_func(advantage_s_a)[:, :-1, ...].float()
                    binary_filter_ = binary_filter(advantage_s_a)[:, :-1, ...].float()
                    
            else:
                # Behavior Cloning (f(A(s, a)) = 1)
                filter_ = binary_filter_ = torch.ones(
                    (B, L - 1, G, 1), dtype=torch.float32, device=s_rep.device
                )
            # log pi(a | s)
            if self.discrete:
                logp_a = a_dist.log_prob(a_buffer).unsqueeze(-1)
            elif self.multibinary:
                logp_a = a_dist.log_prob(a_buffer).mean(-1, keepdim=True)
            else:
                logp_a = a_dist.log_prob(a_buffer).sum(-1, keepdim=True)
            # throw away last action that was a duplicate
            logp_a = logp_a[:, :-1, ...]
            actor_loss += self.offline_coeff * -(filter_.detach() * logp_a)
            if log_step:
                filter_stats = self._filter_stats(actor_mask, logp_a, filter_)
                self.update_info.update(filter_stats)

        if self.online_coeff > 0:
            #########################
            ## "Online" (DPG) Loss ##
            #########################
            # TODO: possible to recycle this q_val for the FBC loss above, as is done in Agent.
            # For now, only call rsample when specifically using online_coeff > 0 (since it's usually turned off)
            assert self.actor.actions_differentiable, "online-style actor loss is not compatible with action distribution"
            a_agent_dpg = torch.stack([a_dist.rsample() for _ in range(K_a)], dim=0)
            q_s_a_agent = self.maximized_critics(s_rep.detach(), a_agent_dpg)
            q_s_a_agent = self.popart.normalize_values(
                # mean over K actions, min over critic ensemble
                self.maximized_critics.bin_dist_to_raw_vals(q_s_a_agent).mean(0).min(2).values
            )
            actor_loss += self.online_coeff * -(q_s_a_agent[:, :-1, ...])
                
        return critic_loss, actor_loss, KL_loss, recons_loss, vae_value_loss


    def get_kl_loss(self, prior_mu: torch.Tensor, prior_logvar: torch.Tensor,
                    posterior_mu: torch.Tensor, posterior_logvar: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Kullback-Leibler (KL) divergence between two isotropic Gaussian
        distributions, batch-wise.

        The KL divergence loss is a regularization term used in VAEs to ensure the 
        posterior distribution (learned by the encoder) stays close to the prior distribution.

        Formula used (KL(N(posterior | prior))):
        KL = 0.5 * sum(
            (posterior_variance / prior_variance) 
            + ((posterior_mu - prior_mu)^2 / prior_variance)
            - 1 
            + (prior_logvar - posterior_logvar)
        )

        Args:
            prior_mu (torch.Tensor): Mean of the prior distribution. Shape (B, D).
            prior_logvar (torch.Tensor): Log-variance of the prior distribution. Shape (B, D).
            posterior_mu (torch.Tensor): Mean of the posterior distribution. Shape (B, D).
            posterior_logvar (torch.Tensor): Log-variance of the posterior distribution. Shape (B, D).

        Returns:
            torch.Tensor: The KL loss summed over the latent dimension (D), shape (B,).
                        The final loss for training is typically the mean of this output.
        """
        
        # 1. Calculate variances from log-variances
        prior_var = torch.exp(prior_logvar)
        posterior_var = torch.exp(posterior_logvar)

        # 2. Calculate the squared difference between the means
        mu_diff_sq = (posterior_mu - prior_mu).pow(2)

        # 3. Apply the generalized KL divergence formula
        # Term 1: Ratio of variances (tr(Sigma_0^-1 * Sigma_1))
        term1 = posterior_var / prior_var
        
        # Term 2: Squared mean difference scaled by prior variance (mu_diff^T * Sigma_0^-1 * mu_diff)
        term2 = mu_diff_sq / prior_var
        
        # Term 3 & 4: Log determinant difference and dimension subtraction
        term3_4 = prior_logvar - posterior_logvar - 1.0 # (log(det(Sigma_0)/det(Sigma_1)) - k)
        
        # Combine terms and multiply by 0.5
        kl_loss = 0.5 * (term1 + term2 + term3_4)
        
        # Sum over the latent dimension (D). This gives one KL value per item in the batch.
        kl_loss_batch_wise = torch.sum(kl_loss, dim=-1)

        return kl_loss_batch_wise

    def calculate_masked_loss(self, loss_tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Calculates the mean loss for a flattened loss tensor, guided by a mask.

        This function performs the following steps:
        1. Pads the mask to match the sequence length 'L'.
        2. Reshapes the mask to match the flattened loss tensor's shape.
        3. Applies the mask to the loss tensor, zeroing out ignored elements.
        4. Computes the mean of the loss, but only over the elements where the mask was 1.

        Args:
            loss_tensor (torch.Tensor): A 1D tensor of shape [B * L] containing per-item losses.
            mask (torch.Tensor): A 3D tensor of shape [B, L-1, 1] containing 0s and 1s.

        Returns:
            torch.Tensor: A 0-dimensional tensor (scalar) representing the final masked loss.
        """
        # --- Step 1: Get shapes and pad the mask ---
        # The mask is for transitions, so it has length L-1. We prepend a '1'
        # for the initial state, assuming its loss is always valid.
        B, L_minus_1, _ = mask.shape
        device = mask.device 
        mask = torch.cat([torch.ones((B, 1, 1), device=device), mask], dim=1)  # mask: [B,L-1,1] -> [B,L,1]
        mask = mask.squeeze(-1).flatten()   # [B, L, 1] -> [B, L] -> [B * L]
        masked_loss_elements = loss_tensor * mask
        total_masked_loss = masked_loss_elements.sum()
        num_active_elements = mask.sum()

        final_loss = total_masked_loss / (num_active_elements + 1e-8)
        return final_loss
    
    
    def forward_traj_encoder_KD(self, batch: Batch, log_step: bool, 
                                allow_actor_critic_adaptation:bool=True):
        # fmt: off
        self.update_info = {}  # holds wandb stats

        ##################
        ## Timestep Emb ##
        ##################
        active_log_dict = self.update_info if log_step else None
        with torch.no_grad():
            o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict)
        straight_from_obs = {k : batch.obs[k] for k in self.pass_obs_keys_to_actor}

        ###################
        ## Get Organized ##
        ###################
        B, L, D_o = o.shape
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1.0, 1.0)
        _B, _L, D_action = a.shape
        assert _L == L - 1
        G = len(self.gammas)
        K_c = self.num_actions_for_value_in_critic_loss
        a_buffer = F.pad(a, (0, 0, 0, 1), "replicate")
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")
        C = len(self.critics)
        assert batch.rews.shape == (B, L - 1, 1)
        assert batch.dones.shape == (B, L - 1, 1)
        r = repeat((self.reward_multiplier * batch.rews).float(), f"b l r -> b l 1 {G} r")
        d = repeat(batch.dones.float(), f"b l d -> b l 1 {G} d")
        gamma = self.gammas.to(r.device).unsqueeze(-1)
        D_emb = self.traj_encoder.emb_dim
        Bins = self.critics.num_bins
        state_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).float()[:, 1:, ...]
        actor_mask = F.pad(state_mask, (0, 0, 0, 1), "constant", 0.0)
        actor_mask = repeat(actor_mask, f"b l 1 -> b l {G} 1")
        critic_mask = repeat(state_mask, f"b l 1 -> b l {C} {G} 1")

        predicted_rep = {}
        ########################
        ## PREDICTIONS        ##
        ########################
        o_ = rearrange( 
                o, "b l (s d) -> b l s d", s=self.scratch_tokens
            )
        s_rep, _ = self.traj_encoder(seq=o_, time_idxs=batch.time_idxs, 
                                                hidden_state=None, log_dict=active_log_dict)
        predicted_rep['s_rep']= s_rep
        assert s_rep.shape == (B, L, D_emb)
        
        if allow_actor_critic_adaptation:

            ################
            ## a ~ \pi(s) ##
            ################
            a_dist = self.actor(s_rep, log_dict=active_log_dict, straight_from_obs=straight_from_obs)
            if self.discrete:
                a_dist = DiscreteLikeContinuous(a_dist)
            if log_step:
                policy_stats = self._policy_stats(actor_mask, a_dist)
                self.update_info.update(policy_stats)
                policy_entropy = a_dist.entropy().mean()
                self.update_info["actor/policy_entropy"] = policy_entropy
            
            predicted_rep['a_dist']= a_dist.logits
            #################
            ## Critic      ##
            #################
            s_a_g = (s_rep, a_buffer.unsqueeze(0))
            q_s_a_g = self.critics(*s_a_g, log_dict=active_log_dict) # Q(s, a)
            assert q_s_a_g.probs.shape == (1, B, L, C, G, Bins)
            predicted_rep['q_s_a_g']= q_s_a_g
        
        
        
        # get true seq_rep from teacher traj encoder
        assert hasattr(self, 'teacher_traj_encoder'), "Agent was not initialized with a teacher_traj_encoder for KD!"
        target_rep = {}
        with torch.no_grad(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            true_s_rep, _ = self.teacher_traj_encoder(seq=o, time_idxs=batch.time_idxs, 
                                                    hidden_state=None, log_dict=active_log_dict)
        
            if allow_actor_critic_adaptation:
                true_a_dist = self.teacher_actor(true_s_rep, log_dict=active_log_dict, straight_from_obs=straight_from_obs)
                if self.discrete:
                    true_a_dist = DiscreteLikeContinuous(true_a_dist)
                s_a_g = (true_s_rep, a_buffer.unsqueeze(0))
                true_q_s_a_g = self.teacher_critics(*s_a_g, log_dict=active_log_dict) # Q(s, a)
                assert true_q_s_a_g.probs.shape == (1, B, L, C, G, Bins)
                target_rep['s_rep']= true_s_rep
                target_rep['a_dist']= true_a_dist.logits
                target_rep['q_s_a_g']= true_q_s_a_g
            
        ################
        ## KD Loss    ##
        ################
        s_rep= self.kd_proj(s_rep)
        traj_recons_loss = F.mse_loss(s_rep, true_s_rep, reduction='none')
        traj_recons_loss = reduce(
                                traj_recons_loss,
                                'b l d -> (b l)', # Keep 'b' and 'l' (flattened), reduce 'd'
                                'mean'            # The reduction operation to apply to 'd'
                            )
        # calculate time-weighted masked loss, giving more weight to later timesteps
        time_weights = torch.linspace(1.0, 2.0, steps=L, device=traj_recons_loss.device)  # shape: [L]
        time_weights = time_weights.unsqueeze(0).expand(B, L)  # shape: [B, L]
        traj_recons_loss = traj_recons_loss * time_weights.flatten()  # shape: [B * L]
        traj_recons_loss=self.calculate_masked_loss(loss_tensor=traj_recons_loss, mask=state_mask)
        
        if allow_actor_critic_adaptation:
        
            pred_a_dist = F.log_softmax(predicted_rep['a_dist'], dim=-1)
            target_a_dist = F.log_softmax(target_rep['a_dist'], dim=-1)
            action_kl_loss= F.kl_div(pred_a_dist, target_a_dist, 
                                    reduction='none', log_target=True).sum(dim=-1)        
            squeezed_mask = actor_mask.squeeze(-1)
            action_kl_loss = action_kl_loss * squeezed_mask
            action_kl_loss = action_kl_loss.sum() / (squeezed_mask.sum() + 1e-5) # Add epsilon for stability
            
            
            # critic matching loss
            predicted_logits = predicted_rep['q_s_a_g'].logits[0, :, :-1, ...]
            target_logits = target_rep['q_s_a_g'].logits[0, :, :-1, ...]
            flat_predicted_logits = rearrange(predicted_logits, "b l c g u -> (b l c g) u")
            flat_target_logits = rearrange(target_logits, "b l c g u -> (b l c g) u")
            log_pred_dist = F.log_softmax(flat_predicted_logits, dim=-1)
            log_target_dist = F.log_softmax(flat_target_logits, dim=-1)
            critic_loss_elements = F.kl_div(log_pred_dist, log_target_dist, reduction='none', log_target=True).sum(dim=-1)
            
            critic_loss = rearrange(
                critic_loss_elements, "(b l c g) -> b l c g 1", b=B, l=L - 1, c=C, g=G
            )
            assert critic_loss.shape == (B, L - 1, C, G, 1)
            critic_mask = critic_mask.float()
            critic_loss = (critic_loss * critic_mask).sum() / (critic_mask.sum() + 1e-5)
        
            return traj_recons_loss, action_kl_loss, critic_loss
        
        return traj_recons_loss, 0, 0        


    def forward_vae_prior_training(self, batch: Batch, log_step: bool, save_dir:str):
        # fmt: off
        self.update_info = {}  # holds wandb stats

        ##################
        ## Timestep Emb ##
        ##################
        active_log_dict = self.update_info if log_step else None
        with torch.no_grad():
            o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict)

        ###################
        ## Get Organized ##
        ###################
        B, L, D_o = o.shape
        a = batch.actions
        a = a.clamp(0, 1.0) if self.discrete else a.clamp(-1.0, 1.0)
        _B, _L, D_action = a.shape
        assert _L == L - 1
        G = len(self.gammas)
        a_buffer = F.pad(a, (0, 0, 0, 1), "replicate")
        a_buffer = repeat(a_buffer, f"b l a -> b l {G} a")
        assert batch.rews.shape == (B, L - 1, 1)
        assert batch.dones.shape == (B, L - 1, 1)

        #################
        ## VAE Loss    ##
        #################
        o_ = rearrange( 
                o, "b l (s d) -> (b l) s d", s=self.scratch_tokens
            )
        # Create a mask to ignore padded parts of the sequence
        # This mask should have the same shape as your input 'o'.
        vae_mask = (~((batch.rl2s == self.pad_val).all(-1, keepdim=True))).float()
        vae_mask = rearrange( 
                vae_mask, "b l d -> (b l) d"
            )
        
        # 1. Unpack the VAE's output tuple
        metrics = self.vae_prior.train_step(o_=o_ , vae_mask=vae_mask, save_every_n_steps=200, save_dir=save_dir)

        if log_step:
            self.update_info.update(metrics)

        return  metrics
    
    
    def find_lookahead_extrema_index(self, tensor: torch.Tensor, H: int, find_highest: bool = True) -> torch.Tensor:
        """
        Finds the index of the highest or lowest value in a look-ahead window of size H 
        for each element in the sequence dimension of a (Batch, seq-len) tensor.

        Args:
            tensor: The input tensor of shape (Batch, seq-len).
            H: The look-ahead window size (number of steps to look ahead, excluding the current index).
            find_highest: If True, finds the index of the highest value (argmax); 
                        If False, finds the index of the lowest value (argmin).

        Returns:
            A tensor of shape (Batch, seq-len) containing the original sequence index 
            of the highest/lowest value within the look-ahead window.
        """
        B, L = tensor.shape
        
        # 1. Pad the tensor to handle the look-ahead window at the end of the sequence.
        # We pad with L-1 to ensure we have enough "slack" for the look-ahead window.
        # The padding value is set to a very small/large number depending on whether 
        # we are looking for max/min. This prevents the padded values from being 
        # selected as the extrema unless the look-ahead window extends beyond L.
        
        # Use a large negative number for argmax, and a large positive number for argmin.
        # This ensures that for padded regions, the original values are chosen.
        if find_highest:
            # PADDING_VALUE = -infinity (or a very small number)
            padding_value = torch.finfo(tensor.dtype).min if tensor.is_floating_point() else tensor.min() - 1 
        else:
            # PADDING_VALUE = +infinity (or a very large number)
            padding_value = torch.finfo(tensor.dtype).max if tensor.is_floating_point() else tensor.max() + 1
        
        # Pad H zeros at the end of the sequence dimension (dim=1).
        padded_tensor = torch.nn.functional.pad(tensor, (0, H), value=padding_value)
        
        # 2. Unfold the padded tensor to create the look-ahead windows.
        # The size of the window is H + 1 (current index + H look-ahead steps).
        window_size = H + 1
        # Step/stride is 1 to get a window for every index.
        unfolded_tensor = padded_tensor.unfold(dimension=1, size=window_size, step=1)
        # Shape of unfolded_tensor: (Batch, L, H + 1)
        
        # 3. Find the index of the maximum/minimum value within the window (last dimension).
        # Since we are excluding the current index (index 0 in the window), we slice
        # the window from index 1 to H+1.
        lookahead_window = unfolded_tensor[:, :, 1:] # Shape: (Batch, L, H)
        
        # Use argmax or argmin on the look-ahead window (dimension 2).
        if find_highest:
            # local_idx will be 0 to H-1, indicating position within the H steps
            local_idx = torch.argmax(lookahead_window, dim=2) 
        else:
            local_idx = torch.argmin(lookahead_window, dim=2)
            
        # Shape of local_idx: (Batch, L)
        
        # 4. Convert the local index to the global sequence index.
        # The local index (0 to H-1) is relative to the *start* of the look-ahead (which is curr_idx + 1).
        # Global index = curr_idx + 1 (start of look-ahead) + local_idx
        
        # Create a tensor of current sequence indices (0 to L-1).
        curr_idx = torch.arange(L, device=tensor.device).unsqueeze(0).expand(B, L)
        # Shape of curr_idx: (B, L)
        
        # Calculate the final global index.
        global_idx = curr_idx + 1 + local_idx
        
        # The global index needs to be capped at L-1 for the padded regions, 
        # but the logic of argmax/argmin with the padding value handles this naturally:
        # If the window extends beyond L, the padding value is chosen *unless* an 
        # actual value in the window is higher/lower. However, the requirement is 
        # simply to return the *index* of the extrema. By design, our padding logic 
        # ensures that if the window extends past the end of the sequence (L-1), the 
        # index returned will correspond to the **last valid element** of the sequence 
        # in the window, *unless* the actual extrema is an element outside the 
        # sequence (which is prevented by using min/max as padding).
        
        # Let's add a clip for safety, though for the padded case, the local_idx 
        # will already correspond to the index within the original tensor if any 
        # original value is chosen. If the padding is chosen, the resulting index 
        # might be > L-1, which we should cap to L-1 (or the largest index that was 
        # actually in the window).
        
        # The indices should not exceed L-1, as we're interested in the index of 
        # the highest/lowest *original* element.
        final_idx = torch.clamp(global_idx, max=L - 1)
        
        return final_idx
    
        
    def apply_lookahead_indices(
        self, 
        data_tensor: torch.Tensor, 
        idx_tensor: torch.Tensor
    ) -> torch.Tensor:
        """
        Applies the look-ahead sequence indices (from find_lookahead_extrema_index) 
        to a 4D tensor of shape [Batch, seq-len, scratch-tokens, dim].

        The output tensor will have the same shape as idx_tensor repeated across 
        the last two dimensions: [Batch, seq-len, scratch-tokens, dim].

        Args:
            data_tensor: The 4D input tensor [B, L, S, D].
            idx_tensor: The 2D index tensor [B, L] containing the target sequence 
                        index (0 to L-1) for each element.

        Returns:
            A 4D tensor of the same shape, where data_tensor[b, l, s, d] 
            is replaced by data_tensor[b, idx_tensor[b, l], s, d].
        """
        
        # 1. Expand the index tensor to match the shape of the data_tensor's last two dims.
        # The data tensor has shape [B, L, S, D].
        # The index tensor is [B, L]. We need it to be [B, L, 1, 1] for broadcasting.
        # The unsqueeze operation inserts dimensions of size 1 at the end.
        expanded_idx = idx_tensor.unsqueeze(-1).unsqueeze(-1)
        
        # Now expanded_idx shape is [B, L, 1, 1].
        # When used with take_along_dim, it will broadcast to [B, L, S, D].

        # 2. Use torch.take_along_dim to perform the selective indexing.
        # We are indexing along dimension 1 (seq-len).
        # For each element at (b, l, s, d) in the output, it will take the value 
        # from data_tensor at (b, expanded_idx[b, l, 0, 0], s, d).
        
        output_tensor = torch.take_along_dim(
            data_tensor, 
            expanded_idx, 
            dim=1
        )
        
        return output_tensor
    


    def _td_stats(
        self, mask, raw_q_s_a_g, q_s_a_g, r, d, td_target, raw_q_bins
    ) -> dict:
        stats = super()._td_stats(
            mask=mask,
            raw_q_s_a_g=raw_q_s_a_g,
            q_s_a_g=q_s_a_g,
            r=r,
            d=d,
            td_target=td_target,
            raw_q_bins=raw_q_bins,
        )

        # === ADD THIS CODE FOR TD ERROR LOGGING ===
        # raw_q_s_a_g has shape (B, L, C, G, 1). We average over the critic ensemble (C).
        mean_q_prediction = raw_q_s_a_g.mean(dim=2)

        # td_target has shape (B, L, 1, G, 1). We need to match shapes to subtract.
        # Squeeze the critic dimension from the target.
        td_error_abs = torch.abs(mean_q_prediction - td_target.squeeze(2))

        # The mask has shape (B, L, C, G, 1). Average it along the critic dim to match the error tensor.
        td_error_mask = mask.mean(2)
        
        stats["critic/TD_error_abs"] = (td_error_mask * td_error_abs).sum() / td_error_mask.sum()
        return stats