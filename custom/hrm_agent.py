from amago.agent import binary_filter, Multigammas
import gin
import torch 
import itertools
from amago.loading import Batch, MAGIC_PAD_VAL
from typing import Type, Optional, Tuple, Any, List, Iterable
import torch.nn.functional as F
from einops import repeat, rearrange
from metamon.rl.metamon_to_amago import MetamonTstepEncoder
import amago.nets.actor_critic as actor_critic
from custom.traj_encoder import HRMTrajEncoder
import gymnasium as gym
import torch.nn as nn
import enum
import os
from amago.agent import MultiTaskAgent
from metamon import  METAMON_CACHE_DIR
import itertools
from typing import Type, Optional, Tuple, Any, List, Iterable

import torch
from torch import nn
import torch.nn.functional as F
from einops import repeat, rearrange
import numpy as np
import wandb
import gin
import gymnasium as gym

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
        fake_filter: bool = False,
        num_actions_for_value_in_critic_loss: int = 1,
        num_actions_for_value_in_actor_loss: int = 3,
        fbc_filter_func: callable = binary_filter,
        popart: bool = True,
        use_target_actor: bool = True,
        use_multigamma: bool = True,
        actor_type: Type[actor_critic.BaseActorHead] = actor_critic.Actor,
        critic_type: Type[actor_critic.BaseCriticHead] = actor_critic.NCriticsTwoHot,
        pass_obs_keys_to_actor: Optional[Iterable[str]] = None,
        # below args for ckpt initialization
        tstep_component_type: Type[InitComponent] = InitComponent,
        traj_component_type: Type[InitComponent] = InitComponent, 
        actor_component_type: Type[InitComponent] = InitComponent, 
        critic_component_type: Type[InitComponent]= InitComponent, 
        maximized_critics_component_type: Type[InitComponent]= InitComponent, 
        target_critic_component_type: Type[InitComponent]= InitComponent, 
        target_actor_component_type: Type[InitComponent]= InitComponent, 
        popart_component_type: Type[InitComponent]= InitComponent, 

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

        self.tstep_encoder = tstep_encoder_type(
            obs_space=obs_space,
            rl2_space=rl2_space,
        )
        self.traj_encoder = traj_encoder_type(
            tstep_dim=self.tstep_encoder.d_model,
            max_seq_len=max_seq_len,
        )
        self.emb_dim = self.traj_encoder.emb_dim
        self.scratch_tokens = self.tstep_encoder.scratch_tokens 

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
            "state_dim": self.traj_encoder.emb_dim,
            "action_dim": self.action_dim,
            "discrete": self.discrete,
            "gammas": self.gammas,
        }
        self.critics = critic_type(**ac_kwargs, num_critics=num_critics)
        self.target_critics = critic_type(**ac_kwargs, num_critics=num_critics)
        self.maximized_critics = critic_type(**ac_kwargs, num_critics=num_critics)
        self.actor = actor_type(**ac_kwargs)
        self.target_actor = actor_type(**ac_kwargs)
        
        # initializing component from ckpts
        self.components = [tstep_component_type(), traj_component_type(), 
                           actor_component_type(), critic_component_type(),
                           target_critic_component_type(), target_actor_component_type(), 
                           popart_component_type() , maximized_critics_component_type()
                           ]
        
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
            state_dict={k.replace(f"{state_dict_prefix}.", '', 1):v for k,v in ckpt.items() if k.startswith(state_dict_prefix)}
            component.load_state_dict(state_dict)
            print(f"✅ SUCCESS: {component_name} initialized from ckpt ({ckpt_path})")
        except Exception as e: 
            print(f"😥 FAILED: {component_name} initialization from ckpt ({ckpt_path}).\nERROR: {e}")
    
    def freeze_component(self, component:nn.Module):
        for param in component.parameters():
            param.requires_grad = False
        
    def get_actions(
        self,
        obs: dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state=None,
        sample: bool = True,
    ) -> Tuple[torch.Tensor, Any]:
        """Get rollout actions from the current policy.

        Note the standard torch `forward` implements the training step, while `get_actions`
        is the inference step. Most of the arguments here are easily gathered from the
        AMAGOEnv gymnasium wrapper. See `amago.experiment.Experiment.interact` for an example.

        Args:
            obs: Dictionary of (batched) observation tensors. AMAGOEnv makes all
                observations into dicts.
            rl2s: Batched Tensor of previous action and reward. AMAGOEnv makes these.
            time_idxs: Batched Tensor indicating the global timestep of the episode.
                Mainly used for position embeddings when the sequence length is much shorter
                than the episode length.
            hidden_state: Hidden state of the TrajEncoder. Defaults to None.
            sample: Whether to sample from the action distribution or take the argmax
                (discrete) or mean (continuous). Defaults to True.

        Returns:
            tuple:
                - Batched Tensor of actions to take in each parallel env *for the primary
                  ("test-time") discount factor* `Agent.gamma`.
                - Updated hidden state of the TrajEncoder.
        """
        
        tstep_emb = self.tstep_encoder(obs=obs, rl2s=rl2s)
        # sequence model embedding [batch, length, d_emb]
        tstep_emb_ = rearrange( 
                tstep_emb, "b l (s d) -> b l s d", s=self.tstep_encoder.scratch_tokens
            )
        traj_emb_t, hidden_state = self.traj_encoder(
            tstep_emb_, time_idxs=time_idxs, hidden_state=hidden_state
        )
        # generate action distribution [batch, length, len(self.gammas), d_action]
        action_dists = self.actor(
            traj_emb_t,
            straight_from_obs={k: obs[k] for k in self.pass_obs_keys_to_actor},
        )
        if sample: actions = action_dists.sample()
        else:
            if self.discrete: actions = torch.argmax(action_dists.probs, dim=-1, keepdim=True)
            else: actions = action_dists.mean
        # get intended gamma distribution (always in -1 idx)
        actions = actions[..., -1, :]
        dtype = torch.uint8 if (self.discrete or self.multibinary) else torch.float32
        return actions.to(dtype=dtype), hidden_state

    
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
        

        return critic_loss, actor_loss

    