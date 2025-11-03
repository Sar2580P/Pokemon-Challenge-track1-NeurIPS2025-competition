from amago.agent import binary_filter, Multigammas
import gin
import torch 
from typing import Type, Optional, Tuple, Any, List, Iterable, Dict
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
from metamon.rl.metamon_to_amago import MetamonTstepEncoder
import amago.nets.actor_critic as actor_critic
import gymnasium as gym
from amago.agent import MultiTaskAgent
from amago.nets.traj_encoders import TformerTrajEncoder
from metamon import  METAMON_CACHE_DIR
from typing import Type, Optional, Tuple, Any, List, Iterable
import torch.distributions as pyd
from einops import rearrange, repeat
from einops import repeat, rearrange
import gin
import gymnasium as gym
from torch.distributions import Distribution

from amago.loading import Batch, MAGIC_PAD_VAL
from amago.nets import actor_critic
import torch
import torch.nn.functional as F
from einops import rearrange
import numpy as np

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

    
torch._dynamo.disable()


@gin.configurable
class InferenceEntityMTA(MultiTaskAgent):
    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        tstep_encoder_type: Type[MetamonTstepEncoder],
        traj_encoder_type: Type[TformerTrajEncoder],
        max_seq_len: int,
        num_critics: int = 4,
        num_critics_td: int = 2,
        online_coeff: float = 0.0,
        offline_coeff: float = 1.0,
        gamma: float = 0.999,
        reward_multiplier: float = 10.0,
        tau: float = 0.003,
        use_metamon_models:bool=True, 
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
        parallel_actors: int=12,      #🛕🐚
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

        # full weight copy to targets
        self.hard_sync_targets()
        self.pass_obs_keys_to_actor = pass_obs_keys_to_actor or []
        
        # to hold the state-rep, action
        self.state_rep = None
        self.action_dists=None
        self.hidden_state = None
        
        self.parallel_actors=parallel_actors
        self.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        self.prev_state_value = None

    def get_refined_policy_distribution(
        self,
        T:float=1,
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
        with torch.inference_mode():
            # --- Preliminary Step: Get Full Critic Evaluation & Actor Probabilities ---
            q_dist = get_q_dist_for_all_actions(
                critic=self.critics,
                state_rep=self.state_rep ,
                num_actions=self.action_dim,
                num_gammas=len(self.gammas)
            )
            q_values_per_action = self.critics.bin_dist_to_raw_vals(q_dist)
            q_values_stable = q_values_per_action.mean(dim=-3).squeeze(-1) # Shape: [B, L, A, G]
            actor_probs = self.action_dists.probs # Shape: [B, L, G, A]
                
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
            
        return final_refined_probs_per_gamma, advantage_values_per_gamma

    def get_actions_dists(
        self,
        obs: Dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        hidden_state: Any = None
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

        traj_emb_t, hidden_state = self.traj_encoder.forward(tstep_emb_, time_idxs=time_idxs, 
                                                          hidden_state=hidden_state)

        # Step 2: Get the raw action distributions from the actor
        action_dists = self.actor(
            traj_emb_t,
            straight_from_obs={k: obs[k] for k in self.pass_obs_keys_to_actor},
        )

        self.state_rep=traj_emb_t
        self.action_dists = action_dists
        return action_dists, hidden_state
        
       
    
    def get_td_err(self, rewards: torch.Tensor, dones:np.ndarray=None):
        # TD err = [r + γ*V(s_t+1)] - V(s_t)
        # TD target = r + γ*V(s_t+1)
        
        # Get the value of the *new* state (s_t+1)
        # This function returns shape [B, L, G']
        curr_V_s = self.get_state_value() 
        
        dev=curr_V_s.device
        B, L, G_prime = curr_V_s.shape
        assert L == 1, "Inference loop must have L=1"
        if dones is None: dones= np.array([False]*B)

        if self.prev_state_value is None:
            self.prev_state_value = curr_V_s
            # so no expert weights are changed on the first step.
            return torch.zeros_like(curr_V_s).to(dev)
        
        prev_V_s = self.prev_state_value  # Shape: [B, L, G']
        r = torch.from_numpy(rewards).unsqueeze(-1).to(dev)   # [B, L]  --> [B, L, 1] for broadcasting
        g = self.gammas.to(dev).view(1, 1, -1)  # [G'] --> [1, 1, G']
        
        # Create the done mask. Shape: [B, 1, 1]
        done_mask = torch.from_numpy(dones).to(dev, dtype=torch.float32).view(B, 1, 1)

        # --- 3. Calculate TD Target (Respecting `dones`) ---
        # The (1.0 - done_mask) ensures V(s_t+1) is zeroed out if
        # the episode just terminated. This is the correct TD-Target.
        # td_target = r_t + g * (1.0 - done_t) * V(s_t+1)
        td_target = r + g * (1.0 - done_mask) * curr_V_s
        
        # Calculate the "real" TD Error
        td_error = td_target - prev_V_s  # Shape: [B, L, G']

        # --- 4. Store the state value for the next loop ---
        # This is the critical state update.
        # If the game is done, the *next* state's "previous value" should be 0.
        self.prev_state_value = curr_V_s 
        
        # --- 5. Apply your heuristic ---
        # Now, return the final error, masking it to 0 if the game was done.
        # This prevents the terminal reward from affecting the Hedge weights.
        return td_error * (1.0 - done_mask)
    
        
    def get_state_value(
        self,
    ) -> torch.Tensor:
        """
        Calculates the "real" (unnormalized) state-value V(s) for a specific expert.
        
        V(s) is computed as the exact expected Q-value: V(s) = E[Q(s,a)] = sum(pi(a|s) * Q(s,a))

        Args:
            s_rep: The state representation from the TrajEncoder.
                Shape: [B, L, D_emb]
            action_probs: The actor's probability distribution over actions.
                        Shape: [B, L, G, A]
            expert_critic: The specific critic module for this expert.
            expert_popart: The specific PopArt module for this expert.

        Returns:
            v_state_real: The real, unnormalized state-value V(s).
                        Shape: [B, L, G]
        """
        action_probs = self.action_dists.probs   # shape: [B, L, G, A]
        s_rep=self.state_rep
        expert_critic=self.critics
        B, L, G, A = action_probs.shape
        device = s_rep.device
        
        # --- 1. Get Q(s,a) for ALL actions ---
        
        # All 9 possible actions (as one-hot vectors)
        action_identity = torch.eye(A, device=device)
        
        # We need to feed this to the critic. The critic expects actions in shape
        # [K_c, B, L, G, D_action]. Here, K_c = A = 9 and D_action = A = 9.
        
        # Reshape identity to [A, 1, 1, 1, A]
        all_actions_tensor = action_identity.view(A, 1, 1, 1, A)
        all_actions_tensor = all_actions_tensor.repeat(1, B, L, G, 1)   # Repeat to match batch, length, and gamma dimensions

        # Call the critic ONCE. It will compute Q(s,a) for all 9 actions
        # in parallel. `s_rep` is broadcasted against the first dim of actions.
        # q_all_actions_dist.probs shape: [A, B, L, C, G, Bins]
        q_all_actions_dist = expert_critic(s_rep, all_actions_tensor)
        
        # a) Convert from distributional bins to a normalized scalar
        q_all_actions = expert_critic.bin_dist_to_raw_vals(q_all_actions_dist)   # Shape: [A, B, L, C, G, 1]

        # --- 3. Calculate V(s) ---
        
        # a) Average over the critic ensemble (C dimension)
        # Shape: [A, B, L, G, 1]
        q_all_actions = q_all_actions.mean(dim=3) 
        
        # b) Squeeze and permute to match action_probs shape
        # Shape: [B, L, G, A]
        q_all_actions = q_all_actions.squeeze(-1).permute(1, 2, 3, 0)
        
        # c) Calculate the expectation: V(s) = sum( pi(a|s) * Q(s,a) )
        # This is the weighted sum you proposed.
        # ( [B,L,G,A] * [B,L,G,A] ).sum(dim=-1)
        v_state_real = (action_probs * q_all_actions).sum(dim=-1)
        
        # Final shape: [B, L, G]
        return v_state_real
            
        
@gin.configurable
class InferenceTribeMTA(MultiTaskAgent):
    def __init__(
        self,
        obs_space: gym.spaces.Dict,
        rl2_space: gym.spaces.Box,
        action_space: gym.spaces.Space,
        tstep_encoder_type: Type[MetamonTstepEncoder],
        traj_encoder_type: Type[TformerTrajEncoder],
        
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
        num_actions_for_value_in_critic_loss: int = 1,
        num_actions_for_value_in_actor_loss: int = 3,
        fbc_filter_func: callable = binary_filter,
        popart: bool = True,
        use_target_actor: bool = True,
        use_multigamma: bool = True,
        actor_type: Type[actor_critic.BaseActorHead] = actor_critic.Actor,
        critic_type: Type[actor_critic.BaseCriticHead] = actor_critic.NCriticsTwoHot,
        pass_obs_keys_to_actor: Optional[Iterable[str]] = None,
        inference_entities: List[InferenceEntityMTA]=None,   #🛕🐚
        
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
        self.use_metamon_models=use_metamon_models
        
        # self.tstep_encoder = tstep_encoder_type(
        #     obs_space=obs_space,
        #     rl2_space=rl2_space,
        # )
        # self.scratch_tokens = self.tstep_encoder.scratch_tokens 

        # self.traj_encoder = traj_encoder_type(
        #     tstep_dim=self.tstep_encoder.d_model if not self.use_metamon_models else self.tstep_encoder.d_model*self.scratch_tokens,
        #     max_seq_len=max_seq_len,
        # )
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

        # self.popart = actor_critic.PopArtLayer(gammas=len(gammas), enabled=popart)

        # ac_kwargs = {
        #     "state_dim": self.traj_encoder.emb_dim ,
        #     "action_dim": self.action_dim,
        #     "discrete": self.discrete,
        #     "gammas": self.gammas,
        # }
        # self.critics = critic_type(**ac_kwargs, num_critics=num_critics)
        # self.target_critics = critic_type(**ac_kwargs, num_critics=num_critics)
        # self.maximized_critics = critic_type(**ac_kwargs, num_critics=num_critics)
        # self.actor = actor_type(**ac_kwargs)
        # self.target_actor = actor_type(**ac_kwargs)

        # full weight copy to targets
        self.hard_sync_targets()
        self.pass_obs_keys_to_actor = pass_obs_keys_to_actor or []
    
        self.inference_entities=inference_entities
        self.policy_weights = None
        self.per_expert_G_t = None
        self.eta_base=0.7

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

    def ensemble_policy_population(self, actor_probs_per_gamma: torch.Tensor=None, 
                                   advantage_values_per_gamma:torch.Tensor=None, 
                                   final_ensemble_method: str = 'dynamic_advantage', 
                                   entropy_fallback_threshold:float=1):
        
        if final_ensemble_method == 'entropy':
            probs = actor_probs_per_gamma + 1e-9
            entropy = -torch.sum(probs * torch.log2(probs), dim=-1)
            
            all_uncertain = torch.all(entropy > entropy_fallback_threshold, dim=-1, keepdim=True)
            default_probs = probs[..., -1, :]
            
            confidence_weights = F.softmax(-entropy, dim=-1).unsqueeze(-1)
            ensembled_probs = torch.sum(probs * confidence_weights, dim=-2)
            
            return torch.where(all_uncertain, default_probs, ensembled_probs)
        
        elif final_ensemble_method == 'mean':
            # a) Simple, robust mean of the refined policies
            final_probs = torch.mean(actor_probs_per_gamma, dim=-2)
            
        elif final_ensemble_method == 'dynamic_advantage':
            # b) Weighted ensemble based on the "optimism" of each refined policy
            # We calculate the expected advantage of each *refined* policy
            expected_advantage_of_refined = torch.sum(
                actor_probs_per_gamma * advantage_values_per_gamma, dim=-1
            )
            # Use softmax on these scores to get the final weights
            final_weights = F.softmax(expected_advantage_of_refined, dim=-1).unsqueeze(-1)
            # Apply weights to get the final ensembled distribution
            final_probs = torch.sum(actor_probs_per_gamma * final_weights, dim=-2)
            
        elif final_ensemble_method == 'dynamic_entropy':
            # b) Weighted ensemble based on the "confidence" of each refined policy
            # Calculate the entropy of each refined distribution
            entropy = -torch.sum(
                actor_probs_per_gamma * torch.log2(actor_probs_per_gamma + 1e-9), dim=-1
            )
            # Low entropy (high confidence) gets a higher weight
            final_weights = F.softmax(-entropy, dim=-1).unsqueeze(-1)
            final_probs = torch.sum(actor_probs_per_gamma * final_weights, dim=-2)
        
        elif final_ensemble_method in ('none' , 'default'):
            return actor_probs_per_gamma[..., -1, :]
        
        else: # Default to simple mean for safety
            final_probs = torch.mean(actor_probs_per_gamma, dim=-2)

        return final_probs

    def get_actions_v1(
        self,
        obs: Dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        sample: bool = True,
        hidden_state: Any = None, 
        ensembling_method: str = 'dynamic_advantage', # Options: 'entropy', 'mean', 'none', 'q_value'
        sampling_method: str = 'nucleus',   # Options: 'nucleus', 'standard'
        nucleus_p: float = 0.9,
        entropy_fallback_threshold: float = 1.5,
    ) -> Tuple[torch.Tensor, Any]:
        """
        Get rollout actions from the current policy with advanced sampling and ensembling.
        """
        T=3 # temperature
        hidden_state_=[]
        for i, agent in enumerate(self.inference_entities):
            a,h = agent.get_actions_dists(obs=obs, rl2s=rl2s, 
                                           time_idxs=time_idxs, 
                                           hidden_state=hidden_state[i])
            hidden_state_.append(h)

        refined_action_probs_per_gamma_=[]
        advantage_values_per_gamma_=[]
        for agent in self.inference_entities:
            refined_action , adv_value= agent.get_refined_policy_distribution(T=T)
            refined_action_probs_per_gamma_.append(refined_action)
            advantage_values_per_gamma_.append(adv_value)
        
        # self.log_actor_diagnostics(action_dists, time_idxs)
        # Step 3: Use a helper function to ensemble the gamma policies
        actor_probs_per_gamma= torch.cat(refined_action_probs_per_gamma_, dim=-2)
        advantage_values_per_gamma= torch.cat(advantage_values_per_gamma_, dim=-2)
        final_probs = self.ensemble_policy_population(
            final_ensemble_method=ensembling_method,
            entropy_fallback_threshold=entropy_fallback_threshold, 
            actor_probs_per_gamma=actor_probs_per_gamma,
            advantage_values_per_gamma=advantage_values_per_gamma,
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
        return actions.to(dtype=dtype) , hidden_state_
        
        
    def get_policy_weights(self, batch_size:int):
        if self.policy_weights is None:
            assert self.inference_entities is not None
            N= len(self.inference_entities)
            G=len(self.gammas)
            K= 1/(N*G)
            self.policy_weights = torch.tensor([[K]*(N*G)]*batch_size)
            assert self.policy_weights.shape == (batch_size, N*G)
        return self.policy_weights
    
    
    def get_actions(
        self,
        obs: Dict[str, torch.Tensor],
        rl2s: torch.Tensor,
        time_idxs: torch.Tensor,
        rewards:torch.Tensor=None,
        sample: bool = True,
        hidden_state: Any = None, 
        sampling_method: str = 'nucleus',   # Options: 'nucleus', 'standard'
        nucleus_p: float = 0.7,
        dones=None, 
    ) -> Tuple[torch.Tensor, Any]:
        
        lr_method="rms_style_per_policy_lr"
        all_gammas= self.gammas.tolist()*len(self.inference_entities)
        # STEP-1: Get the action distribution from each actor
        actions, hidden_state_=[], []
        for i, agent in enumerate(self.inference_entities):
            a,h = agent.get_actions_dists(obs=obs, rl2s=rl2s, 
                                           time_idxs=time_idxs, 
                                           hidden_state=hidden_state[i])
            hidden_state_.append(h)
            actions.append(a.probs)   # action_probs shape: [B, Seq(=1), G(gammas), A(actions)]
        
        
        # STEP-2: Find ensemble policy based on weights
        actions = torch.cat(actions, dim=2)    
        
        print(f"BEFORE: action distr: max: {torch.max(actions[0], dim=-1)[0]} ||| min: {torch.min(actions[0], dim=-1)[0]} ")
        weights = self.get_policy_weights(batch_size=actions.shape[0]).to(actions.device)
        # weights = weights/torch.sum(weights+ 1e-8, dim=1, keepdim=True)   # shape: [B, #Agents * action_dim]

        
        # --- 1. Define K (number of top policies to use) ---
        gamma_mask = torch.tensor(all_gammas, device=weights.device, dtype=weights.dtype)
        # 2. Create the boolean mask (True where gamma >= 0.1), shape: [G']
        gamma_mask = (gamma_mask >= 0.1) 
        # 3. Apply the mask to the weights. Broadcasting [B, G'] * [G']
        weights = weights * gamma_mask
        K = 4
        
        # 'weights' is your original policy_weights tensor of shape [B, G']
        
        # --- 2. Find the top K weights and their indices ---
        # We do this along the last dimension (the experts dim)
        top_k_values, top_k_indices = torch.topk(weights, K, dim=-1)
        
        # --- 3. Create a new tensor of zeros ---
        # This will hold only the top-k weights
        top_k_weights = torch.zeros_like(weights)
        
        # --- 4. "Scatter" the top-k values back into the new tensor ---
        # This places the top values back at their original indices
        top_k_weights.scatter_(-1, top_k_indices, top_k_values)
        
        # --- 5. [CRITICAL] Re-normalize the new Top-K weights ---
        # This ensures the new sparse weights still sum to 1,
        # otherwise the resulting policy is not a valid probability distribution.
        sum_of_top_k = top_k_weights.sum(dim=-1, keepdim=True)
        normalized_top_k_weights = top_k_weights / (sum_of_top_k + 1e-8)
        
        # --- 6. Action sampling from the new Top-K ensembled policy ---
        # We use your original einsum, but with the new normalized_top_k_weights
        ensemble_policy = torch.einsum('biga, bg -> bia', actions, normalized_top_k_weights)

        # # STEP-3: Action sampling from ensembled policy
        # ensemble_policy = torch.einsum('biga, bg -> bia', actions, weights)
        print(f"AFTER: action distr: max: {torch.max(ensemble_policy[0], dim=-1)[0]} ||| min: {torch.min(ensemble_policy[0], dim=-1)[0]} \n\n\n")
        sampled_actions = self._select_action_from_probs(
            ensemble_policy,
            sample=sample,
            method=sampling_method if self.discrete else 'mean', # Fallback for continuous
            nucleus_p=nucleus_p
        ).unsqueeze(-1)
        dtype = torch.uint8 if (self.discrete or self.multibinary) else torch.float32
        
        # STEP-4: update weights
        self.update_policy_weights(
            rewards=rewards, 
            lr_method=lr_method, 
            dones=dones
        )
        return sampled_actions.to(dtype=dtype), hidden_state_
    
    
    def update_policy_weights(self, rewards:torch.Tensor, dones, 
                              lr_method='rms_style_per_policy_lr'):
        # STEP-1: Calculate the TD error for each entity
        # G' = G*(# agents)
        td_err_=[]
        popart_sigmas_=[]
        for _, agent in enumerate(self.inference_entities):
            tde = agent.get_td_err(rewards=rewards, dones=dones)
            td_err_.append(tde)
            popart_sigmas_.append(agent.popart.sigma)
            
        sigmas = torch.cat(popart_sigmas_, dim=0)    # [G', 1]
        td_error = torch.cat(td_err_, dim=-1) # [B, L, G' ]
        print(f"TD ERROR ==> {td_error[0]}\n\n")
        assert td_error.shape[1]==1, f"The seq len should be L=1, but found (L={td_error.shape[1]})"
        
        # We need to divide [B, L, G'] by [G', 1].
        # We reshape sigma to [1, 1, G'] to match the last dimension.
        sigmas_reshaped = sigmas.view(1, 1, -1)
        
        # --- 2. Calculate Standardized z-Error ---
        # We add a small epsilon for numerical stability.
        epsilon = 1e-6 
        z_error = td_error / (sigmas_reshaped + epsilon)
        
        # --- 3. Calculate Final Squashed Loss ---
        # L_i(t) = -tanh(z_error). This is bounded in [-1, 1].
        loss = -torch.tanh(z_error)
        print(f"loss ==> min: {torch.min(loss[0], dim=-1)[0]} ||| max: {torch.max(loss[0], dim=-1)[0]} ||| mean: {torch.mean(loss[0], dim=-1)}")
        lr = self.get_hedge_lr(     # [B, G']
            loss=loss, 
            method=lr_method)  
        lr= torch.clamp(lr, max=3)
        update_factor = torch.exp(-lr * loss.squeeze(1))
        # --- 4. Update the Weights ---
        # w_i(t+1) = w_i(t) * update_factor
        self.policy_weights = self.policy_weights.to(update_factor.device) * update_factor
        self.policy_weights = torch.clamp(self.policy_weights, max=5)
        print(f"policy-weights : min: {torch.mean(torch.min(self.policy_weights, dim=1)[0])} ||| max: {torch.mean(torch.max(self.policy_weights, dim=1)[0])}")
    
    def get_hedge_lr(self,
                     loss: torch.Tensor,  
                     method='rms_style_per_policy_lr',
                     rms_beta: float = 0.67
                     ):
        if method == "rms_style_per_policy_lr":
            # loss shape is [B, L, G'], where L=1
            B, L, G = loss.shape            
            loss_sq = loss.squeeze(1)
            
            G_t1 = self.per_expert_G_t if self.per_expert_G_t is not None \
                   else torch.zeros(size=(B, G), device=loss.device)
            G_t1 = G_t1.to(loss.device)

            # G_t = beta * G_{t-1} + (1 - beta) * L_t^2
            G_t2 = rms_beta * G_t1 + (loss_sq ** 2)
            self.per_expert_G_t = G_t2
            return self.eta_base / (torch.sqrt(G_t2) + 1e-8)
        else:
            raise NotImplementedError(f"Hedge LR method '{method}' not implemented.")
        
        
    def reset_hedge_state(self, dones: torch.Tensor):
        # dones: [B,]  boolean tensor
        '''
        things to reset ==>
        1) the policy weights
        2) G
        '''
        if not np.any(dones): return

        # 1. Reset self.policy_weights
        # We reset the weights for the 'done' environments back to a 
        # uniform distribution.
        if self.policy_weights is not None:
            G_prime = self.policy_weights.shape[1]
            K = 1.0 / G_prime
            self.policy_weights[dones] = K
        
        # 2. Reset self.per_expert_G_t
        if self.per_expert_G_t is not None:
            self.per_expert_G_t[dones] = 0.0
            
