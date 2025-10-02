import gin
from amago.nets.utils import symlog
from amago.nets.actor_critic import NCriticsTwoHot, gammas_as_input_seq
from metamon.rl.metamon_to_amago import MetamonTstepEncoder, unknown_token_mask
import torch
import torch.nn.functional as F
from torch import distributions as pyd
from einops import repeat, rearrange
from einops.layers.torch import EinMix as Mix
from typing import Optional, Type

@gin.configurable
class NoCompileMetamonTstepEncoder(MetamonTstepEncoder):
    def inner_forward(self, obs, rl2s, log_dict=None):
        if self.training and self.token_mask_aug:
            obs["text_tokens"] = unknown_token_mask(obs["text_tokens"])
        extras = F.leaky_relu(self.extra_emb(symlog(rl2s)))
        numerical = torch.cat((obs["numbers"], extras), dim=-1)
        turn_emb = self.turn_embedding(
            token_inputs=obs["text_tokens"], numerical_inputs=numerical
        )
        return turn_emb

@gin.configurable
class NoCompileNCriticsTwoHot(NCriticsTwoHot):    
    def critic_network_forward(
        self, state: torch.Tensor, action: torch.Tensor, log_dict: Optional[dict] = None
    ) -> pyd.Categorical:
        """Compute a categorical distribution over bins from a state and action.

        Args:
            state: The "state" sequence (the output of the TrajEncoder). Has shape
                (Batch, Length, state_dim).
            action: The action sequence. Has shape (K, Batch, Length, Gammas, action_dim),
                where K is a dimension denoting multiple action samples from the same state
                (can be 1, but must exist). Discrete actions are expected to be one-hot vectors.

        Returns:
            The categorical distribution over bins with shape (K, Batch, Length, num_critics, output_bins).
        """
        K, B, L, G, D = action.shape
        assert G == self.num_gammas
        state = repeat(state, "b l d -> (k b g) l d", k=K, g=self.num_gammas)
        action = rearrange(action.clamp(-0.999, 0.999), "k b l g d -> (k b g) l d")
        gammas_rep = gammas_as_input_seq(self.gammas, K * B, L).to(action.device)
        inp = torch.cat((state, gammas_rep, action), dim=-1)
        outputs, phis = self.net(inp)
        outputs = rearrange(
            outputs, "(k b g) l c o -> k b l c g o", k=K, g=self.num_gammas
        )
        val_dist = pyd.Categorical(logits=outputs)
        clip_probs = val_dist.probs.clamp(1e-6, 0.999)
        safe_probs = clip_probs / clip_probs.sum(-1, keepdims=True).detach()
        safe_dist = pyd.Categorical(probs=safe_probs)
        return safe_dist
