from metamon.rl.metamon_to_amago import MetamonTstepEncoder
import gin
from metamon.tokenizer import PokemonTokenizer, UNKNOWN_TOKEN


@gin.configurable
class CustomTstepEncoder(MetamonTstepEncoder):
    def __init__(self,       
                obs_space,
                rl2_space,
                tokenizer: PokemonTokenizer,
                extra_emb_dim: int = 18,
                d_model: int = 100,
                n_layers: int = 3,
                n_heads: int = 5,
                scratch_tokens: int = 4,
                numerical_tokens: int = 6,
                token_mask_aug: bool = False,
                dropout: float = 0.05,):
        super().__init__(
            obs_space=obs_space,
            rl2_space=rl2_space,
            tokenizer=tokenizer,
            extra_emb_dim=extra_emb_dim,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            scratch_tokens=scratch_tokens,
            numerical_tokens=numerical_tokens,
            token_mask_aug=token_mask_aug,
            dropout=dropout,
        )
        self.scratch_tokens= scratch_tokens