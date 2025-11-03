import os
from pathlib import Path
import warnings
import json
import collections
import functools
from typing import Optional, Dict, Any, Callable, List
import metamon
from metamon.rl.pretrained import (
    get_pretrained_model,
    get_pretrained_model_names,
    PretrainedModel,
)
from metamon.baselines import get_baseline
from metamon.rl.metamon_to_amago import (
    make_baseline_env,
    make_local_ladder_env,
    make_pokeagent_ladder_env,
)

warnings.filterwarnings("ignore")


def red_warning(msg: str):
    print(f"\033[91m{msg}\033[0m")

import amago
import metamon
from metamon.rl.metamon_to_amago import (
    make_placeholder_env,
)
from metamon.interface import (
    ObservationSpace,
    RewardFunction,
    DefaultObservationSpace,
    TeamPreviewObservationSpace,
    ExpandedObservationSpace,
    TokenizedObservationSpace,
    ActionSpace,
    DefaultActionSpace,
    MinimalActionSpace,
    DefaultShapedReward,
)
from metamon.tokenizer import PokemonTokenizer, get_tokenizer
from custom.experiment import CustomExperiment

if metamon.METAMON_CACHE_DIR is None:
    raise ValueError("Set METAMON_CACHE_DIR environment variable")
# downloads checkpoints to the metamon cache dir where we're putting all the other data
MODEL_DOWNLOAD_DIR = os.path.join(metamon.METAMON_CACHE_DIR, "pretrained_models")

# registry for pretrained models
ALL_PRETRAINED_MODELS = {}

def pretrained_model(name: Optional[str] = None):
    """
    Decorator to register pretrained model classes.

    Args:
        name: Optional custom name for the model. If not provided, uses the class name.

    Usage:
        @pretrained_model()
        class MyModel(PretrainedModel):
            pass

        @pretrained_model("CustomName")
        class AnotherModel(PretrainedModel):
            pass
    """

    def _register(cls):
        model_name = name if name is not None else cls.__name__
        if model_name in ALL_PRETRAINED_MODELS:
            raise ValueError(f"Pretrained model '{model_name}' is already registered!")
        ALL_PRETRAINED_MODELS[model_name] = cls
        return cls

    return _register

def get_pretrained_model_names():
    return sorted(ALL_PRETRAINED_MODELS.keys())

def get_pretrained_model(name: str):
    if name not in ALL_PRETRAINED_MODELS:
        raise ValueError(
            f"Unknown pretrained model '{name}' (available models: {get_pretrained_model_names()})"
        )
    return ALL_PRETRAINED_MODELS[name]()

def make_placeholder_experiment(
    ckpt_base_dir: str,
    run_name: str,
    log: bool,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
):
    """
    Initialize an AMAGO experiment that will be used to load a pretrained checkpoint
    and manage agent/env interaction.
    """
    # the environment is only used to initialize the network
    # before loading the correct checkpoint
    penv = make_placeholder_env(
        observation_space=observation_space,
        action_space=action_space,
    )
    dummy_dset = amago.loading.DoNothingDataset()
    dummy_env = lambda: penv
    experiment = CustomExperiment(
        # assumes that positional args
        # agent_type, tstep_encoder_type,
        # traj_encoder_type, and max_seq_len
        # are set in the gin file
        ckpt_base_dir=ckpt_base_dir,
        run_name=run_name,
        dataset=dummy_dset,
        make_train_env=dummy_env,
        make_val_env=dummy_env,
        env_mode="sync",
        async_env_mp_context="spawn",
        parallel_actors=1,
        exploration_wrapper_type=None,
        epochs=0,
        start_learning_at_epoch=float("inf"),
        start_collecting_at_epoch=float("inf"),
        train_timesteps_per_epoch=0,
        stagger_traj_file_lengths=False,
        train_batches_per_epoch=0,
        val_interval=None,
        val_timesteps_per_epoch=0,
        ckpt_interval=None,
        always_save_latest=False,
        always_load_latest=False,
        log_interval=1,
        batch_size=1,
        dloader_workers=0,
        log_to_wandb=log,
        # wandb_project=os.environ.get("METAMON_WANDB_PROJECT"),
        # wandb_entity=os.environ.get("METAMON_WANDB_ENTITY"),
        verbose=True,
    )
    return experiment


class PretrainedModel:
    """
    Create an AMAGO agent and load a pretrained checkpoint from the HuggingFace Hub.

    This class handles downloading pretrained model weights from HuggingFace Hub,
    configuring the model architecture using gin files, and initializing the
    evaluation experiment.

    Args:
        model_gin_config: Path to gin config file that modifies the model architecture
            (layers, size, etc.)
        train_gin_config: Path to training gin config file. Does not have to be 1:1
            with training, but should match any architecture changes that were used.
        model_name: Model identifier used to locate the model in the HuggingFace Hub.
        tokenizer: Tokenizer for the text component of the observation space.
        observation_space: Observation space configuration. Uses original paper
            observation space by default.
        action_space: Action space configuration. The paper action space is now
            called MinimalActionSpace.
        reward_function: Reward function configuration. Uses original paper reward
            function by default.
        hf_cache_dir: Cache directory for HuggingFace Hub downloads. Note that
            these checkpoint files are large.
        default_checkpoint: Default checkpoint epoch to load. 40 corresponds to
            approximately 1M gradient steps with original paper training settings.
        gin_overrides: Optional dictionary of one-off gin overrides if there's a small tweak to an existing config file.
    """

    HF_REPO_ID = "jakegrigsby/metamon"

    def __init__(
        self,
        model_gin_config_path: str,
        train_gin_config_path: str,
        model_name: str,
        tokenizer:PokemonTokenizer = get_tokenizer("allreplays-v3"),
        observation_space: ObservationSpace = DefaultObservationSpace(),
        action_space: ActionSpace = DefaultActionSpace(),
        reward_function: RewardFunction = DefaultShapedReward(),
        hf_cache_dir: Optional[str] = None,
        gin_overrides: Optional[dict] = None,
    ):
        self.model_name = model_name
        self.model_gin_config_path = model_gin_config_path
        self.train_gin_config_path = train_gin_config_path
        self.hf_cache_dir = hf_cache_dir or MODEL_DOWNLOAD_DIR
        self.tokenizer = tokenizer
        self.observation_space = TokenizedObservationSpace(
            base_obs_space=observation_space,
            tokenizer=  tokenizer,
        )
        self.action_space = action_space
        self.reward_function = reward_function
        self.gin_overrides = gin_overrides
        os.makedirs(self.hf_cache_dir, exist_ok=True)

    @property
    def base_config(self) -> dict:
        """
        Override to set one-off changes to the gin config files
        """
  
        config = {
            # "amago.nets.traj_encoders.TformerTrajEncoder.attention_type": attn_type,    # not needed for HRM TrajEncoder
            "MetamonTstepEncoder.tokenizer": self.tokenizer,
            # skip cpu-intensive init, because we're going to be replacing the weights
            # with a checkpoint anyway....
            "amago.nets.transformer.SigmaReparam.fast_init": True,
        }
        if self.gin_overrides is not None:
            config.update(self.gin_overrides)
        return config

    def initialize_agent(
        self, ckpt_path: Optional[str] = None, log: bool = False
    ) -> CustomExperiment:
        # use the base config and the gin file to configure the model
        amago.cli_utils.use_config(
            self.base_config,
            [self.model_gin_config_path, self.train_gin_config_path],
            finalize=False,
        )
        ckpt_base_dir = str(Path(ckpt_path).parents[2])
        # build an experiment
        experiment = make_placeholder_experiment(
            ckpt_base_dir=ckpt_base_dir,
            run_name=self.model_name,
            log=log,
            observation_space=self.observation_space,
            action_space=self.action_space,
        )
        
        print(f"Radha raman laal bhajo radhe govinda==> {ckpt_path}")
        # starting the experiment will build the initial model
        experiment.start()
        if ckpt_path is not None:
            # replace the weights with the pretrained checkpoint
            experiment.load_checkpoint_from_path(ckpt_path, is_accelerate_state=False)
        return experiment
    
@pretrained_model()
class HRM_gen1ou(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="hrm_gen1ou",
            model_gin_config_path="custom/gen1/configs/hrm_agent.gin",
            train_gin_config_path="custom/gen1/configs/train.gin",
            action_space=MinimalActionSpace(),
            reward_function=DefaultShapedReward(),
            observation_space=DefaultObservationSpace(), 
            tokenizer=get_tokenizer(choice="allreplays-v3")
        )

@pretrained_model()
class HRM_gen9ou_ABRA(PretrainedModel):
    def __init__(self):
        super().__init__(
            model_name="hrm_gen9ou_abra",
            model_gin_config_path="custom/gen9/configs/hrm_agent.gin",
            train_gin_config_path="custom/gen9/configs/train.gin",
            action_space=DefaultActionSpace(),
            reward_function=DefaultShapedReward(),
            observation_space=TeamPreviewObservationSpace(), 
            tokenizer=get_tokenizer(choice="DefaultObservationSpace-v1")
        )



HEURISTIC_COMPOSITE_BASELINES = [
    "PokeEnvHeuristic",
    "Gen1BossAI",
    "Grunt",
    "GymLeader",
    "EmeraldKaizo",
    "RandomBaseline",
]


def pretrained_vs_baselines(
    pretrained_model: PretrainedModel,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    ckpt_path: Optional[str] = None,
    total_battles: int = 250,
    parallel_actors_per_baseline: int = 5,
    async_mp_context: str = "forkserver",
    battle_backend: str = "poke-env",
    log_to_wandb: bool = False,
    save_trajectories_to: Optional[str] = None,
    save_team_results_to: Optional[str] = None,
    baselines: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Evaluate a pretrained model against built-in baseline opponents.

    Defaults to the 6 baselines that the paper calls the "Heuristic Composite Score",
    but you can specify a list of any of the available baselines (see metamon.baselines.get_all_baseline_names()).
    """
    agent = pretrained_model.initialize_agent(ckpt_path=ckpt_path, log=log_to_wandb)
    baselines = baselines or HEURISTIC_COMPOSITE_BASELINES
    agent.async_env_mp_context = async_mp_context
    # create envs that match the agent's observation/actions/rewards
    make_envs = [
        functools.partial(
            make_baseline_env,
            battle_format=battle_format,
            observation_space=pretrained_model.observation_space,
            action_space=pretrained_model.action_space,
            reward_function=pretrained_model.reward_function,
            save_trajectories_to=save_trajectories_to,
            save_team_results_to=save_team_results_to,
            battle_backend=battle_backend,
            team_set=team_set,
            opponent_type=get_baseline(opponent),
        )
        for opponent in baselines
    ]
    # amago will play `parallel_actors_per_baseline` copies of each baseline
    # in parallel and aggregate the results by baseline name.
    make_envs *= parallel_actors_per_baseline
    # evaluate
    agent.parallel_actors = len(make_envs)
    results = agent.evaluate_test(
        make_envs,
        timesteps=total_battles * 250 // len(make_envs),
        episodes=total_battles,
    )
    return results

def _pretrained_on_ladder(
    pretrained_model: PretrainedModel,
    make_ladder: Callable,
    total_battles: int,
    ckpt_path: Optional[str] ,
    log_to_wandb: bool,
    **ladder_kwargs,
) -> Dict[str, Any]:
    """Helper function for ladder-based evaluation."""
    agent = pretrained_model.initialize_agent(ckpt_path=ckpt_path, log=log_to_wandb)
    agent.env_mode = "sync"
    agent.parallel_actors = 1
    agent.verbose = False  # turn off tqdm progress bar and print poke-env battle status

    make_env = functools.partial(
        make_ladder,
        observation_space=pretrained_model.observation_space,
        action_space=pretrained_model.action_space,
        reward_function=pretrained_model.reward_function,
        num_battles=total_battles,
        **ladder_kwargs,
    )

    results = agent.evaluate_test(
        [make_env],
        timesteps=total_battles * 1000,
        episodes=total_battles,
    )
    return results

def pretrained_vs_local_ladder(
    pretrained_model: PretrainedModel,
    username: str,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    total_battles: int,
    avatar: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    battle_backend: str = "poke-env",
    save_trajectories_to: Optional[str] = None,
    save_team_results_to: Optional[str] = None,
    log_to_wandb: bool = False,
) -> Dict[str, Any]:
    """Evaluate a pretrained model on the ladder of your Local Showdown server.

    Make sure you've started your local server in the background with
    `node pokemon-showdown start --no-security`. Usernames must be unique,
    but do not need to be registered in advance, and do not require a password.

    Will automatically queue the agent for battles against any other agents (or humans)
    that are also online. This is the simplest way to evaluate pretrained models head-to-head
    and generate self-play data. It is also how the paper handled evals against third-party
    baselines like PokéLLMon.
    """

    return _pretrained_on_ladder(
        pretrained_model=pretrained_model,
        make_ladder=make_local_ladder_env,
        total_battles=total_battles,
        ckpt_path=ckpt_path,
        log_to_wandb=log_to_wandb,
        player_username=username,
        player_avatar=avatar,
        player_team_set=team_set,
        battle_backend=battle_backend,
        battle_format=battle_format,
        save_trajectories_to=save_trajectories_to,
        save_team_results_to=save_team_results_to,
    )

def pretrained_vs_pokeagent_ladder(
    pretrained_model: PretrainedModel,
    username: str,
    password: str,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    total_battles: int,
    avatar: Optional[str] = None,
    ckpt_path: Optional[str] = None,
    battle_backend: str = "poke-env",
    save_trajectories_to: Optional[str] = None,
    save_team_results_to: Optional[str] = None,
    log_to_wandb: bool = False,
) -> Dict[str, Any]:
    """Evaluate a pretrained model on the PokéAgent Challenge ladder.

    Must provide a registered username and password. See instructions in the README!

    Will automatically queue the agent for ranked battles against any other agents (or humans)
    that are logged into the PokéAgent Challenge ladder.

    Once eval begins, you can watch battles in real time by visiting
    http://pokeagentshowdown.com.insecure.psim.us and clicking "Watch a Battle".
    Visit http://pokeagentshowdown.com.insecure.psim.us/ladder to see the live
    leaderboard.
    """
    return _pretrained_on_ladder(
        pretrained_model=pretrained_model,
        make_ladder=make_pokeagent_ladder_env,
        total_battles=total_battles,
        ckpt_path=ckpt_path,
        log_to_wandb=log_to_wandb,
        player_username=username,
        player_password=password,
        player_avatar=avatar,
        player_team_set=team_set,
        battle_backend=battle_backend,
        battle_format=battle_format,
        save_trajectories_to=save_trajectories_to,
        save_team_results_to=save_team_results_to,
    )

def _get_default_eval(args, base_eval_kwargs):
    """Get the appropriate evaluation helper and update required args based on eval_type."""
    if args.eval_type == "heuristic":
        base_eval_kwargs.update(
            {
                "baselines": HEURISTIC_COMPOSITE_BASELINES,
                "async_mp_context": args.async_mp_context,
            }
        )
        return pretrained_vs_baselines
    elif args.eval_type == "il":
        base_eval_kwargs.update(
            {
                "baselines": ["BaseRNN"],
                "async_mp_context": args.async_mp_context,
                # sets this low to avoid overloading CPU with RNN baseline inference
                "parallel_actors_per_baseline": 1,
            }
        )
        return pretrained_vs_baselines
    elif args.eval_type == "ladder":
        base_eval_kwargs.update(
            {
                "username": args.username,
                "avatar": args.avatar,
            }
        )
        return pretrained_vs_local_ladder
    elif args.eval_type == "pokeagent":
        base_eval_kwargs.update(
            {
                "username": args.username,
                "password": args.password,
                "avatar": args.avatar,
            }
        )
        return pretrained_vs_pokeagent_ladder
    else:
        raise ValueError(f"Invalid evaluation type: {args.eval_type}")

def _run_default_evaluation(args) -> Dict[str, List[Dict[str, Any]]]:
    pretrained_model = get_pretrained_model(args.agent)
    all_results = collections.defaultdict(list)
    for gen in args.gens:
        for format_name in args.formats:
            battle_format = f"gen{gen}{format_name.lower()}"
            player_team_set = metamon.env.get_metamon_teams(
                battle_format, args.team_set
            )
            for ckpt_path in args.checkpoints:
                eval_kwargs = {
                    "pretrained_model": pretrained_model,
                    "battle_format": battle_format,
                    "team_set": player_team_set,
                    "total_battles": args.total_battles,
                    "ckpt_path": ckpt_path,
                    "battle_backend": args.battle_backend,
                    "save_trajectories_to": args.save_trajectories_to,
                    "save_team_results_to": args.save_team_results_to,
                    "log_to_wandb": args.log_to_wandb,
                }
                eval_function = _get_default_eval(args, eval_kwargs)
                results = eval_function(**eval_kwargs)
                print(json.dumps(results, indent=4, sort_keys=True))
                all_results[battle_format].append(results)
    return all_results


def add_cli(parser):
    parser.add_argument(
        "--agent",
        required=True,
        choices=get_pretrained_model_names(),
        help="Choose a pretrained model to evaluate.",
    )
    parser.add_argument(
        "--eval_type",
        required=True,
        choices=["heuristic", "il", "ladder", "pokeagent"],
        help=(
            "Type of evaluation to perform. 'heuristic' will run against 6 "
            "heuristic baselines, 'il' will run against a BCRNN baseline, "
            "'ladder' will queue the agent for battles on your self-hosted Showdown ladder, "
            "'pokeagent' will submit the agent to the NeurIPS 2025 PokéAgent Challenge ladder!"
        ),
    )
    parser.add_argument(
        "--gens",
        type=int,
        nargs="+",
        default=[1],
        help="Specify the Pokémon generations to evaluate.",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=["ou"],
        choices=["ubers", "ou", "uu", "nu"],
        help="Specify the battle tier.",
    )
    parser.add_argument(
        "--total_battles",
        type=int,
        default=10,
        help=(
            "Number of battles to run before returning eval stats. "
            "Note this is the total sample size across all parallel actors (if applicable)."
        ),
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        nargs="+",
        default=[None],
        help="Path of checkpoints to evaulate by loading in policy.",
    )
    parser.add_argument(
        "--username",
        default="shyamsundar",
        help="Username for the Showdown server.",
    )
    parser.add_argument(
        "--password",
        default="shyamsundar",
        help="Password for the Showdown server.",
    )
    parser.add_argument(
        "--avatar",
        default="red-gen1main",
        help="Avatar to use for the battles.",
    )
    parser.add_argument(
        "--team_set",
        default="competitive",
        choices=[
            "competitive",
            "paper_variety",
            "paper_replays",
            "modern_replays",
            "pokeagent_modern_replays",
        ],
        help="Team Set. See the README for more details.",
    )
    parser.add_argument(
        "--battle_backend",
        type=str,
        default="poke-env",
        choices=["poke-env", "metamon"],
        help=(
            "Method for interpreting Showdown's requests and simulator messages. "
            "poke-env is the default. metamon is an experimental option that aims to "
            "remove sim2sim gap by reusing the code that generates our huggingface "
            "replay dataset."
        ),
    )
    parser.add_argument(
        "--async_mp_context",
        type=str,
        default="forkserver",
        help="Async environment setup method. Does not apply to `--eval_type ladder` or `--eval_type pokeagent`. Options: 'forkserver' (recommended, fast), 'fork' (fastest but unsafe with threads), 'spawn' (slowest but safest). Use 'spawn' only if others hang.",
    )
    parser.add_argument(
        "--save_trajectories_to",
        default=None,
        help="Save replays (in the parsed replay format) to a directory.",
    )
    parser.add_argument(
        "--save_team_results_to",
        default=None,
        help="Save records of team selection, opponent, and outcome.",
    )
    parser.add_argument(
        "--log_to_wandb",
        action="store_true",
        help="Log results to Weights & Biases.",
    )

    return parser


if __name__ == "__main__":
    from argparse import ArgumentParser

    parser = ArgumentParser(
        description="Evaluate a pretrained Metamon model by playing battles against opponents. "
        "This script allows you to evaluate a pretrained model's performance against a set of "
        "heuristic baselines, local ladder, or the PokéAgent Challenge ladder. It can also save replays in the same format "
        "as the human replay dataset for further training."
    )
    add_cli(parser)
    args = parser.parse_args()
    _run_default_evaluation(args)
