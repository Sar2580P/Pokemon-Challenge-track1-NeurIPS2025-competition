import json
import collections
import functools
from typing import Optional, Dict, Any, Callable, List, Tuple
import amago
from pathlib import Path
from metamon.rl.metamon_to_amago import (
    make_placeholder_experiment,
)
from metamon.rl.pretrained import PretrainedModel
import types
from metamon.rl.pretrained import get_pretrained_model, get_pretrained_model_names
from inference.experiment_tribe import TribeExperiment
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
import os

from metamon.rl.metamon_to_amago import make_placeholder_env
import metamon
from metamon.rl.pretrained import (
    PretrainedModel,
)
from metamon.baselines import get_baseline
from metamon.rl.metamon_to_amago import (
    make_baseline_env,
    make_local_ladder_env,
    make_pokeagent_ladder_env,
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
    agent, 
    pretrained_model: PretrainedModel,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    checkpoint: Optional[int] = None,
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
    # agent = pretrained_model.initialize_agent(checkpoint=checkpoint, log=log_to_wandb)
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
    agent, 
    pretrained_model: PretrainedModel,
    make_ladder: Callable,
    total_battles: int,
    checkpoint: Optional[int],
    log_to_wandb: bool,
    **ladder_kwargs,
) -> Dict[str, Any]:
    """Helper function for ladder-based evaluation."""
    # agent = pretrained_model.initialize_agent(checkpoint=checkpoint, log=log_to_wandb)
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
    assert isinstance(agent, TribeExperiment)
    results = agent.evaluate_test(
        [make_env],
        timesteps=total_battles * 1000,
        episodes=total_battles,
    )
    return results


def pretrained_vs_local_ladder(
    agent, 
    pretrained_model: PretrainedModel,
    username: str,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    total_battles: int,
    avatar: Optional[str] = None,
    checkpoint: Optional[int] = None,
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
        agent=agent, 
        pretrained_model=pretrained_model,
        make_ladder=make_local_ladder_env,
        total_battles=total_battles,
        checkpoint=checkpoint,
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
    agent, 
    pretrained_model: PretrainedModel,
    username: str,
    password: str,
    battle_format: str,
    team_set: metamon.env.TeamSet,
    total_battles: int,
    avatar: Optional[str] = None,
    checkpoint: Optional[int] = None,
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
        agent=agent, 
        pretrained_model=pretrained_model,
        make_ladder=make_pokeagent_ladder_env,
        total_battles=total_battles,
        checkpoint=checkpoint,
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

def _run_default_evaluation(args , agent) -> Dict[str, List[Dict[str, Any]]]:
    all_results = collections.defaultdict(list)
    pretrained_model = get_pretrained_model(args.agent)
    for gen in args.gens:
        for format_name in args.formats:
            battle_format = f"gen{gen}{format_name.lower()}"
            player_team_set = metamon.env.get_metamon_teams(
                battle_format, args.team_set
            )
            eval_kwargs = {
                "pretrained_model": pretrained_model,
                "battle_format": battle_format,
                "team_set": player_team_set,
                "total_battles": args.total_battles,
                "battle_backend": args.battle_backend,
                "save_trajectories_to": args.save_trajectories_to,
                "save_team_results_to": args.save_team_results_to,
                "log_to_wandb": args.log_to_wandb,
            }
            eval_function = _get_default_eval(args, eval_kwargs)
            results = eval_function(agent=agent, **eval_kwargs)
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
        "--username",
        default="Metamon",
        help="Username for the Showdown server.",
    )
    parser.add_argument(
        "--password",
        default=None,
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


from typing import Optional, Iterable
import math

import gin
from termcolor import colored
import torch
from torch.utils.data import DataLoader
import numpy as np
from einops import repeat
import gymnasium as gym
from accelerate import Accelerator, DistributedDataParallelKwargs
from accelerate.utils import tqdm

from amago import utils
from amago.envs.env_utils import (
    DummyAsyncVectorEnv,
    AlreadyVectorizedEnv,
)

def init_envs(self) -> str:
    """Construct parallel training and validation environments.

    Returns:
        str: Description of the environment setup printed to the console when
            Experiment.verbose is True.
    """
    assert self.traj_save_len >= self.max_seq_len

    if self.env_mode in ["async", "sync"]:
        # default environment mode wrapping individual gym environments in a pool of async processes
        # and handling resets by waiting for the termination signal to reach the highest wrapper level
        to_env_list = lambda e: (
            [e] * self.parallel_actors if not isinstance(e, Iterable) else e
        )
        make_val_envs = to_env_list(self.make_val_env)
        make_train_envs = to_env_list(self.make_train_env)
        if not len(make_train_envs) == self.parallel_actors:
            utils.amago_warning(
                f"`Experiment.parallel_actors` is {self.parallel_actors} but `make_train_env` is a list of length {len(make_train_envs)}"
            )
        if not len(make_val_envs) == self.parallel_actors:
            utils.amago_warning(
                f"`Experiment.parallel_actors` is {self.parallel_actors} but `make_val_env` is a list of length {len(make_val_envs)}"
            )
        if self.env_mode == "async":
            Par = gym.vector.AsyncVectorEnv
            par_kwargs = dict(context=self.async_env_mp_context)
        else:
            Par = DummyAsyncVectorEnv
            par_kwargs = dict()
    elif self.env_mode == "already_vectorized":
        # alternate environment mode designed for jax / gpu-accelerated envs that handle parallelization
        # with a batch dimension on the lowest wrapper level. These envs must auto-reset and treat the last
        # timestep of a trajectory as the first timestep of the next trajectory.
        make_train_envs = [self.make_train_env]
        make_val_envs = [self.make_val_env]
        Par = AlreadyVectorizedEnv
        par_kwargs = dict()
    else:
        raise ValueError(f"Invalid `env_mode` {self.env_mode}")

    if self.exploration_wrapper_type is not None and not issubclass(
        self.exploration_wrapper_type, ExplorationWrapper
    ):
        utils.amago_warning(
            f"Implement exploration strategies by subclassing `ExplorationWrapper` and setting the `Experiment.exploration_wrapper_type`"
        )

    if self.max_seq_len < self.traj_save_len and self.stagger_traj_file_lengths:
        """
        If the rollout length of the environment is much longer than the `traj_save_len`,
        almost every datapoint will be exactly `traj_save_len` long and spaced `traj_save_len` apart.
        For example if the `traj_save_len` is 100 the trajectory files will all be snippets from
        [0, 100], [100, 200], [200, 300], etc. This can lead to a problem at test-time because the model
        has never seen a sequence from timesteps [50, 150] or [150, 250], etc. We can mitigate this by
        randomizing the trajectory lengths in a range around `traj_save_len`.
        """
        save_every_low = self.traj_save_len - self.max_seq_len
        save_every_high = self.traj_save_len + self.max_seq_len
    else:
        save_every_low = save_every_high = self.traj_save_len

    # wrap environments to save trajectories to replay buffer
    shared_env_kwargs = dict(
        save_trajs_as=self.save_trajs_as,
        save_every_low=save_every_low,
        save_every_high=save_every_high,
    )
    make_train = [
        EnvCreator(
            make_env=env_func,
            # save trajectories to disk
            save_trajs_to=self.dataset.save_new_trajs_to,
            # adds exploration noise
            exploration_wrapper_type=self.exploration_wrapper_type,
            **shared_env_kwargs,
        )
        for env_func in make_train_envs
    ]
    make_val = [
        EnvCreator(
            make_env=env_func,
            # do not save trajectories to disk
            save_trajs_to=None,
            # no exploration noise
            exploration_wrapper_type=None,
            **shared_env_kwargs,
        )
        for env_func in make_val_envs
    ]

    # make parallel envs
    self.train_envs = Par(make_train, **par_kwargs)
    self.val_envs = Par(make_val, **par_kwargs)
    # self.train_envs.reset()
    self.rl2_space = make_train[0].rl2_space
    self.hidden_state = None  # holds train_env hidden state between epochs

    if self.env_mode == "already_vectorized":
        _inner = f"Vectorized Gym Env x{self.parallel_actors}"
        _desc = f"{Par.__name__}({_inner})"
    else:
        _inner = "Gym Env"
        _desc = f"{Par.__name__}({_inner} x {self.parallel_actors})"
    return _desc

def initialize_agent(
    self, checkpoint: Optional[int] = None, log: bool = False
) -> amago.Experiment:
    # use the base config and the gin file to configure the model
    
    amago.cli_utils.use_config(
        self.base_config,
        [self.model_gin_config_path, self.train_gin_config_path],
        finalize=False,
    )
    checkpoint = checkpoint if checkpoint is not None else self.default_checkpoint
    ckpt_path = self.get_path_to_checkpoint(checkpoint)
    ckpt_base_dir = str(Path(ckpt_path).parents[2])
    # build an experiment
    experiment = make_placeholder_experiment(
        ckpt_base_dir=ckpt_base_dir,
        run_name=self.model_name,
        log=log,
        observation_space=self.observation_space,
        action_space=self.action_space,
    )
    # starting the experiment will build the initial model
    experiment.init_envs()
    experiment.init_model()
    # experiment.train_envs=None
    # experiment.val_envs=None
    if checkpoint > 0:
        # replace the weights with the pretrained checkpoint
        experiment.load_checkpoint_from_path(ckpt_path, is_accelerate_state=False)
    
    experiment.policy.parallel_actors= experiment.parallel_actors
    return experiment.policy


def make_placeholder_experiment_tribe_sardar(
    ckpt_base_dir: str,
    run_name: str,
    log: bool,
    observation_space: ObservationSpace,
    action_space: ActionSpace,
)-> TribeExperiment:
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
    experiment = TribeExperiment(
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
        wandb_project=os.environ.get("METAMON_WANDB_PROJECT"),
        wandb_entity=os.environ.get("METAMON_WANDB_ENTITY"),
        verbose=True,
    )
    return experiment


def intialize_agent_tribe_sardar(self, checkpoint: Optional[str] = None, log: bool = False)-> TribeExperiment:
    # use the base config and the gin file to configure the model
    self.model_gin_config_path= self.model_gin_config_path.replace('/metamon/rl/configs/models', '')
    self.train_gin_config_path= self.train_gin_config_path.replace('/metamon/rl/configs/training', '')
    amago.cli_utils.use_config(
        self.base_config,
        [self.model_gin_config_path, self.train_gin_config_path],
        finalize=False,
    )
    ckpt_base_dir = checkpoint
    # build an experiment
    experiment = make_placeholder_experiment_tribe_sardar(
        ckpt_base_dir=ckpt_base_dir,
        run_name=self.model_name,
        log=log,
        observation_space=self.observation_space,
        action_space=self.action_space,
    )
    assert isinstance(experiment, TribeExperiment)
    # starting the experiment will build the initial model
    experiment.start()
    # experiment.init_envs()
    # experiment.init_model()
    return experiment


def create_tribe_population(members_config:List[Tuple]=None):
    from metamon.rl.pretrained import (SyntheticRLV0, SyntheticRLV1, 
                                       SyntheticRLV1_SelfPlay, SyntheticRLV1_PlusPlus, SyntheticRLV2)
    gen1_models = {SyntheticRLV0.__name__: SyntheticRLV0, 
                   SyntheticRLV1.__name__: SyntheticRLV1, 
                   SyntheticRLV1_SelfPlay.__name__: SyntheticRLV1_SelfPlay, 
                   SyntheticRLV1_PlusPlus.__name__: SyntheticRLV1_PlusPlus, 
                   SyntheticRLV2.__name__: SyntheticRLV2}

    model_tribe=[]
    for ele in members_config:
        model_name, cfg= ele[0], ele[1]
        cls = gen1_models.get(model_name, None)
        if cls is not None: 
            instance:PretrainedModel = cls()
            instance.model_gin_config_path = cfg['model_gin_config']
            instance.train_gin_config_path = cfg['train_gin_config']
            instance.default_checkpoint=cfg['checkpoint']            
            
            # 🐒🐒🐒 monkey-patching
            instance.initialize_agent = types.MethodType(initialize_agent, instance)
            try:
                policy = instance.initialize_agent()
                policy.eval()
                model_tribe.append(policy)
                print(f"🤖 SUCCESS: initialize {model_name} with ckpt={cfg['checkpoint']}")
            except Exception as e: 
                print(f"❌ 💔 FAILED: initialize {model_name} with ckpt={cfg['checkpoint']}.\nERROR: {e}")
    return model_tribe
    
def create_sarpanch(
        model_gin_config_path:str , train_gin_config_path:str, gen:int, 
        action_space=MinimalActionSpace()
    ) -> TribeExperiment:

    sardar = PretrainedModel(
        model_name=f"KabeeleKaSardar-gen{gen}", 
        model_gin_config=model_gin_config_path, 
        train_gin_config=train_gin_config_path, 
        action_space=action_space
    )
    # 🐒🐒🐒 monkey-patching
    sardar.initialize_agent = types.MethodType(intialize_agent_tribe_sardar, sardar)
    
    sardar = sardar.initialize_agent(checkpoint='/vol/PAC-dataset/')
    print(f"👑 Initialized kabeele ka sardar")
    return sardar
    
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
