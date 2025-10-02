import gin
import warnings
from dataclasses import dataclass
from metamon.rl.metamon_to_amago import MetamonAMAGOExperiment
from accelerate.state import DistributedType
from amago import utils
from custom.hrm_agent import HRM_MultiTaskAgent
import warnings
from dataclasses import dataclass
from typing import Optional, Iterable, Any
import os
import gin
import gymnasium as gym
import torch
from amago import utils
from amago.envs.env_utils import (
    DummyAsyncVectorEnv,
    AlreadyVectorizedEnv,
)
from amago.envs.exploration import (
    ExplorationWrapper,
    EpsilonGreedy
)
from amago.envs import EnvCreator
from tqdm import tqdm


@gin.configurable
@dataclass
class CustomExperiment(MetamonAMAGOExperiment):
    ckpt_vol:Any=None
    def start(self):
        """
        Overrides the parent's start method to inject print statements for tracking.
        This helps identify exactly where the initialization process gets stuck.
        """
        print("🚀 Starting Experiment Initialization...")
        
        print("⏳ Initializing datasets...", end=" || ")
        self.init_dsets()
        print("✅ Done!")
        
        print("⏳ Initializing environments...", end=" || ")
        # The warnings context manager is a good practice you kept
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("always", category=utils.AmagoWarning)
            env_summary = self.init_envs()
        print("✅ Done!")

        print("⏳ Initializing model...", end=" || ")
        self.init_model()
        print("✅ Done!")
    
        print("⏳ Initializing dataloaders...", end=" || ")
        self.init_dloaders()
        print("✅ Done!")

        print("⏳ Initializing checkpoints...", end=" || ")
        self.init_checkpoints()
        print("✅ Done!")

        print("⏳ Initializing logger...", end=" || ")
        self.init_logger()
        if self.verbose:
            # The summary call is important for visibility
            self.summary(env_summary=env_summary)
        print("✅ Done!")
        print("🎉 Initialization Complete! Ready to learn. 🎉")

        
    @property
    def policy(self):
        if self.accelerator.state.distributed_type != DistributedType.NO:
            return self.accelerator.unwrap_model(self.policy_aclr)
        return self.policy_aclr
    
    def init_model(self) -> None:
        """Build an initial policy based on observation shapes"""
        policy_kwargs = {
            "tstep_encoder_type": self.tstep_encoder_type,
            "traj_encoder_type": self.traj_encoder_type,
            "obs_space": self.rl2_space["obs"],
            "rl2_space": self.rl2_space["rl2"],
            "action_space": self.train_envs.single_action_space,
            "max_seq_len": self.max_seq_len,
        }
        self.agent = self.agent_type(**policy_kwargs)
        assert isinstance(self.agent, HRM_MultiTaskAgent)
        optimizer = self.init_optimizer(self.agent)
        lr_schedule = utils.get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=self.lr_warmup_steps
        )
        self.policy_aclr, self.optimizer, self.lr_schedule = self.accelerator.prepare(
            self.agent, optimizer, lr_schedule
        )
        self.accelerator.register_for_checkpointing(self.lr_schedule)
        self.grad_update_counter = 0

    def init_envs(self) -> str:
        """Construct parallel training and validation environments.

        Returns:
            str: Description of the environment setup printed to the console when
                Experiment.verbose is True.
        """
        assert self.traj_save_len >= self.max_seq_len

        if self.env_mode in ["async", "sync"]:
            print(f"\t🌍Createing {self.parallel_actors} parallel envs in '{self.env_mode}' mode...")
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
            print(f"\t🌍Using already vectorized envs with {self.parallel_actors} environments...")
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
        print(f"\t🌍Creating parallel train({len(make_train)}) /val({len(make_val)}) envs...")
        self.train_envs = Par(make_train, **par_kwargs)
        print(f"\t🌍\t🌿Done creating parallel training envs (total= {len(make_train)})...")
        # self.val_envs = Par(make_val, **par_kwargs)
        # print(f"\t🌍\t🌿Done creating parallel validation envs (total= {len(make_val)})...")
        self.train_envs.reset()
        print(f"\t🌍Reset training envs...")
        self.rl2_space = make_train[0].rl2_space
        self.hidden_state = None  # holds train_env hidden state between epochs

        if self.env_mode == "already_vectorized":
            _inner = f"Vectorized Gym Env x{self.parallel_actors}"
            _desc = f"{Par.__name__}({_inner})"
        else:
            _inner = "Gym Env"
            _desc = f"{Par.__name__}({_inner} x {self.parallel_actors})"
        return _desc

    def save_checkpoint(self) -> None:
        """Save both the training state and the policy weights to the ckpt_dir."""
        ckpt_name = f"{self.run_name}_epocckpt_nameh_{self.epoch}"
        self.accelerator.save_state(
            os.path.join(self.ckpt_dir, "training_states", ckpt_name),
            safe_serialization=True,
        )
        if self.accelerator.is_main_process:
            # create backup of raw weights unrelated to the more complex process of resuming an accelerate state
            torch.save(
                self.policy.state_dict(),
                os.path.join(
                    self.ckpt_dir, "policy_weights", f"policy_epoch_{self.epoch}.pt"
                ),
            )
        self.ckpt_vol.commit()

    def write_latest_policy(self) -> None:
        """Write absolute latest policy to a hardcoded location used by `read_latest_policy`"""
        ckpt_name = os.path.join(self.ckpt_dir, "latest", "policy.pt")
        torch.save(self.policy.state_dict(), ckpt_name)
        self.ckpt_vol.commit()
        
