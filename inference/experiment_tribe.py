import gin
import warnings
from dataclasses import dataclass
from metamon.rl.metamon_to_amago import MetamonAMAGOExperiment
from accelerate.state import DistributedType
from amago import utils
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
from einops import repeat
from amago.loading import (
    Batch,
    RLDataset,
    RLData_pad_collate,
    MAGIC_PAD_VAL,
)
from amago.envs import EnvCreator
from tqdm import tqdm
from amago.envs import ReturnHistory, SpecialMetricHistory, EnvCreator
import numpy as np

@gin.configurable
@dataclass
class TribeExperiment(MetamonAMAGOExperiment):
    start_epoch:int=0
    
    def start(self):
        """
        Overrides the parent's start method to inject print statements for tracking.
        This helps identify exactly where the initialization process gets stuck.
        """        
        self.init_dsets()
        
        # The warnings context manager is a good practice you kept
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            warnings.filterwarnings("always", category=utils.AmagoWarning)
            env_summary = self.init_envs()

        self.init_model()
    
        self.init_dloaders()

        self.init_checkpoints()
        
        # set the start epoch
        self.epoch=self.start_epoch
        self.epochs=self.epochs+self.start_epoch

        self.init_logger()
        if self.verbose:
            # The summary call is important for visibility
            self.summary(env_summary=env_summary)
        print("🎉 Initialization Complete!! 🎉")

        
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
        # self.val_envs = Par(make_val, **par_kwargs)
        # print(f"\t🌍\t🌿Done creating parallel validation envs (total= {len(make_val)})...")
        self.train_envs.reset()
        self.rl2_space = make_train[0].rl2_space
        self.hidden_state = None  # holds train_env hidden state between epochs

        if self.env_mode == "already_vectorized":
            _inner = f"Vectorized Gym Env x{self.parallel_actors}"
            _desc = f"{Par.__name__}({_inner})"
        else:
            _inner = "Gym Env"
            _desc = f"{Par.__name__}({_inner} x {self.parallel_actors})"
        return _desc
        
                
    def load_checkpoint_from_path(
        self, path: str, is_accelerate_state: bool = True
    ) -> None:
        """Load a checkpoint from a given path.

        Args:
            path: Full path to the checkpoint fle to load.
            is_accelerate_state: Whether the checkpoint is a full accelerate state (True) or
                pytorch weights only (False). Defaults to True.
        """
        # if not is_accelerate_state:
        #     ckpt = utils.retry_load_checkpoint(path, map_location=self.DEVICE)
        #     model_dict = self.policy.state_dict()

        #     # 1. Filter the loaded checkpoint.
        #     #    Keep a key 'k' only if it exists in the current model's state_dict.
        #     filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_dict}
            
        #     # Optional: You can check if the shapes match too, though PyTorch does this automatically.
        #     # filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}

        #     # 2. Load the perfectly filtered state dictionary
        #     self.policy.load_state_dict(filtered_ckpt)
        # else:
        #     self.accelerator.load_state(path)
        
        print(f"🌷🎀 This is tribe experiment. Skipping ckpt loading...")

    def interact(
        self,
        envs,
        timesteps: int,
        hidden_state=None,
        render: bool = False,
        save_on_done: bool = False,
        episodes: Optional[int] = None,
    ) -> tuple[ReturnHistory, SpecialMetricHistory]:
        """Main policy loop for interacting with the environment.

        Args:
            envs: The (parallel) environments to interact with.
            timesteps: The number of timesteps to interact with each environment.

        Keyword Args:
            hidden_state: The hidden state of the policy. If None, a fresh hidden state is
                initialized. Defaults to None.
            render: Whether to render the environment. Defaults to False.
            save_on_done: If True, save completed trajectory sequences to disk as soon as they
                are finished. If False, wait until all rollouts are completed. Only applicable
                if the provided envs are configured to save rollouts to disk. Defaults to False.
            episodes: The number of episodes to interact with the environment. If provided, the
                loop will terminate after this many episodes have been completed OR we hit the
                `timesteps` limit, whichever comes first. Defaults to None.

        Returns:
            tuple[ReturnHistory, SpecialMetricHistory]: Objects that keep track of standard
                eval stats (average returns) and any additional eval metrics the envs have been
                configured to record.
        """
        policy = self.policy
        policy.eval()

        if self.verbose:
            iter_ = tqdm(
                range(timesteps),
                desc="Env Interaction",
                total=timesteps,
                leave=False,
                colour="yellow",
            )
        else:
            iter_ = range(timesteps)

        # clear results statistics
        # (can make train-time stats useless depending on horizon vs. `timesteps`)
        utils.call_async_env(envs, "reset_stats")
        
        if hidden_state is None:
            # init new hidden state
            hidden_state = []
            for agent in self.policy.inference_entities:
               h= agent.traj_encoder.init_hidden_state(
                        self.parallel_actors, self.DEVICE
                    )   
               hidden_state.append(h)
        
        
               
            
        def get_t():
            # fetch `Timestep.make_sequence` from all envs
            par_obs_rl2_time = utils.call_async_env(envs, "current_timestep")
            _obs, _rl2s, _time_idxs = [], [], []
            for _o, _r, _t in par_obs_rl2_time:
                _obs.append(_o)
                _rl2s.append(_r)
                _time_idxs.append(_t)
            # stack all the results
            _obs = utils.stack_list_array_dicts(_obs, axis=0, cat=True)
            _rl2s = np.concatenate(_rl2s, axis=0)
            _time_idxs = np.concatenate(_time_idxs, axis=0)
            # ---> torch --> GPU --> dummy length dim
            _obs = {
                k: torch.from_numpy(v).to(self.DEVICE).unsqueeze(1)
                for k, v in _obs.items()
            }
            _rl2s = torch.from_numpy(_rl2s).to(self.DEVICE).unsqueeze(1)
            _time_idxs = torch.from_numpy(_time_idxs).to(self.DEVICE).unsqueeze(1)
            return _obs, _rl2s, _time_idxs

        obs, rl2s, time_idxs = get_t()
        episodes_finished = 0
        prev_reward=0
        past_dones=np.array([False]*time_idxs.shape[0])
        for _ in iter_:
            with torch.no_grad():
                with self.caster():
                    actions, hidden_state = policy.get_actions(
                        obs=obs,
                        rl2s=rl2s,
                        time_idxs=time_idxs,
                        sample=self.sample_actions,
                        hidden_state=hidden_state,
                        rewards=prev_reward, 
                        dones=past_dones
                    )
            next_obs, reward, terminated, truncated, info = envs.step(actions.squeeze(1).cpu().numpy())
            done = terminated | truncated
            prev_reward=reward
            if done.ndim == 2:
                done = done.squeeze(1)
            if done.any():
                if save_on_done:
                    utils.call_async_env(envs, "save_finished_trajs")
                episodes_finished += done.sum()
            obs, rl2s, time_idxs = get_t()
            
            
            for idx, agent in enumerate(self.policy.inference_entities):
                hidden_state[idx] = agent.traj_encoder.reset_hidden_state(hidden_state[idx], done)
                
            policy.reset_hedge_state(done)
            past_dones=done
            if render:
                envs.render()
            if episodes is not None and episodes_finished >= episodes:
                break

        return_history = utils.call_async_env(envs, "return_history")
        special_history = utils.call_async_env(envs, "special_history")
        return hidden_state, (return_history, special_history)
