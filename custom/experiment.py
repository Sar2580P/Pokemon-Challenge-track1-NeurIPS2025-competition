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
from einops import repeat
from amago.loading import (
    Batch,
    RLDataset,
    RLData_pad_collate,
    MAGIC_PAD_VAL,
)
from amago.envs import EnvCreator
from tqdm import tqdm
from amago.envs import SequenceWrapper, ReturnHistory, SpecialMetricHistory, EnvCreator
import numpy as np 


@gin.configurable
@dataclass
class CustomExperiment(MetamonAMAGOExperiment):
    ckpt_vol:Any=None
    start_epoch:int=0
    # opponent modeling specific params
    cvae_kl_loss_weights:float= 2
    cvae_recons_loss_weights:float = 3
    selector_loss_weights : float = 5
    
    # knowledge distillation specific params
    kd_action_kl_weight:float=0.1
    kd_critic_weight:float=1.0
    
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
        
        # set the start epoch
        self.epoch=self.start_epoch
        self.epochs=self.epochs+self.start_epoch
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
        # assert isinstance(self.agent, HRM_MultiTaskAgent)
        optimizer = self.init_optimizer(self.agent)
        lr_schedule = utils.get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=self.lr_warmup_steps
        )
        self.policy_aclr, self.optimizer, self.lr_schedule = self.accelerator.prepare(
            self.agent, optimizer, lr_schedule
        )
        self.accelerator.register_for_checkpointing(self.lr_schedule)
        self.grad_update_counter = 0
        

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
            hidden_state = policy.traj_encoder.init_hidden_state(
                self.parallel_actors, self.DEVICE
            )

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
        for _ in iter_:
            with torch.no_grad():
                with self.caster():
                    actions, hidden_state = policy.get_actions(
                        obs=obs,
                        rl2s=rl2s,
                        time_idxs=time_idxs,
                        sample=self.sample_actions,
                        hidden_state=hidden_state,
                    )
            next_obs, reward, terminated, truncated, info = envs.step(actions.squeeze(1).cpu().numpy())
            
            done = terminated | truncated
            if done.ndim == 2:
                done = done.squeeze(1)
            if done.any():
                if save_on_done:
                    utils.call_async_env(envs, "save_finished_trajs")
                episodes_finished += done.sum()
            obs, rl2s, time_idxs = get_t()
            hidden_state = policy.traj_encoder.reset_hidden_state(hidden_state, done)

            if render:
                envs.render()
            if episodes is not None and episodes_finished >= episodes:
                break
        return_history = utils.call_async_env(envs, "return_history")
        special_history = utils.call_async_env(envs, "special_history")
        return hidden_state, (return_history, special_history)

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
        

    def compute_loss_opponent_modeling(self, batch: Batch, log_step: bool, 
                                       curr_step:int , total_steps:int) -> dict:
        """Core computation of the actor and critic RL loss terms from a `Batch` of data.

        Args:
            batch: The batch of data.
            log_step: Whether to compute extra metrics for wandb logging.

        Returns:
            dict: loss terms and any logging metrics. "Actor Loss", "Critic Loss", "Sequence
                Length", "Batch Size (in Timesteps)", "Unmasked Batch Size (in Timesteps)" are
                always provided. Additional keys are determined by what is logged in the
                Agent.forward method.
        """
        # Agent.forward handles most of the work
        (critic_loss, actor_loss , KL_loss, 
         recons_loss, vae_value_loss) = self.policy_aclr.forward_opponent_modeling(batch, log_step=log_step, 
                                                                                   curr_step=curr_step , total_steps=total_steps)
        update_info = self.policy.update_info
        B, L_1, G, _ = actor_loss.shape
        C = len(self.policy.critics)

        # mask sequence losses
        state_mask = (~((batch.rl2s == MAGIC_PAD_VAL).all(-1, keepdim=True))).bool()
        critic_state_mask = repeat(state_mask[:, 1:, ...], f"B L 1 -> B L {C} {G} 1")
        actor_state_mask = repeat(state_mask[:, 1:, ...], f"B L 1 -> B L {G} 1")
        # hook to allow custom masks
        actor_state_mask = self.edit_actor_mask(batch, actor_loss, actor_state_mask)
        critic_state_mask = self.edit_critic_mask(batch, critic_loss, critic_state_mask)
        batch_size = B * L_1
        unmasked_batch_size = actor_state_mask[..., 0, 0].sum()
        masked_actor_loss = utils.masked_avg(actor_loss, actor_state_mask)
        if isinstance(critic_loss, torch.Tensor):
            masked_critic_loss = utils.masked_avg(critic_loss, critic_state_mask)
        else:
            assert critic_loss is None
            masked_critic_loss = 0.0

        # all of this is logged
        return {
            "Critic Loss": masked_critic_loss,
            "Actor Loss": masked_actor_loss,
            "CVAE_KL_loss": KL_loss, 
            "CVAE_recons_loss": recons_loss, 
            "Subgoal_selector_value_loss": vae_value_loss, 
            "Sequence Length": L_1 + 1,
            "Batch Size (in Timesteps)": batch_size,
            "Unmasked Batch Size (in Timesteps)": unmasked_batch_size,
        } | update_info

    def compute_loss_KD(self, batch: Batch, log_step: bool, allow_actor_critic_adaptation:bool) -> dict:
        traj_recons_loss, action_kl_loss, critic_loss = self.policy_aclr.forward_traj_encoder_KD(batch, log_step=log_step,
                                                                                                 allow_actor_critic_adaptation=allow_actor_critic_adaptation)
        update_info = self.policy.update_info
        # all of this is logged
        return {
            "traj_recons_loss": traj_recons_loss,
            "action_kl_loss": action_kl_loss,
            "critic_loss": critic_loss
        } | update_info

    def train_step_opponent_modeling(self, batch: Batch, log_step: bool, curr_step:int , total_steps:int):
        """Take a single training step on a `batch` of data.

        Args:
            batch: The batch of data.
            log_step: Whether to compute extra metrics for wandb logging.

        Returns:
            dict: metrics from the compute_loss method.
        """
        with self.accelerator.accumulate(self.policy_aclr):
            self.optimizer.zero_grad()
            l = self.compute_loss_opponent_modeling(batch, log_step=log_step, 
                                                    curr_step=curr_step , total_steps=total_steps)
            loss = (l["Actor Loss"] +   
                    self.critic_loss_weight * l["Critic Loss"]+
                    self.cvae_kl_loss_weights* l["CVAE_KL_loss"]+
                    self.cvae_recons_loss_weights* l["CVAE_recons_loss"]+
                    self.selector_loss_weights * l["Subgoal_selector_value_loss"]
                    )
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.policy_aclr.parameters(), self.grad_clip
                )
                self.policy.soft_sync_targets()
                self.grad_update_counter += 1
                if log_step:
                    l.update(
                        {"Learning Rate": self.lr_schedule.get_last_lr()[0]}
                        | self.get_grad_norms()
                    )
            self.optimizer.step()
            self.lr_schedule.step()
        return l


    def train_step_KD_modeling(self, batch: Batch, log_step: bool, allow_actor_critic_adaptation:bool):
        with self.accelerator.accumulate(self.policy_aclr):
            self.optimizer.zero_grad()
            l = self.compute_loss_KD(batch, log_step=log_step, 
                                     allow_actor_critic_adaptation=allow_actor_critic_adaptation)
            loss = (l["traj_recons_loss"]+
                    l["action_kl_loss"]*self.kd_action_kl_weight+
                    l["critic_loss"]*self.kd_critic_weight
                    )
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.policy_aclr.parameters(), self.grad_clip
                )
                self.policy.soft_sync_targets()
                self.grad_update_counter += 1
                if log_step:
                    l.update(
                        {"Learning Rate": self.lr_schedule.get_last_lr()[0]}
                        | self.get_grad_norms()
                    )
            self.optimizer.step()
            self.lr_schedule.step()
        return l

    def learn_opponent_modeling(self) -> None:
        """Main training loop for the experiment.

        For every epoch, we:
            1. Load the latest policy checkpoint if `always_load_latest` is True.
            2. Evaluate the policy on the validation set if `val_interval` is not None and the
                current epoch is divisible by `val_interval`.
            3. Collect new training data if `train_timesteps_per_epoch` is not None and the
                current epoch >= to `start_collecting_at_epoch`.
            4. Train the policy on the training data for `train_batches_per_epoch` batches if
                `self.dataset.ready_for_training` is True.
            5. Save the policy checkpoint if `ckpt_interval` is not None and the current epoch
                is divisible by `ckpt_interval`.
            6. Write the latest policy checkpoint if `always_save_latest` is True.

        Experiment be configured so that processes do some or all of the above. For example, an
        actor process might only do steps 1, 2, and 3, while a learner process might only do
        steps 4, 5, and 6.
        """

        def make_pbar(loader, epoch_num):
            if self.verbose:
                return tqdm(
                    enumerate(loader),
                    desc=f"{self.run_name} Epoch {epoch_num} Train",
                    total=self.train_batches_per_epoch,
                    colour="green",
                )
            else:
                return enumerate(loader)

        start_epoch = self.epoch
        total_epochs= self.epochs-start_epoch
        for epoch in range(start_epoch, self.epochs):
            if self.always_load_latest:
                self.read_latest_policy()

            # environment interaction
            self.policy_aclr.eval()
            if (
                self.val_interval
                and epoch % self.val_interval == 0
                and self.val_timesteps_per_epoch > 0
            ):
                self.evaluate_val()
            if (
                epoch >= self.start_collecting_at_epoch
                and self.train_timesteps_per_epoch > 0
            ):
                self.collect_new_training_data()
            self.accelerator.wait_for_everyone()

            dset_log = self.dataset.on_end_of_collection(experiment=self)
            self.log(dset_log, key="dataset")
            self.init_dloaders()
            if not self.dataset.ready_for_training:
                utils.amago_warning(
                    f"Skipping training on epoch {epoch} because `dataset.ready_for_training` is False"
                )
                continue

            # training
            elif epoch < self.start_learning_at_epoch:
                continue
            if self.train_batches_per_epoch > 0:
                self.policy_aclr.train()
                for train_step, batch in make_pbar(self.train_dloader, epoch):
                    total_step = (epoch * self.train_batches_per_epoch) + train_step
                    log_step = total_step % self.log_interval == 0
                    loss_dict = self.train_step_opponent_modeling(batch, log_step=log_step, 
                                                                  curr_step=total_step , 
                                                                  total_steps=total_epochs*self.train_batches_per_epoch)
                    if log_step:
                        self.log(loss_dict, key="train-update")
            self.accelerator.wait_for_everyone()
            del self.train_dloader

            # end epoch
            self.epoch = epoch
            if self.ckpt_interval and epoch % self.ckpt_interval == 0:
                self.save_checkpoint()
            if self.always_save_latest:
                self.write_latest_policy()

    def learn_vae_prior(self) -> None:
        """Main training loop for the experiment.

        For every epoch, we:
            1. Load the latest policy checkpoint if `always_load_latest` is True.
            2. Evaluate the policy on the validation set if `val_interval` is not None and the
                current epoch is divisible by `val_interval`.
            3. Collect new training data if `train_timesteps_per_epoch` is not None and the
                current epoch >= to `start_collecting_at_epoch`.
            4. Train the policy on the training data for `train_batches_per_epoch` batches if
                `self.dataset.ready_for_training` is True.
            5. Save the policy checkpoint if `ckpt_interval` is not None and the current epoch
                is divisible by `ckpt_interval`.
            6. Write the latest policy checkpoint if `always_save_latest` is True.

        Experiment be configured so that processes do some or all of the above. For example, an
        actor process might only do steps 1, 2, and 3, while a learner process might only do
        steps 4, 5, and 6.
        """

        def make_pbar(loader, epoch_num):
            if self.verbose:
                return tqdm(
                    enumerate(loader),
                    desc=f"{self.run_name} Epoch {epoch_num} Train",
                    total=self.train_batches_per_epoch,
                    colour="green",
                )
            else:
                return enumerate(loader)

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.epochs):
            # if (
            #     self.val_interval
            #     and epoch % self.val_interval == 0
            #     and self.val_timesteps_per_epoch > 0
            # ):
            #     self.evaluate_val()
            # if (
            #     epoch >= self.start_collecting_at_epoch
            #     and self.train_timesteps_per_epoch > 0
            # ):
            #     self.collect_new_training_data()
            # self.accelerator.wait_for_everyone()

            dset_log = self.dataset.on_end_of_collection(experiment=self)
            self.log(dset_log, key="dataset")
            self.init_dloaders()
            if not self.dataset.ready_for_training:
                utils.amago_warning(
                    f"Skipping training on epoch {epoch} because `dataset.ready_for_training` is False"
                )
                continue

            # training
            elif epoch < self.start_learning_at_epoch:
                continue
            if self.train_batches_per_epoch > 0:
                for train_step, batch in make_pbar(self.train_dloader, epoch):
                    total_step = (epoch * self.train_batches_per_epoch) + train_step
                    log_step = total_step % self.log_interval == 0
                    loss_dict = self.agent.forward_vae_prior_training(batch=batch, log_step=log_step, save_dir=self.ckpt_dir)
                    if log_step:
                        self.log(loss_dict, key="train-update")
            self.accelerator.wait_for_everyone()
            del self.train_dloader


    def learn_traj_encoder_KD(self, allow_actor_critic_adaptation:bool=True) -> None:
        """Main training loop for the experiment.

        For every epoch, we:
            1. Load the latest policy checkpoint if `always_load_latest` is True.
            2. Evaluate the policy on the validation set if `val_interval` is not None and the
                current epoch is divisible by `val_interval`.
            3. Collect new training data if `train_timesteps_per_epoch` is not None and the
                current epoch >= to `start_collecting_at_epoch`.
            4. Train the policy on the training data for `train_batches_per_epoch` batches if
                `self.dataset.ready_for_training` is True.
            5. Save the policy checkpoint if `ckpt_interval` is not None and the current epoch
                is divisible by `ckpt_interval`.
            6. Write the latest policy checkpoint if `always_save_latest` is True.

        Experiment be configured so that processes do some or all of the above. For example, an
        actor process might only do steps 1, 2, and 3, while a learner process might only do
        steps 4, 5, and 6.
        """

        def make_pbar(loader, epoch_num):
            if self.verbose:
                return tqdm(
                    enumerate(loader),
                    desc=f"{self.run_name} Epoch {epoch_num} Train",
                    total=self.train_batches_per_epoch,
                    colour="green",
                )
            else:
                return enumerate(loader)

        start_epoch = self.epoch
        for epoch in range(start_epoch, self.epochs):
            # if (
            #     self.val_interval
            #     and epoch % self.val_interval == 0
            #     and self.val_timesteps_per_epoch > 0
            # ):
            #     self.evaluate_val()
            # if (
            #     epoch >= self.start_collecting_at_epoch
            #     and self.train_timesteps_per_epoch > 0
            # ):
            #     self.collect_new_training_data()
            # self.accelerator.wait_for_everyone()

            dset_log = self.dataset.on_end_of_collection(experiment=self)
            self.log(dset_log, key="dataset")
            self.init_dloaders()
            if not self.dataset.ready_for_training:
                utils.amago_warning(
                    f"Skipping training on epoch {epoch} because `dataset.ready_for_training` is False"
                )
                continue

            # training
            elif epoch < self.start_learning_at_epoch:
                continue
            if self.train_batches_per_epoch > 0:
                for train_step, batch in make_pbar(self.train_dloader, epoch):
                    total_step = (epoch * self.train_batches_per_epoch) + train_step
                    log_step = total_step % self.log_interval == 0
                    loss_dict = self.train_step_KD_modeling(batch=batch, log_step=log_step, 
                                                            allow_actor_critic_adaptation=allow_actor_critic_adaptation)
                    if log_step:
                        self.log(loss_dict, key="train-update")
            self.accelerator.wait_for_everyone()
            del self.train_dloader
            # end epoch
            self.epoch = epoch
            if self.ckpt_interval and epoch % self.ckpt_interval == 0:
                self.save_checkpoint()
            if self.always_save_latest:
                self.write_latest_policy()
                
    def load_checkpoint_from_path(
        self, path: str, is_accelerate_state: bool = True
    ) -> None:
        """Load a checkpoint from a given path.

        Args:
            path: Full path to the checkpoint fle to load.
            is_accelerate_state: Whether the checkpoint is a full accelerate state (True) or
                pytorch weights only (False). Defaults to True.
        """
        if not is_accelerate_state:
            ckpt = utils.retry_load_checkpoint(path, map_location=self.DEVICE)
            model_dict = self.policy.state_dict()
            # 1. Filter the loaded checkpoint.
            #    Keep a key 'k' only if it exists in the current model's state_dict.
            filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_dict}
            
            # Optional: You can check if the shapes match too, though PyTorch does this automatically.
            # filtered_ckpt = {k: v for k, v in ckpt.items() if k in model_dict and v.shape == model_dict[k].shape}

            # 2. Load the perfectly filtered state dictionary
            self.policy.load_state_dict(filtered_ckpt)
        else:
            self.accelerator.load_state(path)