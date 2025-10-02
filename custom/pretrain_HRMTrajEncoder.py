from metamon.rl.metamon_to_amago import MetamonTstepEncoder
from amago.nets.traj_encoders import TformerTrajEncoder
from custom.traj_encoder import HRMTrajEncoder
import torch.nn as nn
from metamon.rl.train import create_offline_dataset
from torch.utils.data import DataLoader
from amago.loading import (
    Batch,
    RLDataset,
    RLData_pad_collate,
    MAGIC_PAD_VAL,
)
import gin
from accelerate import Accelerator, DistributedDataParallelKwargs
import torch
from tqdm import tqdm 
import  amago.utils as utils
import os 
import contextlib
import torch.nn.functional as F
from metamon.tokenizer.tokenizer import get_tokenizer
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

MetamonTstepEncoder_params={
    "obs_space":TeamPreviewObservationSpace(),
    "rl2_space":2,
    "tokenizer": get_tokenizer("DefaultObservationSpace-v1"),
    "extra_emb_dim": 18,
    "d_model": 160,
    "n_layers": 5,
    "n_heads": 8,
    "scratch_tokens": 11,
    "numerical_tokens": 6,
    "token_mask_aug": True,
    "dropout": 0.05
}

HRM_traj_encoder_params={}

Tformer_traj_encoder_params= {
    "TformerTrajEncoder": {
        "n_layers": 9,
        "n_heads": 20,
        "d_ff": 5120,
        "d_model": 1280,
        "normformer_norms": True,
        "sigma_reparam": True,
        "norm": "layer",
        "head_scaling": True,
        "activation": "leaky_relu"
    }
}

dataset_params={}
optim_args = dict(lr=2, weight_decay=3)

class Backbone(nn.Module):
    def __init__(self, tstep_config: dict, traj_config:dict, 
                 traj_class:TformerTrajEncoder|HRMTrajEncoder, 
                 ckpt_path:str=None, is_teacher:bool=False):
        super().__init__()
        self.tstep_encoder=MetamonTstepEncoder(**tstep_config)   # see how it is initialized in the metamon code
        self.traj_encoder = traj_class(**traj_config)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.ckpt_path=ckpt_path
        self.is_teacher=is_teacher
        
    def initialize_tstep_encoder(self):
        ckpt=torch.load(self.ckpt_path,map_location=self.device)
        tstep_encoder_state_dict={k:v for k,v in ckpt.items() if k.startswith('tstep_encoder')}
        self.tstep_encoder.load_state_dict(tstep_encoder_state_dict)

    def initialize_traj_encoder(self):
        assert isinstance(self.traj_encoder, TformerTrajEncoder)
        ckpt=torch.load(self.ckpt_path,map_location=self.device)
        tformer_encoder_state_dict={k:v for k,v in ckpt.items() if k.startswith('traj_encoder')}
        self.traj_encoder.load_state_dict(tformer_encoder_state_dict)
        
    @classmethod
    def get_backbone(cls, is_teacher:bool=False) -> 'Backbone':
        if is_teacher:
            backbone=cls(tstep_config=MetamonTstepEncoder_params, 
                            traj_config=Tformer_traj_encoder_params, 
                            traj_class=TformerTrajEncoder, is_teacher=is_teacher)
            backbone.initialize_tstep_encoder()
            backbone.initialize_traj_encoder()
            backbone.eval()    
        else: 
            backbone=cls(tstep_config=MetamonTstepEncoder_params, is_teacher=is_teacher,
                            traj_config=HRM_traj_encoder_params, traj_class=HRMTrajEncoder)
            backbone.initialize_tstep_encoder()
            backbone.train()
        return backbone
    
    def save_checkpoint(self, ckpt_path:str):
        assert not self.is_teacher, "Invalid setup, teacher is being saved rather than student(learner)"
        torch.save(self.traj_encoder.state_dict(), ckpt_path)
        print(f'saved HRM traj encoder to -> {ckpt_path}')
     
        
    def forward(self, batch: Batch, log_step: bool): 
        self.update_info = {}  # holds wandb stats
        active_log_dict = self.update_info if log_step else None
        o = self.tstep_encoder(obs=batch.obs, rl2s=batch.rl2s, log_dict=active_log_dict)
        
        s_rep, hidden_state = self.traj_encoder(seq=o, time_idxs=batch.time_idxs, 
                                                hidden_state=None, log_dict=active_log_dict)
        D_emb = self.traj_encoder.emb_dim
        B, L, D_o = o.shape
        assert s_rep.shape == (B, L, D_emb)
        return s_rep
        
# hrm_backbone=Backbone(tstep_config=MetamonTstepEncoder_params, 
#                           traj_config=HRM_traj_encoder_params, traj_class=HRMTrajEncoder)
# tformer_backbone=Backbone(tstep_config=MetamonTstepEncoder_params, traj_config=Tformer_traj_encoder_params, 
#                           traj_class=TformerTrajEncoder)

# # initialize from ckpt
# hrm_backbone.initialize_tstep_encoder()
# tformer_backbone.initialize_tstep_encoder()
# tformer_backbone.initialize_traj_encoder()

from argparse import ArgumentParser
from metamon.rl.train import add_cli, create_offline_rl_trainer
from metamon.rl.metamon_to_amago import (
    MetamonAMAGOExperiment,
    MetamonAMAGODataset,
    make_baseline_env,
    make_placeholder_env,
)
from metamon.tokenizer import get_tokenizer
from metamon.interface import (
    get_observation_space,
    get_reward_function,
    get_action_space,
)

parser = ArgumentParser(
    description="Train a Metamon RL agent from scratch using offline RL on parsed replay data. "
    "This script trains new models using imitation learning or reinforcement learning objectives "
    "on the dataset of human Pokémon battles (& an optional custom dataset of self-play data you've collected)."
)
add_cli(parser)
args = parser.parse_args()


# agent input/output/rewards
obs_space = TokenizedObservationSpace(
    get_observation_space(args.obs_space), get_tokenizer(args.tokenizer)
)
reward_function = get_reward_function(args.reward_function)
action_space = get_action_space(args.action_space)

WANDB_PROJECT = os.environ.get("METAMON_WANDB_PROJECT")
WANDB_ENTITY = os.environ.get("METAMON_WANDB_ENTITY")


# metamon dataset
amago_dataset = create_offline_dataset(
    obs_space=obs_space,
    action_space=action_space,
    reward_function=reward_function,
    parsed_replay_dir=args.parsed_replay_dir,
    custom_replay_dir=args.custom_replay_dir,
    custom_replay_sample_weight=args.custom_replay_sample_weight,
    formats=args.formats,
)

experiment = create_offline_rl_trainer(
        ckpt_dir=args.save_dir,
        run_name=args.run_name,
        model_gin_config=args.model_gin_config,
        train_gin_config=args.train_gin_config,
        obs_space=obs_space,
        action_space=action_space,
        reward_function=reward_function,
        amago_dataset=amago_dataset,
        eval_gens=args.eval_gens,
        async_env_mp_context=args.async_env_mp_context,
        dloader_workers=args.dloader_workers,
        epochs=args.epochs,
        grad_accum=args.grad_accum,
        batch_size_per_gpu=args.batch_size_per_gpu,
        log=args.log,
        wandb_project=WANDB_PROJECT,
        wandb_entity=WANDB_ENTITY,
    )
experiment.start()
print(experiment.rl2_space, "   ihihiooidwqw")

class Trainer:
    def __init__(self, dataset_params:dict, batch_size:int, num_workers:int ,
                 batches_per_update:int, mixed_precision:str="no", epoch:int=0, total_epochs:int=25, 
                 start_learning_at_epoch:int=0, train_batches_per_epoch:int=100, val_interval: int=5, val_timesteps_per_epoch:int=5, 
                 log_interval:int=2, verbose:bool=False, log_to_wandb: bool=True, wandb_group_name: str = None, run_name:str="HRM_knowledge_distillation",
                 optim_args:dict=None,  lr_warmup_steps:int=4, ckpt_interval: int=3, ckpt_base_dir: str="results/ckpts", 
                 ):
        super().__init__()
        self.batch_size=batch_size
        self.num_workers=num_workers
        self.batches_per_update=batches_per_update
        self.mixed_precision=mixed_precision
        self.dataset_params=dataset_params
                
        self.epoch=epoch
        self.total_epochs=total_epochs
        self.start_learning_at_epoch=start_learning_at_epoch
        self.train_batches_per_epoch=train_batches_per_epoch
        
        self.val_interval=val_interval
        self.val_timesteps_per_epoch=val_timesteps_per_epoch
        
        self.verbose=verbose
        self.run_name=run_name
        self.log_interval=log_interval
        self.log_to_wandb=log_to_wandb
        self.wandb_group_name=wandb_group_name
        self.wandb_project: str = os.environ.get("AMAGO_WANDB_PROJECT")
        self.wandb_entity: str = os.environ.get("AMAGO_WANDB_ENTITY")
        
        self.ckpt_interval=ckpt_interval
        self.ckpt_dir=os.path.join(ckpt_base_dir, self.run_name, "ckpts")
        self.log_dir=os.path.join(ckpt_base_dir, self.run_name, "logs")
        os.makedirs(self.ckpt_dir)
        os.makedirs(self.log_dir)
        
        self.optim_args=optim_args
        self.lr_warmup_steps=lr_warmup_steps
        self.accelerator=Accelerator(
                                gradient_accumulation_steps=self.batches_per_update,
                                device_placement=True,
                                log_with="wandb",
                                kwargs_handlers=[
                                    DistributedDataParallelKwargs(find_unused_parameters=True)
                                ],
                                mixed_precision=self.mixed_precision,
                            )
        self.train_dloader=self.init_dloaders()
        self.init_logger()
        self.teacher, self.student=Backbone.get_backbone(is_teacher=True), Backbone.get_backbone()
        optimizer = self.init_optimizer(self.student.traj_encoder)
        lr_schedule = utils.get_constant_schedule_with_warmup(
            optimizer=optimizer, num_warmup_steps=self.lr_warmup_steps
        )
        self.student_aclr, self.optimizer, self.lr_schedule = self.accelerator.prepare(
            self.student, optimizer, lr_schedule
        )
        self.accelerator.register_for_checkpointing(self.lr_schedule)
        self.grad_update_counter = 0

    def init_optimizer(self, model: nn.Module) -> torch.optim.Optimizer:
        return torch.optim.AdamW(model.trainable_params, **self.optim_args)
       
    def init_dloaders(self) -> DataLoader:
        """Create pytorch dataloaders to batch trajectories in parallel."""
        self.dataset=create_offline_dataset(**self.dataset_params)

        train_dloader = DataLoader(
            self.dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            collate_fn=RLData_pad_collate,
            pin_memory=True,
        )
        return self.accelerator.prepare(train_dloader)


    def compute_loss(self, batch: Batch, log_step: bool) -> torch.Tensor:
        with torch.no_grad():
            y_true = self.teacher(batch, log_step)
        y_pred=self.student(batch, log_step)
        loss = F.mse_loss(y_pred, y_true)
        return loss

    def train_step(self, batch: Batch, log_step: bool):
        """Take a single training step on a `batch` of data.

        Args:
            batch: The batch of data.
            log_step: Whether to compute extra metrics for wandb logging.

        Returns:
            dict: metrics from the compute_loss method.
        """
        with self.accelerator.accumulate(self.student_aclr):
            self.optimizer.zero_grad()
            loss = self.compute_loss(batch, log_step=log_step)
            self.accelerator.backward(loss)
            if self.accelerator.sync_gradients:
                self.accelerator.clip_grad_norm_(
                    self.student_aclr.parameters(), self.grad_clip
                )
                self.student.soft_sync_targets()
                self.grad_update_counter += 1
                if log_step:
                    log={
                        "MSE_loss": loss.detach().item(), 
                        "lr": self.lr_schedule.get_lr()
                    }
            self.optimizer.step()
            self.lr_schedule.step()
        return log

    def caster(self):
        """Get the context manager for mixed precision training."""
        if self.mixed_precision != "no":
            return torch.autocast(device_type="cuda")
        else:
            return contextlib.suppress()

    @property
    def model(self) -> Backbone:
        """Returns the current Agent policy free from the accelerator wrapper."""
        return self.accelerator.unwrap_model(self.student_aclr)

    def init_logger(self) -> None:
        """Configure log dir and wandb compatibility."""
        gin_config = gin.operative_config_str()
        config_path = os.path.join(self.log_dir, "config.txt")
        with open(config_path, "w") as f:
            f.write(gin_config)
        if self.log_to_wandb:
            # records the gin config on the wandb dashboard
            gin_as_wandb = utils.gin_as_wandb_config()
            self.accelerator.init_trackers(
                project_name=self.wandb_project,
                config=gin_as_wandb,
                init_kwargs={
                    "wandb": dict(
                        entity=self.wandb_entity,
                        dir=self.log_dir,
                        name=self.run_name,
                        group=self.wandb_group_name,
                    )
                },
            )


    def log(
        self, metrics_dict: dict[str, torch.Tensor | int | float], key: str
    ) -> None:
        """Log a dict of metrics to the `key` panel of the wandb console alongisde current
        x-axis metrics."""
        log_dict = {}
        for k, v in metrics_dict.items():
            if isinstance(v, torch.Tensor):
                if v.ndim == 0:
                    log_dict[k] = v.detach().cpu().float().item()
            else:
                log_dict[k] = v

        if self.log_to_wandb:
            wandb_dict = {
                f"{key}/{subkey}": val for subkey, val in log_dict.items()
            } | self.x_axis_metrics()
            self.accelerator.log(wandb_dict)

    def learn(self) -> None:
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
        for epoch in range(start_epoch, self.total_epochs):
            # if self.always_load_latest:         ☀️
            #     self.read_latest_policy()

            # environment interaction
            self.student_aclr.eval()
            # if (                                ☀️
            #     self.val_interval
            #     and epoch % self.val_interval == 0
            #     and self.val_timesteps_per_epoch > 0
            # ):
            #     self.evaluate_val()
            # if (                                ☀️
            #     epoch >= self.start_collecting_at_epoch
            #     and self.train_timesteps_per_epoch > 0
            # ):
            #     self.collect_new_training_data()
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
            elif epoch < self.start_learning_at_epoch:continue
            if self.train_batches_per_epoch > 0:
                self.student_aclr.train()
                for train_step, batch in make_pbar(self.train_dloader, epoch):
                    total_step = (epoch * self.train_batches_per_epoch) + train_step
                    log_step = total_step % self.log_interval == 0
                    log_dict = self.train_step(batch, log_step=log_step)
                    if log_step:
                        self.log(log_dict, key="train-update")
            self.accelerator.wait_for_everyone()
            del self.train_dloader

            # end epoch
            self.epoch = epoch
            if self.ckpt_interval and epoch % self.ckpt_interval == 0:
                
                ckpt_name = f"epoch={self.epoch}.pt"
                ckpt_path=os.path.join(self.ckpt_dir, ckpt_name)
                self.student.save_checkpoint(ckpt_path=ckpt_path)
