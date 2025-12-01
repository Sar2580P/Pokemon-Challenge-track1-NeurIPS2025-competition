from custom.utils import get_modal_stuff
import modal
import pathlib

app, img= get_modal_stuff(app_name=f"Metamon_training_gen1")
volume = modal.Volume.from_name(f"pokemon-showdown-gen1", create_if_missing=True)
VOL_MOUNT_PATH = pathlib.Path("/vol")

@app.function(
    image=img,
    gpu="A10", 
    timeout=6 * 60 * 60,  # 6 hour timeout
    secrets=[
        modal.Secret.from_name("wandb-secret"),
    ],
    volumes={VOL_MOUNT_PATH: volume, 
             },
)
def train_model(*arglist):
# if __name__ == "__main__":      
    import os
    from  os.path import join as osp
    import wandb
    from argparse import ArgumentParser
    from functools import partial
    from typing import List, Optional, Any
    import subprocess
    from metamon import METAMON_CACHE_DIR
    from metamon.interface import (
        get_observation_space,
        get_reward_function,
        get_action_space,
    )
    
    from metamon.interface import (
        TokenizedObservationSpace,
        ActionSpace,
        RewardFunction,
    )
    from metamon.tokenizer import get_tokenizer
    from custom.utils import get_env_var
    from custom.experiment import CustomExperiment

    import amago
    from metamon.env import get_metamon_teams
    from metamon.interface import (
        TokenizedObservationSpace,
        ActionSpace,
        RewardFunction,
    )
    from metamon.tokenizer import get_tokenizer
    from metamon.data import ParsedReplayDataset
    from metamon.rl.metamon_to_amago import (
        MetamonAMAGOExperiment,
        MetamonAMAGODataset,
        make_baseline_env,
        make_placeholder_env,
    )
    from metamon import baselines
    from metamon.rl.train import  create_offline_dataset
    from custom.utils import add_cli
    from custom.gen9.replay_data_sampling import download_gen9ou_subset_parsed_replays
    
    parser = ArgumentParser(
        description="Train a Metamon RL agent from scratch using offline RL on parsed replay data. "
        "This script trains new models using imitation learning or reinforcement learning objectives "
        "on the dataset of human Pokémon battles (& an optional custom dataset of self-play data you've collected)."
    )
    add_cli(parser)
    args = parser.parse_args(arglist)   # arglist
    # import os
    # os.environ["ACCELERATE_USE_TORCH_COMPILE"] = "false"
    # print(os.environ["ACCELERATE_USE_TORCH_COMPILE"])
            
    print("🤖 Starting the main training process...")

    EVAL_OPPONENTS = [
        baselines.heuristic.basic.PokeEnvHeuristic,
        baselines.heuristic.basic.Gen1BossAI,
        baselines.heuristic.basic.Grunt,
        baselines.heuristic.basic.GymLeader,
        baselines.heuristic.kaizo.EmeraldKaizo,
    ]
    
    def create_offline_rl_trainer(
        ckpt_dir: str,
        ckpt_volume:Any , 
        run_name: str,
        model_gin_config: str,
        train_gin_config: str,
        obs_space: TokenizedObservationSpace,
        action_space: ActionSpace,
        reward_function: RewardFunction,
        amago_dataset: amago.loading.Dataset,
        eval_gens: List[int] = [1, 2, 3, 4, 9],
        async_env_mp_context: str = "spawn",
        dloader_workers: int = 8,
        epochs: int = 40,
        ckpt_interval: int = 2, 
        val_interval: int= 1,    
        grad_accum: int = 1,
        steps_per_epoch: int = 25_000,
        batch_size_per_gpu: int = 16,
        log: bool = False,
        verbose:bool=True, 
        log_interval: int = 300,
        manual_gin_overrides: Optional[dict] = None,
    ):
        """
        Convenience function that creates an AMAGO experiment with default arguments
        set for offline RL in metamon.
        """
        # configuration
        config = {f"MetamonTstepEncoder.tokenizer": obs_space.tokenizer}
        if manual_gin_overrides is not None:
            config.update(manual_gin_overrides)
        model_config_path = os.path.join(model_gin_config)
        training_config_path = os.path.join(train_gin_config)
        amago.cli_utils.use_config(config, [model_config_path, training_config_path])

        # validation environments (evaluated throughout training)
        make_envs = [
            partial(
                make_baseline_env,
                battle_format=f"gen{gen}ou",
                observation_space=obs_space,
                action_space=action_space,
                reward_function=reward_function,
                team_set=get_metamon_teams(f"gen{gen}ou", "competitive"),
                opponent_type=opponent,
            )
            for gen in set(eval_gens)
            for opponent in EVAL_OPPONENTS
        ]
        experiment = CustomExperiment(
            ## required ##
            run_name=run_name,
            ckpt_base_dir=ckpt_dir,
            ckpt_vol=ckpt_volume, 
            # max_seq_len = should be set in the gin file
            dataset=amago_dataset,
            # tstep_encoder_type = should be set in the gin file
            # traj_encoder_type = should be set in the gin file
            # agent_type = should be set in the gin file
            val_timesteps_per_epoch=300,  # per actor
            ## environment ##
            make_train_env=partial(make_placeholder_env, obs_space, action_space),
            make_val_env=make_envs,
            env_mode="async",
            async_env_mp_context=async_env_mp_context,
            parallel_actors=len(make_envs),
            # no exploration
            exploration_wrapper_type=None,
            sample_actions=True,
            force_reset_train_envs_every=None,
            ## logging ##
            log_to_wandb=log,
            # wandb_project=wandb_project,
            # wandb_entity=wandb_entity,
            verbose=verbose,
            log_interval=log_interval,
            ## replay ##
            padded_sampling="none",
            dloader_workers=dloader_workers,
            ## learning schedule ##
            epochs=epochs,
            # entirely offline RL
            start_learning_at_epoch=0,
            start_collecting_at_epoch=float("inf"),
            train_timesteps_per_epoch=0,
            train_batches_per_epoch=steps_per_epoch * grad_accum,
            val_interval=val_interval,
            ckpt_interval=ckpt_interval,
            ## optimization ##
            batch_size=batch_size_per_gpu,
            batches_per_update=grad_accum,
        )
        return experiment
    
    # agent input/output/rewards
    obs_space = TokenizedObservationSpace(
        get_observation_space(args.obs_space), get_tokenizer(args.tokenizer)
    )
    reward_function = get_reward_function(args.reward_function)
    action_space = get_action_space(args.action_space)
        
    amago_dataset = create_offline_dataset(
        obs_space=obs_space,
        action_space=action_space,
        reward_function=reward_function,
        parsed_replay_dir=osp(METAMON_CACHE_DIR, args.parsed_replay_dir) if args.parsed_replay_dir else None,
        custom_replay_dir=osp(METAMON_CACHE_DIR, args.custom_replay_dir) if args.custom_replay_dir else None,
        custom_replay_sample_weight=args.custom_replay_sample_weight,
        formats=args.formats,
    )
    volume.commit()
    print(f"✅ Dataset setup complete... ")

    experiment = create_offline_rl_trainer(
        ckpt_dir=str(VOL_MOUNT_PATH /args.save_dir) ,
        ckpt_volume=volume,  #ckpt_volume, 
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
        ckpt_interval=args.ckpt_interval,
        val_interval=args.val_interval,
        log_interval=args.log_interval ,
        steps_per_epoch=args.steps_per_epoch,
        verbose=args.verbose
    )
    print(f"✅ Experiment setup complete... ")
    
    if not os.path.exists(os.environ["METAMON_CACHE_DIR"]):
        print(f"Downloading metamon dataset ... ", end="\t")
        subprocess.run(["python", "-m", "metamon.data.download", "usage-stats"])
        volume.commit()
        print("✅")
    else: print("ℹ️ Metamon dataset already present, skipping download.")
    
    experiment.start()

    if args.ckpt is not None:
        # resume training from a checkpoint
        experiment.load_checkpoint(args.ckpt)
    print(f"✅ Checkpoint loaded... ")
    print(f"🚀 Starting training for {args.epochs} epochs...")
    experiment.learn_vae_prior()
    wandb.finish()
    