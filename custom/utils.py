import os
from dotenv import load_dotenv
from huggingface_hub import snapshot_download
from omegaconf import OmegaConf

def get_env_var(var_name: str, default: str = None) -> str:
    load_dotenv()
    """Get the environment variable or return a default value."""
    return os.getenv(var_name, default)


def visualize_model(model, input_data, file_name, save_dir="media"):
    from torchview import draw_graph
    draw_graph(
        model,
        input_data=input_data,
        graph_name=file_name,
        save_graph=True,
        filename=file_name,
        directory=save_dir,
        expand_nested=True,
        graph_dir='TB',  # Top to Bottom
        hide_module_functions=False,
        hide_inner_tensors=False,
        roll=False,
        show_shapes=True,
        depth=3
    )
    print("graph saved successfully...")
        
def read_yaml(file_path):
    conf = OmegaConf.load(file_path)
    config = OmegaConf.create(OmegaConf.to_yaml(conf, resolve=True))
    return config

def download_hf_ckpt(allow_patterns="medium-rl/ckpts/policy_weights/*40.pt"):
    # Download a specific model variant (e.g., 'large-il')
    from metamon import METAMON_CACHE_DIR

    model_path = snapshot_download(
        repo_id="jakegrigsby/metamon",
        allow_patterns=allow_patterns,  # Replace 'large-il' with your desired model
        local_dir=f"{METAMON_CACHE_DIR}/metamon_models",  # Local directory to save files
        local_dir_use_symlinks=False
    )

def get_modal_stuff(app_name:str="Modal training"):
    import modal
    packages=read_yaml('environment.yml')['dependencies'][-1]['pip']


    image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            list(packages)+['torch']   # gpu based version
        )
        .apt_install("git",  "screen",  "npm",  "nodejs") 
        .run_commands("mkdir /root/amago/" , "git clone https://github.com/UT-Austin-RPL/amago.git /root/amago/")
        .run_commands("cd /root/amago && pip install -e .")
        .apt_install("curl")
        .pip_install("tqdm")
        .run_commands("mkdir /root/metamon2/" , "git clone --recursive https://github.com/Sar2580P/metamon-personal.git /root/metamon2/")
        .run_commands( "mv /root/metamon2/* root")
        .run_commands("cd /root/ && pip install -e .")

        .env({"METAMON_WANDB_PROJECT": "PAC-dataset"})
        .env({"METAMON_WANDB_ENTITY": "PAC-dataset"})
        .env({"METAMON_CACHE_DIR": "/vol/PAC-dataset"})
        .add_local_dir(
            local_path="~/Pokemon-Challenge-track1/custom", remote_path="/root/custom"
        )
        .add_local_file(local_path="~/Pokemon-Challenge-track1/environment.yml", remote_path="/root/environment.yml")        
        
    )

    app = modal.App(app_name, image=image)
    return app, image


def get_modal_stuff_evaluation(app_name:str="Modal training"):
    import modal
    from pathlib import Path
    packages=read_yaml('environment.yml')['dependencies'][-1]['pip']
    HOME_DIR = Path.home()/"Pokemon-Challenge-track1"

    image = (
        modal.Image.debian_slim(python_version="3.10")
        .pip_install(
            list(packages)+['torch']   # gpu based version
        )
        .apt_install("curl")
        .run_commands(
        "curl -fsSL https://tailscale.com/install.sh | sh",
        "tailscale version" # Optional: confirms installation worked
        )
        .apt_install("git",  "screen",  "npm",  "nodejs") 
        .run_commands("mkdir /root/amago/" , "git clone https://github.com/UT-Austin-RPL/amago.git /root/amago/")
        .run_commands("cd /root/amago && pip install -e .")
        .pip_install("tqdm")
        .run_commands("mkdir /root/metamon2/" , "git clone --recursive https://github.com/Sar2580P/metamon-personal.git /root/metamon2/")
        .run_commands( "mv /root/metamon2/* root")
        .run_commands("cd /root/ && pip install -e .")
        
        
        .add_local_file(HOME_DIR/"entrypoint.sh", "/root/entrypoint.sh", copy=True)
        .run_commands("chmod a+x /root/entrypoint.sh")
        .pip_install("PySocks")
        
        # 3. SET THE ENTRYPOINT TO RUN THE TAILSCALE SCRIPT FIRST
        # This replaces the default startup command. Your Python function call 
        # will be passed as arguments to this script.
        .entrypoint(["/root/entrypoint.sh"])
        
        .env({"METAMON_WANDB_PROJECT": "PAC-dataset"})
        .env({"METAMON_WANDB_ENTITY": "PAC-dataset"})
        .env({"METAMON_CACHE_DIR": "/vol/PAC-dataset"})
        .add_local_dir(
            local_path=HOME_DIR/"custom", remote_path="/root/custom"
        )
        .add_local_file(local_path=HOME_DIR/"environment.yml", remote_path="/root/environment.yml")

    )

    app = modal.App(app_name, image=image)
    return app, image



def add_cli(parser):
    parser.add_argument(
        "--run_name",
        required=True,
        help="Give the run a name to identify logs and checkpoints.",
    )
    parser.add_argument(
        "--obs_space",
        type=str,
        default="TeamPreviewObservationSpace",
        help="See the README for a description of the different observation spaces.",
    )
    parser.add_argument(
        "--reward_function",
        type=str,
        default="DefaultShapedReward",
        help="See the README for a description of the different reward functions.",
    )
    parser.add_argument(
        "--action_space",
        type=str,
        default="DefaultActionSpace",
        help="See the README for a description of the different action spaces.",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        required=True,
        help="Path to save checkpoints. Find checkpoints under {save_dir}/{run_name}/ckpts/",
    )
    parser.add_argument(
        "--ckpt",
        type=int,
        default=None,
        help="Resume training from an existing run with this run_name. Provide the epoch checkpoint to load.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of epochs to train for. In offline RL model, an epoch is an arbitrary interval (here: 25k) of training steps on a fixed dataset.",
    )
    parser.add_argument(
        "--batch_size_per_gpu",
        type=int,
        default=12,
        help="Batch size per GPU. Total batch size is batch_size_per_gpu * num_gpus.",
    )
    parser.add_argument(
        "--grad_accum",
        type=int,
        default=1,
        help="Number of gradient accumulations per update.",
    )
    parser.add_argument(
        "--model_gin_config",
        type=str,
        required=True,
        help="Path to a gin config file that edits the model architecture. See provided rl/configs/models/",
    )
    parser.add_argument(
        "--train_gin_config",
        type=str,
        required=True,
        help="Path to a gin config file that edits the training or hparams. See provided rl/configs/training/",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="DefaultObservationSpace-v1",
        help="The tokenizer to use for the text observation space. See metamon.tokenizer for options.",
    )
    parser.add_argument(
        "--dloader_workers",
        type=int,
        default=10,
        help="Number of workers for the data loader.",
    )
    parser.add_argument(
        "--parsed_replay_dir",
        type=str,
        default=None,
        help="Path to the parsed replay directory. Defaults to the official huggingface version.",
    )
    parser.add_argument(
        "--custom_replay_dir",
        type=str,
        default=None,
        help="Path to an optional second parsed replay dataset (e.g., self-play data you've collected).",
    )
    parser.add_argument(
        "--custom_replay_sample_weight",
        type=float,
        default=0.25,
        help="[0, 1] portion of each batch to sample from the custom dataset (if provided).",
    )
    parser.add_argument(
        "--async_env_mp_context",
        type=str,
        default="forkserver",
        help="Async environment setup method. Options: 'forkserver' (recommended, fast), 'fork' (fastest but unsafe with threads), 'spawn' (slowest but safest). Use 'spawn' only if others hang.",
    )
    parser.add_argument(
        "--eval_gens",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4, 9],
        help="Generations (of OU) to play against heuristics between training epochs. Win rates usually saturate at 90\%+ quickly, so this is mostly a sanity-check. Reduce gens to save time on launch!",
    )
    parser.add_argument(
        "--formats",
        nargs="+",
        default=None,
        help="Showdown battle formats to include in the dataset. Defaults to all supported formats.",
    )
    parser.add_argument(
        "--ckpt_interval",
        type=int,
        default=2,
        help="Number of epochs between saving model checkpoints.",
    )
    parser.add_argument(
        "--log_interval",
        type=int,
        default=300,
        help="The no. of steps to wait before each log.",
    )
    parser.add_argument(
        "--val_interval",
        type=int,
        default=None,
        help="Number of epochs between validations. Default None skips validation entirely.",
    )
    parser.add_argument(
        "--steps_per_epoch",
        type=int,
        default=500,
        help="steps_per_epoch",
    )
    parser.add_argument(
        "--verbose",
        type=bool,
        default=True,
        help="verbose",
    )
    
    
    parser.add_argument("--log", action="store_true", help="Log to wandb.")
    return parser
