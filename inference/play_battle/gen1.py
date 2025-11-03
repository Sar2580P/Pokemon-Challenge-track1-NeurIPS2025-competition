from custom.utils import get_modal_stuff
import modal
import pathlib 

app, img= get_modal_stuff(app_name=f"Metamon_evaluation_gen1")
volume = modal.Volume.from_name(f"pokemon-showdown-gen1", create_if_missing=True)
VOL_MOUNT_PATH = pathlib.Path("/vol")

@app.function(
    image=img,
    gpu="L4", 
    timeout=6 * 60 * 60,  # 6 hour timeout
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("poke-username"),
        modal.Secret.from_name("poke-passwd"),
    ],
    volumes={VOL_MOUNT_PATH: volume, 
             },
)
def evaluate_model(*arglist):
    
    
    # from custom.evaluate import add_cli , _run_default_evaluation
    from inference.evaluate import (add_cli, _run_default_evaluation, 
                                    create_sarpanch, create_tribe_population)
    from argparse import ArgumentParser
    import os
    import subprocess 
    import time  
    from inference.population_config import GEN1_Models
    
    policy_tribe = create_tribe_population(members_config=GEN1_Models)
    kabeela_sardar = create_sarpanch(
        model_gin_config_path="inference/configs/models/moe_tribe.gin", 
        train_gin_config_path="inference/configs/training/binary_rl.gin",  # just for no purpose
        gen=1
    )
    kabeela_sardar.policy_aclr.inference_entities = policy_tribe
     
    # Initialize the kabeela sardar
    
    parser = ArgumentParser(
        description="Evaluate a pretrained Metamon model by playing battles against opponents. "
        "This script allows you to evaluate a pretrained model's performance against a set of "
        "heuristic baselines, local ladder, or the PokéAgent Challenge ladder. It can also save replays in the same format "
        "as the human replay dataset for further training."
    )
    add_cli(parser)
    args = parser.parse_args(arglist)
    
    REPLAY_DIR = f"Kabeele_ka_Sardar| {args.eval_type}"
    args.save_trajectories_to=str(VOL_MOUNT_PATH/os.path.join(args.save_trajectories_to, REPLAY_DIR))
    
    if args.eval_type in ("pokeagent", "ladder"):
        args.username=os.getenv("POKEMON_USERNAME")
        args.password=os.getenv("POKEMON_PASSWD")
    
    elif args.eval_type=="heuristic":
        # 3. Define the path to the server directory
        #    Assuming it's in the root of your project
        server_dir = "/root/server/pokemon-showdown"
        print("Setting up the local Pokémon Showdown server...")

        # 4. Install dependencies (blocking call, as this must finish first)
        print("Running npm install...")
        install_process = subprocess.run(
            ["npm", "install"],
            cwd=server_dir,
            capture_output=True,
            text=True
        )
        if install_process.returncode != 0:
            print("npm install failed!")
            print(install_process.stderr)
            raise RuntimeError("Failed to install server dependencies.")
        print("npm install successful.")

        # 5. Start the server (non-blocking call to run in the background)
        print("Starting server in the background...")
        server_process = subprocess.Popen(
            ["node", "pokemon-showdown", "start", "--no-security"],
            cwd=server_dir,
        )
        
        # 6. IMPORTANT: Wait a few seconds for the server to initialize
        print("Waiting for server to start...")
        time.sleep(5)
        print("Server should be running. Starting evaluation...")
    
    os.makedirs(args.save_trajectories_to, exist_ok=True)
    print(f"🚀 Starting the battle... Kursi ki peti baandh lo...")
    _run_default_evaluation(args, agent=kabeela_sardar)
    
    volume.commit()