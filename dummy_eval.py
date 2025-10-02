
from custom.utils import get_modal_stuff
import modal
import pathlib 
import subprocess # NEW: For running Tailscale commands
import time       # NEW: For pausing to let the tunnel stabilize
import os         # For accessing environment variables

YEAR=2025
app, img= get_modal_stuff(app_name=f"Metamon_evaluation_gen9ou_{YEAR}")
volume = modal.Volume.from_name(f"pokemon-showdown-gen9ou_{YEAR}", create_if_missing=True)
VOL_MOUNT_PATH = pathlib.Path("/vol")


@app.function(
    image=img,
    gpu="A10", 
    timeout=6 * 60 * 60,  # 6 hour timeout
    secrets=[
        modal.Secret.from_name("tailscale-auth"), # Corrected name
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("poke-username"),
        modal.Secret.from_name("poke-passwd"),
    ],
    volumes={VOL_MOUNT_PATH: volume},
)
def evaluate_model(*arglist):
    POKEMON_USERNAME=os.getenv("POKEMON_USERNAME")
    POKEMON_PASSWD=os.getenv("POKEMON_PASSWD")
    
    command = [
    "python",
    "-m", "metamon.rl.evaluate",
    "--eval_type", "pokeagent",
    "--agent", "SyntheticRLV2",
    "--gens", "1",
    "--formats", "ou",
    "--total_battles", "10",
    "--username", POKEMON_USERNAME,  # Dynamically inserted secret
    "--password", POKEMON_PASSWD,  # Dynamically inserted secret
    "--team_set", "competitive",
    # Use an absolute path if required to write results to the mounted volume
        # "--output_dir", str(VOL_MOUNT_PATH / "results"), 
    ]
    
    # Log the command being executed (excluding passwords for safety in logs)
    display_command = ' '.join(command).replace(POKEMON_PASSWD, '********')
    print(f"--- Starting Evaluation ---")
    print(f"Executing: {display_command}")
    
    # 3. Execute the command using subprocess.run
    try:
        # check=True raises a CalledProcessError if the command returns a non-zero exit code (failure)
        result = subprocess.run(
            command,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT, # Merge stdout and stderr for combined logging
            text=True
        )
        
        print(f"\n--- Command Completed Successfully ---")
        print(result.stdout)
        
    except subprocess.CalledProcessError as e:
        print(f"\n--- Command Failed (Exit Code {e.returncode}) ---")
        print(f"Output/Error Log:\n{e.output}")
        # Re-raise the exception to signal failure in the Modal run
        raise
    
    print("--- Evaluation Finished ---")