
# from custom.utils import get_modal_stuff_evaluation
# import modal
# import pathlib
# import os
# import requests # For the verification step

# # --- Configuration ---
# # The IP is fetched from an environment variable for security and flexibility.
# EXIT_NODE_IP = os.environ.get("MY_EXIT_NODE_IP", "100.107.166.30") # Fallback for local testing
# # IMPORTANT: The proxy URL now uses the http:// scheme
# PROXY_URL = "http://localhost:1080"
# YEAR = 2025

# # --- Modal App Setup ---
# app, img = get_modal_stuff_evaluation(app_name=f"Metamon_evaluation_gen9ou_{YEAR}")
# volume = modal.Volume.from_name(f"pokemon-showdown-gen9ou_{YEAR}", create_if_missing=True)
# VOL_MOUNT_PATH = pathlib.Path("/vol")

# @app.function(
#     image=img,
#     gpu="L4",
#     timeout=6 * 60 * 60,
#     secrets=[
#         modal.Secret.from_name("tailscale-auth"),
#         modal.Secret.from_name("wandb-secret"),
#         modal.Secret.from_name("poke-username"),
#         modal.Secret.from_name("poke-passwd"),
#         # Pass the exit node IP to the container's environment.
#         modal.Secret.from_dict({"EXIT_NODE_IP": EXIT_NODE_IP}),
#         # Set the standard HTTP proxy environment variables.
#         # This is the key change that all libraries will understand.
#         modal.Secret.from_dict({
#             "HTTP_PROXY": PROXY_URL,
#             "HTTPS_PROXY": PROXY_URL,
#         }),
#     ],
#     volumes={VOL_MOUNT_PATH: volume},
# )
# def evaluate_model(*arglist):

#     # --- NO MORE SOCKS5 SOCKET PATCHING! ---
#     # We have removed the 'import socket', 'import socks',
#     # and the two lines that configured the proxy in Python.
#     # The environment variables set above handle this automatically now.

#     # --- Verification Step ---
#     # The requests library automatically uses the HTTP_PROXY/HTTPS_PROXY env vars.
#     try:
#         print("Verifying public IP from Python (using HTTP proxy)...")
#         response = requests.get("https://ip.me", timeout=15)
#         public_ip = response.text.strip()
#         print(f"✅ Python is using public IP: {public_ip}")
#     except requests.exceptions.RequestException as e:
#         print(f"❌ Could not verify public IP from Python. Error: {e}")
#         raise

#     # --- Your Main Evaluation Logic ---
#     from custom.evaluate import add_cli, _run_default_evaluation
#     from argparse import ArgumentParser
    
#     print("\nStarting model evaluation...")
#     parser = ArgumentParser(description="Evaluate a pretrained Metamon model.")
#     add_cli(parser)
#     args = parser.parse_args(arglist)
    
#     full_ckpts=[]
#     for ckpt in args.checkpoints:
#         full_ckpts.append(str(VOL_MOUNT_PATH/ckpt))
#     args.checkpoints=full_ckpts
#     args.save_trajectories_to = str(VOL_MOUNT_PATH / args.save_trajectories_to)
    
#     if args.eval_type == "pokeagent":
#         args.username = os.getenv("POKEMON_USERNAME")
#         args.password = os.getenv("POKEMON_PASSWD")
#         print(f"Using Pokemon Username: {args.username}")
    
#     os.makedirs(args.save_trajectories_to, exist_ok=True)
#     _run_default_evaluation(args)
    
#     volume.commit()


from custom.utils import get_modal_stuff_evaluation
import modal
import pathlib 
import subprocess # NEW: For running Tailscale commands
import time       # NEW: For pausing to let the tunnel stabilize
import os         # For accessing environment variables

# --- Configuration Constants (Replace with your actual IP) ---
EXIT_NODE_IP = "100.107.166.30" 
PROXY_URL = "socks5://localhost:1080"

YEAR=2025
app, img= get_modal_stuff_evaluation(app_name=f"Metamon_evaluation_gen9ou_{YEAR}")
volume = modal.Volume.from_name(f"pokemon-showdown-gen9ou_{YEAR}", create_if_missing=True)
VOL_MOUNT_PATH = pathlib.Path("/vol")


@app.function(
    image=img,
    gpu="L4", 
    timeout=6 * 60 * 60,  # 6 hour timeout
    secrets=[
        modal.Secret.from_name("tailscale-auth"), # Corrected name
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("poke-username"),
        modal.Secret.from_name("poke-passwd"),
        modal.Secret.from_dict({"EXIT_NODE_IP": EXIT_NODE_IP}), 
        # Also include the proxy vars for Python libraries like requests (optional but helpful)
        modal.Secret.from_dict(
            {
                "ALL_PROXY": PROXY_URL,
                "HTTP_PROXY": PROXY_URL,
                "HTTPS_PROXY": PROXY_URL,
            }
        ),
    ],
    volumes={VOL_MOUNT_PATH: volume},
)
def evaluate_model(*arglist):
    
    import socket
    import socks
    
    print("Configuring Python to use SOCKS5 proxy (via Tailscale Exit Node)...")
    socks.set_default_proxy(socks.SOCKS5, "127.0.0.1", 1080)
    socket.socket = socks.socksocket
    
    
    # from metamon.rl.evaluate import add_cli, _run_default_evaluation
    from custom.evaluate import add_cli, _run_default_evaluation
    from argparse import ArgumentParser
    
    parser = ArgumentParser(
        description="Evaluate a pretrained Metamon model by playing battles against opponents. "
        # ... (rest of description)
    )
    add_cli(parser)
    args = parser.parse_args(arglist)
    full_ckpts=[]
    for ckpt in args.checkpoints:
        full_ckpts.append(str(VOL_MOUNT_PATH/ckpt))
    args.checkpoints=full_ckpts
    args.save_trajectories_to=str(VOL_MOUNT_PATH/args.save_trajectories_to)
    
    if args.eval_type=="pokeagent":
        # POKEMON_USERNAME/POKEMON_PASSWD should be loaded from the poke-username/poke-passwd secrets
        args.username=os.getenv("POKEMON_USERNAME")
        args.password=os.getenv("POKEMON_PASSWD")
        print(f"Using Pokemon Username: {args.username}")
        # print(args.password) # Don't print passwords
    
    os.makedirs(args.save_trajectories_to, exist_ok=True)
    _run_default_evaluation(args)
    
    volume.commit()
    
    # --- TAILSCALE CLEANUP (Optional, but good practice) ---
    print("--- Tailscale Cleanup ---")
    subprocess.run(["tailscale", "down"], check=True)
