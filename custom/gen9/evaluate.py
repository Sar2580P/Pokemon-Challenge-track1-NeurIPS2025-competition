
# from custom.utils import get_modal_stuff_evaluation
# import modal
# import pathlib 

# YEAR=2025
# app, img= get_modal_stuff_evaluation(app_name=f"Metamon_evaluation_gen9ou_{YEAR}")
# volume = modal.Volume.from_name(f"pokemon-showdown-gen9ou_{YEAR}", create_if_missing=True)
# VOL_MOUNT_PATH = pathlib.Path("/vol")


# @app.function(
#     image=img,
#     gpu="L4", 
#     timeout=6 * 60 * 60,  # 6 hour timeout
#     secrets=[
#         modal.Secret.from_name("tailscale-auth"),
#         modal.Secret.from_name("wandb-secret"),
#         modal.Secret.from_name("poke-username"),
#         modal.Secret.from_name("poke-passwd"),
#     ],
#     volumes={VOL_MOUNT_PATH: volume, 
#              },
# )
# def evaluate_model(*arglist):
# # if __name__=="__main__":
#     # from custom.evaluate import add_cli , _run_default_evaluation
#     from metamon.rl.evaluate import add_cli, _run_default_evaluation
#     from argparse import ArgumentParser
#     import os

#     parser = ArgumentParser(
#         description="Evaluate a pretrained Metamon model by playing battles against opponents. "
#         "This script allows you to evaluate a pretrained model's performance against a set of "
#         "heuristic baselines, local ladder, or the PokéAgent Challenge ladder. It can also save replays in the same format "
#         "as the human replay dataset for further training."
#     )
#     add_cli(parser)
#     args = parser.parse_args(arglist)
#     # args.checkpoints=[str(VOL_MOUNT_PATH/ ckpt) for ckpt in args.checkpoints]
#     args.save_trajectories_to=str(VOL_MOUNT_PATH/args.save_trajectories_to)
    
#     if args.eval_type=="pokeagent":
#         args.username=os.getenv("POKEMON_USERNAME")
#         args.password=os.getenv("POKEMON_PASSWD")
#         print(args.username)
#         print(args.password)
    
#     os.makedirs(args.save_trajectories_to, exist_ok=True)
#     _run_default_evaluation(args)
    
#     volume.commit()


# Assuming this is in your Python file containing the Modal stub:

from custom.utils import get_modal_stuff_evaluation
import modal
import pathlib 
import subprocess # NEW: For running Tailscale commands
import time       # NEW: For pausing to let the tunnel stabilize
import os         # For accessing environment variables

# --- Configuration Constants (Replace with your actual IP) ---
EXIT_NODE_IP = "100.107.183.63" 
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
    
    
    from metamon.rl.evaluate import add_cli, _run_default_evaluation
    from argparse import ArgumentParser
    
    parser = ArgumentParser(
        description="Evaluate a pretrained Metamon model by playing battles against opponents. "
        # ... (rest of description)
    )
    add_cli(parser)
    args = parser.parse_args(arglist)
    
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
    # ---


# import subprocess
# import time

# @app.function(
#     image=img,
#     gpu="L4", 
#     timeout=6 * 60 * 60,  # 6 hour timeout
#     secrets=[
#         modal.Secret.from_name("wandb-secret"),
#         modal.Secret.from_name("poke-username"),
#         modal.Secret.from_name("poke-passwd"),
#         # Add your two new VPN secrets here
#         modal.Secret.from_name("my-vpn-config"),
#         modal.Secret.from_name("my-vpn-credentials"),
#     ],
#     volumes={VOL_MOUNT_PATH: volume},
# )
# def evaluate_model(*arglist):
#     from custom.evaluate import add_cli , _run_default_evaluation
#     from argparse import ArgumentParser
#     import os

#     # =================== START VPN DEBUGGING SETUP ===================
#     print("🚀 Starting VPN setup in DEBUG mode...")
    
#     # 1. Create the .ovpn and auth files from secrets
#     ovpn_content = os.environ['OVPN_CONFIG'] 
#     with open("config.ovpn", "w") as f:
#         f.write(ovpn_content)
    
#     print(os.environ['VPN_USERNAME'],"\n\n\n", os.environ['VPN_PASSWORD'])
    
#     with open("vpn_auth.txt", "w") as f:
#         f.write(f"{os.environ['VPN_USERNAME']}\n")
#         f.write(f"{os.environ['VPN_PASSWORD']}\n")

#     # 2. Start OpenVPN and CAPTURE its output
#     print("Launching OpenVPN and capturing logs...")
#     vpn_process = subprocess.Popen([
#         "openvpn",
#         "--config", "config.ovpn",
#         "--auth-user-pass", "vpn_auth.txt",
#     ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) # Capture output

#     print("Waiting 10 seconds for VPN to connect or fail...")
#     time.sleep(10)

#     # 3. Check the process and print any logs/errors
#     print("="*20 + " OpenVPN Logs " + "="*20)
#     stdout, stderr = vpn_process.communicate(timeout=5) # Read output
#     if stdout:
#         print("--- Standard Output ---")
#         print(stdout)
#     if stderr:
#         print("--- Standard Error ---")
#         print(stderr)
#     print("="*54)

#     # 4. Terminate the process
#     vpn_process.terminate()

#     # 5. Verify the IP address
#     print("\nVerifying external IP address...")
#     subprocess.run(["curl", "-s", "ifconfig.me"])
#     print("\nVPN setup complete!")
#     # =================== END VPN SETUP =====================
    
    
#     # Your original evaluation code starts here
#     parser = ArgumentParser(
#         description="Evaluate a pretrained Metamon model by playing battles against opponents."
#     )
#     add_cli(parser)
#     args = parser.parse_args(arglist)
#     args.checkpoints=[str(VOL_MOUNT_PATH/ ckpt) for ckpt in args.checkpoints]
#     args.save_trajectories_to=str(VOL_MOUNT_PATH/args.save_trajectories_to)
    
#     if args.eval_type=="pokeagent":
#         args.username=os.getenv("POKEMON_USERNAME")
#         args.password=os.getenv("POKEMON_PASSWD")
    
#     os.makedirs(args.save_trajectories_to, exist_ok=True)
#     _run_default_evaluation(args)
    
#     volume.commit()


# @app.function(
#     image=img,
#     gpu="L4", 
#     timeout=6 * 60 * 60,
#     secrets=[
#         modal.Secret.from_name("wandb-secret"),
#         modal.Secret.from_name("poke-username"),
#         modal.Secret.from_name("poke-passwd"),
#         modal.Secret.from_name("wireguard-config"),  # WireGuard config
#     ],
#     volumes={VOL_MOUNT_PATH: volume},
# )
# def evaluate_model(*arglist):
#     import subprocess
#     import os
#     import time
    
#     print("🚀 Setting up WireGuard (userspace)...")
    
#     # Create WireGuard config from secret
#     wg_config = os.environ['WG_CONFIG']
#     with open("/tmp/wg0.conf", "w") as f:
#         f.write(wg_config)
    
#     # Use wireguard-go (userspace implementation)
#     subprocess.Popen(["wireguard-go", "wg0"])
#     time.sleep(2)
    
#     # Configure interface
#     subprocess.run(["wg", "setconf", "wg0", "/tmp/wg0.conf"], check=True)
    
#     # Add to your image build:
#     # apt-get install -y wireguard-tools wireguard-go
    
#     # Verify IP
#     result = subprocess.run(["curl", "-s", "ifconfig.me"], capture_output=True, text=True)
#     print(f"New IP: {result.stdout}")
    
#     # Your evaluation code...