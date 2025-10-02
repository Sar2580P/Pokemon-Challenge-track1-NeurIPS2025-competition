
from custom.utils import get_modal_stuff
import modal
import socket
import time

import asyncio
import websockets

import os
import itertools

def list_dirs_up_to_level(root_path, max_level):
    return [
        os.path.join(dirpath, d) 
        for dirpath, dirnames, filenames in os.walk(root_path) 
        for d in dirnames
        if dirpath.count(os.sep) - root_path.count(os.sep) < max_level
    ]

async def websocket_smoke_test(port):
    """Attempts to connect to the server using WebSockets."""
    # NOTE: The path '/showdown/websocket' is a common default for Showdown servers.
    uri = f"ws://localhost:{port}/showdown/websocket"
    print(f"🔬 Performing WebSocket smoke test on {uri}...")
    try:
        # We set a common origin header that the server likely expects.
        async with websockets.connect(
            uri,
            origin=f"http://localhost:{port}",
            open_timeout=10
        ) as websocket:
            print("✅ WebSocket smoke test PASSED. Connection established!")
            # We can even try to receive the first message
            message = await asyncio.wait_for(websocket.recv(), timeout=10)
            print(f"Received initial message: {message[:100]}...")
    except Exception as e:
        print("❌ WebSocket smoke test FAILED.")
        print(f"Error: {e}")
        # Re-raise the exception to stop the script
        raise


def wait_for_server(host: str, port: int, timeout: int = 30):
    """
    Waits for a network port to become available.

    Args:
        host: The host to check (e.g., 'localhost').
        port: The port to check.
        timeout: The maximum time to wait in seconds.
    """
    start_time = time.monotonic()
    while True:
        try:
            with socket.create_connection((host, port), timeout=1):
                print(f"✅ Server is up and listening on port {port}!")
                return
        except (ConnectionRefusedError, socket.timeout):
            if time.monotonic() - start_time >= timeout:
                raise TimeoutError(f"Server on {host}:{port} did not start within {timeout} seconds.")
            print(f"⏳ Server not ready yet. Retrying...")
            time.sleep(1)

app, img= get_modal_stuff(app_name="Metamon training")
import pathlib

volume = modal.Volume.from_name("pokemon-showdown-vol", create_if_missing=True)
VOL_MOUNT_PATH = pathlib.Path("/vol")

@app.function(
    image=img,
    # gpu="T4", 
    timeout=6 * 60 * 60,  # 6 hour timeout
    secrets=[
        modal.Secret.from_name("wandb-secret"),
        modal.Secret.from_name("ssh-secret")
    ],
    volumes={VOL_MOUNT_PATH: volume},
)
def train_model():
# if __name__ == "__main__": 
    import subprocess     
    import os
    os.environ["METAMON_CACHE_DIR"] = str(VOL_MOUNT_PATH / "PAC-dataset")
    PORT = 8000
    SERVER_DIRECTORY = "server/pokemon-showdown"

    # Start the server process (as before)
    print("🚀 Starting Pokémon Showdown server...")
    command = ["node", "pokemon-showdown", "start", "--no-security"]
    server_process = subprocess.Popen(command, cwd=SERVER_DIRECTORY)
    print(f"✅ Server process started with PID: {server_process.pid}")
    try:
        wait_for_server('localhost', PORT, timeout=30)
    except TimeoutError as e:
        print(f"❌ {e}")
        
    print("🔬 Performing a smoke test with curl...")
    try:
        # We expect this to either succeed or fail fast.
        # The `-I` flag just gets the headers, which is faster.
        curl_command = ["curl", "-I", f"http://localhost:{PORT}"]
        result = subprocess.run(curl_command, timeout=10, check=True, capture_output=True, text=True)
        print("✅ Smoke test PASSED. Server is responding to HTTP requests.")
        print(f"Server Headers:\n{result.stdout}")
    except Exception as e:
        print("❌ Smoke test FAILED. The server is not responding correctly.")
        print(f"Error: {e}")
        if hasattr(e, 'stderr'):
            print(f"Curl Stderr: {e.stderr}")
            
    try:
        # --- NEW: Run the WebSocket smoke test ---
        asyncio.run(websocket_smoke_test(PORT))
    except Exception:
        # The function already printed the error, now we stop.
        return
            
    print("🤖 Starting the main training process...")
    
    # adding the usage stats to the volume
    # need to run this command to download the usage-stats --> python -m metamon.data.download usage-stats
    print(list_dirs_up_to_level('/vol', 5))
    
    if not os.path.exists(os.environ["METAMON_CACHE_DIR"]):
        subprocess.run(["python", "-m", "metamon.data.download", "usage-stats"])
        volume.commit()
        print("✅ Usage stats downloaded.")
    else: print("ℹ️ Usage stats already present, skipping download.")
    print("\n\n", list_dirs_up_to_level('/vol', 5), "\n\n")
    # run this command to check if the server is up --> python -m metamon.env
    
    subprocess.run(["python", "-m", "metamon.env"])
    