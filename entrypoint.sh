#!/bin/sh

# Custom entrypoint used to login into Tailscale and start SOCKS5 proxy
# AND configure the Exit Node using the EXIT_NODE_IP environment variable.

set -e

# NOTE: We use the EXIT_NODE_IP variable passed from the Modal function's secrets.

# Start the Tailscale daemon with SOCKS5 proxy and the Exit Node configured.
tailscaled \
    --tun=userspace-networking \
    --socks5-server=localhost:1080 \
    --outbound-http-proxy-listen=localhost:1080 &
    # --- CRITICAL ADDITION START ---
    # The --exit-node and --exit-node-allow-lan-access flags are needed here.
    # The hostname and authkey are set on the 'tailscale up' line below.
    # --- CRITICAL ADDITION END ---

# Connect the Tailscale client and set the Exit Node route.
tailscale up \
    --authkey=${TAILSCALE_AUTHKEY} \
    --hostname=${MODAL_TASK_ID} \
    --exit-node=${EXIT_NODE_IP} \
    --exit-node-allow-lan-access=true
    # Note: We pass the Exit Node configuration to the 'up' command, 
    # not the 'tailscaled' daemon command, for best practice.

# Loop until the maximum number of retries is reached
retry_count=0
MAX_RETRIES=5 # Added MAX_RETRIES for clarity

while [ $retry_count -lt $MAX_RETRIES ]; do
    # Check connectivity through the SOCKS5 proxy
    http_status=$(curl -x socks5://localhost:1080 -o /dev/null -L -s -w '%{http_code}' https://www.google.com)

    if [ "$http_status" -eq 200 ]; then
        echo "Successfully started Tailscale, SOCKS5 proxy, and set Exit Node."
        # Use 'exec "$@"' to run the actual Python function command
        exec "$@" 
    else
        echo "Attempt $((retry_count+1))/$MAX_RETRIES failed: SOCKS5 proxy returned HTTP $http_status"
    fi

    retry_count=$((retry_count+1))
    sleep 1
done

echo "Failed to start Tailscale and confirm Exit Node connectivity."
exit 1





# # Custom entrypoint to start Tailscale and route all traffic through an exit node.

# set -e

# # 1. Start the Tailscale daemon with an HTTP proxy.
# #    We REMOVED the --socks5-server flag and are ONLY using the HTTP proxy.
# tailscaled \
#     --tun=userspace-networking \
#     --outbound-http-proxy-listen=localhost:1080 &

# # Give the daemon a moment to initialize to prevent race conditions.
# sleep 3

# # 2. Connect to the Tailscale network using the 'up' command.
# #    This is where we specify the exit node and auth key.
# tailscale up \
#     --authkey=${TAILSCALE_AUTHKEY} \
#     --hostname=${MODAL_TASK_ID} \
#     --exit-node=${EXIT_NODE_IP} \
#     --exit-node-allow-lan-access=false # Set to 'false' for better security

# # 3. Health Check: Loop until the exit node connection is verified via the HTTP proxy.
# MAX_RETRIES=10
# echo "Verifying connection through HTTP exit node proxy..."

# for i in $(seq 1 $MAX_RETRIES); do
#     # Use the HTTP proxy flag for curl: --proxy http://...
#     public_ip=$(curl -s --proxy http://localhost:1080 https://ip.me)
    
#     # Check if we got any IP address back.
#     if [ -n "$public_ip" ]; then
#         echo "✅ Success! Container is using Public IP: $public_ip"
#         # Run the main command passed to the container (your python script).
#         exec "$@"
#     fi
#     echo "Attempt $i/$MAX_RETRIES: Connection not ready. Retrying in 3 seconds..."
#     sleep 3
# done

# echo "❌ Error: Failed to establish a connection through the Tailscale exit node."
# exit 1