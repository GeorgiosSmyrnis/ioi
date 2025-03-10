# Credits to Anton Lozhkov
#!/bin/bash
#SBATCH --job-name=r1-router
#SBATCH --partition=hopper-cpu
#SBATCH --qos=high
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=1875m
#SBATCH --output=./logs/%x_%j_%n.out
#SBATCH --error=./logs/%x_%j_%n.err
#SBATCH --time=30-00:00:00
#SBATCH --requeue

set -exuo pipefail

# Configuration variables
ROUTER_PORT=39876
SERVER_PORT=39877  # Must match the server script
HEALTH_CHECK_TIMEOUT=10  # Timeout for health checks (seconds)
UV_ENV=/fsx/hynek_kydlicek/projects/ioi-leaderboard/ioi-eval

trap 'scontrol requeue ${SLURM_JOB_ID}; exit 15' SIGUSR1

# Environment setup
source ~/.bashrc
source $UV_ENV/bin/activate

# Start the router
python -m sglang_router.launch_router \
    --port "$ROUTER_PORT" \
    --host 0.0.0.0 \
    --worker-startup-timeout-secs 300 &

ROUTER_PID=$!

# Wait for router to start
sleep 10
if ! curl -s -o /dev/null "http://localhost:${ROUTER_PORT}/health"; then
    echo "Router failed to start"
    kill $ROUTER_PID
    exit 1
fi

echo "Router started successfully on port $ROUTER_PORT"
echo "Scanning for running r1-server instances..."

# Get a list of r1-server job IDs and register their servers
while IFS= read -r jobid; do
    [[ -z "$jobid" ]] && continue

    # Use scontrol to get the nodelist for this job
    nodelist=$(scontrol show job "$jobid" | grep NodeList | tail -n1 | grep -oP 'NodeList=ip[^ ]+')
    [[ -z "$nodelist" ]] && continue
    nodelist=${nodelist#NodeList=}

    # Get first node from the nodelist
    first_node=$(scontrol show hostnames "$nodelist" | head -n1)
    [[ -z "$first_node" ]] && continue
    [[ "$first_node" == "(null)" ]] && continue

    # Convert hostname to IP format
    server_ip=$(echo "$first_node" | sed -E 's/ip-([0-9]+)-([0-9]+)-([0-9]+)-([0-9]+)/\1.\2.\3.\4/')
    server_url="http://${server_ip}:${SERVER_PORT}"

    echo "Found server node: $first_node (${server_ip})"

    # Check if server is responding and register it
    if timeout "$HEALTH_CHECK_TIMEOUT" curl -s -o /dev/null "http://${server_ip}:${SERVER_PORT}/health"; then
        if curl -s -X POST "http://localhost:${ROUTER_PORT}/add_worker?url=${server_url}"; then
            echo "Successfully registered $server_url"
        else
            echo "Failed to register $server_url"
        fi
    else
        echo "Server at $server_url not healthy yet, skipping registration"
    fi
done < <(squeue -h -u "$USER" -n r1-server -t RUNNING -o "%i")

# Just keep router running and healthy
while true; do
    if ! curl -s -o /dev/null "http://localhost:${ROUTER_PORT}/health"; then
        echo "Error: Router health check failed"
        exit 1
    fi
    sleep 300
done