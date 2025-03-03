#!/bin/bash
#SBATCH --job-name=r1-server
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=2
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=./logs/%x_%j_%n.out
#SBATCH --error=./logs/%x_%j_%n.err
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1

set -exuo pipefail

MODEL_PATH="deepseek-ai/DeepSeek-R1"
CONDA_ENV="sglang124"
ROUTER_ADDRESS=""
SERVER_PORT=39877
DIST_PORT=45000

# TODO: Adjust these variables to your cluster configuration
export OUTLINES_CACHE_DIR=/scratch/serve_r1/ocache/
export TRITON_HOME=/scratch/serve_r1/triton/
export GLOO_SOCKET_IFNAME="enp71s0"
export NCCL_SOCKET_IFNAME="enp71s0"

# Add new variables for evaluation
EVAL_SCRIPT_PATH="/fsx/hynek_kydlicek/projects/ioi-leaderboard/evaluate.py"
EVAL_ARGS=""
TP=16

while getopts "m:a:t:h" opt; do
    case $opt in
        m) MODEL_PATH="$OPTARG" ;;
        a) EVAL_ARGS="$OPTARG" ;;
        t) TP="$OPTARG" ;;
        h|?) echo "Usage: sbatch $0 [-m MODEL_PATH] [-e CONDA_ENV] [-r ROUTER_ADDRESS] [-a EVAL_ARGS] [-t TP]"; exit 1 ;;
    esac
done

# Verify required evaluation arguments
if [ -z "$EVAL_ARGS" ]; then
    echo "Error: Evaluation arguments (-a EVAL_ARGS) are required"
    exit 1
fi

if [ -z $MODEL_PATH ]; then
    echo "Error: Model path (-m MODEL_PATH) is required"
    exit 1
fi

module load cuda/12.4
source ~/.bashrc

# Activate uv
source /fsx/hynek_kydlicek/projects/ioi-leaderboard/ioi-eval/bin/activate

CONTEXT_LENGTH=$(python get_context_length.py --model_name $MODEL_PATH)
if [ -z $CONTEXT_LENGTH ]; then
    echo "Error: Context length (-c CONTEXT_LENGTH) is required"
    exit 1
fi

FIRST_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
FIRST_NODE_IP=$(srun --nodes=1 --ntasks=1 -w "$FIRST_NODE" hostname --ip-address)
# FIRST_NODE_IP=$(hostname --ip-address)

# Launch servers synchronously across all nodes
# (--max-running-requests=56 is rough estimate to avoid too many evicted/preempted 16k-long requests)
srun --nodes=2 --ntasks=2 --ntasks-per-node=1 \
    bash -c "python -m sglang.launch_server \
        --model-path '$MODEL_PATH' \
        --tp $TP \
        --dist-init-addr '$FIRST_NODE_IP:$DIST_PORT' \
        --nnodes 2 \
        --node-rank \$SLURM_PROCID \
        --port '$SERVER_PORT' \
        --host 0.0.0.0 \
        --trust-remote-code \
        --max-running-requests 56 \
        --context-length $CONTEXT_LENGTH" &

# Wait for server with timeout
TIMEOUT=3600  # 1h, but model loading should take ~30min
START_TIME=$(date +%s)
echo "Waiting for SGLang server (http://$FIRST_NODE_IP:$SERVER_PORT)..."

while true; do
    if curl -s -o /dev/null -w "%{http_code}" "http://$FIRST_NODE_IP:$SERVER_PORT/health" >/dev/null 2>&1; then
        echo "Server is ready at http://$FIRST_NODE_IP:$SERVER_PORT"
        break
    fi

    CURRENT_TIME=$(date +%s)
    if [ $((CURRENT_TIME - START_TIME)) -gt $TIMEOUT ]; then
        echo "Error: Server failed to start within $TIMEOUT seconds"
        exit 1
    fi

    echo "Still waiting... ($(($CURRENT_TIME - $START_TIME)) seconds elapsed)"
    sleep 60
done

# Register with router only if address was provided
if [ -n "$ROUTER_ADDRESS" ]; then
    echo "Registering with router at $ROUTER_ADDRESS..."
    curl -X POST "http://$ROUTER_ADDRESS/add_worker?url=http://$FIRST_NODE_IP:$SERVER_PORT" || true
    sleep 10
fi

echo "Checking available models..."
curl "http://$FIRST_NODE_IP:$SERVER_PORT/v1/models"
sleep 10

echo "Executing sanity check..."
curl "http://$FIRST_NODE_IP:$SERVER_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d "{
        \"model\": \"default\",
        \"prompt\": \"hi, how are you?\",
        \"max_tokens\": 2048,
        \"temperature\": 0.6
    }"

# After server is ready and health check passes, launch evaluation
if [ -f "$EVAL_SCRIPT_PATH" ]; then
    echo "Starting evaluation with args: $EVAL_ARGS"
    python "$EVAL_SCRIPT_PATH" \
        --model_id "sglang/$MODEL_PATH" \
        --api_base "http://localhost:$SERVER_PORT/v1" \
        $EVAL_ARGS
else
    echo "Warning: Evaluation script not found at $EVAL_SCRIPT_PATH"
fi

# Kill the server and exit
pkill -f "python -m sglang.launch_server"
exit 0
