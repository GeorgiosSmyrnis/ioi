#!/bin/bash
#SBATCH --job-name=ioi-eval-Qwen-Qwen2.5-7B-Instruct
#SBATCH --partition=hopper-prod
#SBATCH --qos=normal
#SBATCH --nodes=1
#SBATCH --gpus-per-node=8
#SBATCH --exclusive
#SBATCH --output=/fsx/hynek_kydlicek/logs/ioi-eval/ioi-eval-Qwen-Qwen2.5-7B-Instruct/%j-%x.out
#SBATCH --error=/fsx/hynek_kydlicek/logs/ioi-eval/ioi-eval-Qwen-Qwen2.5-7B-Instruct/%j-%x.out
#SBATCH --time=7-00:00:00
#SBATCH --ntasks-per-node=1

set -exuo pipefail

SERVER_PORT=39877
DIST_PORT=45000
UV_ENV=/fsx/hynek_kydlicek/projects/ioi-leaderboard/test

# random sleep (0-100) to prevent ddosing server
sleep $((RANDOM % 100 + 1))

# Environment configuration
export OUTLINES_CACHE_DIR=/scratch/serve_r1/ocache/
export TRITON_HOME=/scratch/serve_r1/triton/
export GLOO_SOCKET_IFNAME="enp71s0"
export NCCL_SOCKET_IFNAME="enp71s0"

# Evaluation script path
EVAL_SCRIPT_PATH="/fsx/hynek_kydlicek/projects/ioi-leaderboard/evaluate.py"

module load cuda/12.4
source ~/.bashrc

# Activate uv
source $UV_ENV/bin/activate

# FIRST_NODE=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)
# FIRST_NODE_IP=$(srun --nodes=1 -w "$FIRST_NODE" hostname --ip-address)
FIRST_NODE_IP="$(hostname --ip-address)"

# Launch servers synchronously across all nodes
bash -c "python -m sglang.launch_server \
    --model-path 'Qwen/Qwen2-0.5B' \
    --tp 2 \
    --dist-init-addr '$FIRST_NODE_IP:$DIST_PORT' \
    --nnodes 1 \
    --node-rank \$SLURM_PROCID \
    --port '$SERVER_PORT' \
    --host 0.0.0.0 \
    --trust-remote-code \
    --max-running-requests 100 \
    --context-length 4096" &

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

echo "Checking available models..."
curl "http://$FIRST_NODE_IP:$SERVER_PORT/v1/models"
sleep 10

echo "Executing sanity check..."
curl "http://$FIRST_NODE_IP:$SERVER_PORT/v1/completions" \
    -H "Content-Type: application/json" \
    -d '{
        "model": "default",
        "prompt": "hi, how are you?",
        "max_tokens": 2048,
        "temperature": 0.6
    }'

python "$EVAL_SCRIPT_PATH" \
    --model_id "sglang/Qwen/Qwen2-0.5B" \
    --api_base "http://localhost:$SERVER_PORT/v1" \
    --concurrency 100 \
    --org_id=ioi-leaderboard --num_problems=6 --num_generations=1 --model_postfix=test --num_subtasks=2 --override

# Kill the server and exit
pkill -f "python -m sglang.launch_server"
exit 0
