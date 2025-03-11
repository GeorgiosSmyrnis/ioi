# IOI Problem Evaluation

This repository contains code for evaluating Language Models on IOI 2024 problems using LiteLLM.

## Installation

1. Clone the repository
2. Install dependencies:
```bash

uv pip install torch~=2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv pip install sgl-kernel --force-reinstall --no-deps
uv pip install "sglang[all]>=0.4.2.post4" --find-links https://flashinfer.ai/whl/cu124/torch2.5/flashinfer/
uv pip install -r requirements.txt
```

## Environment Setup (In case you want to use remote models)

1. Copy the environment template:
```bash
cp .env.template .env
```

2. Edit `.env` and:
   - Uncomment the variables for the LLM providers you plan to use
   - Replace the placeholder values with your actual API keys
   - Optional: Configure proxy settings if needed

Example `.env` for using OpenAI's GPT-4:
```bash
OPENAI_API_KEY=your_actual_key_here
OPENAI_ORGANIZATION=your_org_id  # Optional
```

## Usage

### Running with Remote Models

Run the evaluation with remote models:
```bash
python evaluate.py --org_id YOUR_ORG_ID --model_id YOUR_MODEL_ID [--num_generations 50] [--concurrency 5]
```

Command line arguments:
- `--org_id`: Organization ID (required)
- `--model_id`: Model ID in LiteLLM format (required)
- `--api_base`: API base URL for the model (optional)
- `--num_generations`: Number of generations per problem (default: 50)
- `--num_retries`: Number of retries for failed API calls (default: 10)
- `--concurrency`: Number of concurrent generations (default: 20)
- `--num_problems`: Number of problems to evaluate (default: all)
- `--num_subtasks`: Number of subtasks to evaluate per problem (default: 1, use -1 for all)
- `--dry_run`: Run without making actual LLM calls
- `--override`: Override existing results and start fresh
- `--model_postfix`: Postfix for the model name
- `--revision`: Revision to use for the model
- `--timeout`: Timeout for the LLM call in seconds (default: 600)
- `--use_requests`: Use requests instead of litellm
- `--max_tokens`: Maximum number of tokens for generation

### Running with Locally Deployed Models (SGLang)

For locally deployed models using SGLang, you can use the provided scripts:

#### Using SLURM for Distributed Deployment

For HPC environments with SLURM, use run_ioi_slurm.py to evaluate open models:

```bash
python run_ioi_slurm.py --model "MODEL_PATH" --concurrency=30 --startup_delay="7200" --log_dir "DIR_FOR_OUTPUT_LOGS" --slurm_dir "DIR_FOR_SLUR_SCRIPT" --uv_env "PATH_TO_UV_ENV" --eval_args "--org_id=YOUR_ORG_ID --num_problems=50 --num_generations=50 --model_postfix=POSTFIX --num_subtasks=-1"
```

## Output

The results will be saved in the directory structure:
```
{org_id}/{revision}-{model_id}-{postfix}/
```

The output includes:
- Generated code solutions for each problem and subtask
- Metrics on generation performance
- Token usage statistics

You can analyze the results using the saved data to evaluate the model's performance on competitive programming tasks.