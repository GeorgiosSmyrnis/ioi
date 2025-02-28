# IOI Problem Evaluation

This repository contains code for evaluating Language Models on IOI 2024 problems using LiteLLM.

## Installation

1. Clone the repository
2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Environment Setup

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

Run the evaluation:
```bash
python evaluate.py --org_id YOUR_ORG_ID --model_id YOUR_MODEL_ID [--num_generations 50] [--concurrency 5]
```

Arguments:
- `--org_id`: Organization ID for output directory organization
- `--model_id`: Model ID to use with LiteLLM (e.g., "openai/gpt-4")
- `--num_generations`: Number of generations per problem (default: 50)
- `--concurrency`: Number of concurrent generations to run (default: 5)
- `--num_retries`: Number of retries for failed API calls (default: 10)
- `--n_problems`: Number of problems to evaluate (default: all)
- `--dry_run`: Run without making actual LLM calls

## Output

The results will be saved in `{org_id}/{model_id}/` directory with one JSON file per problem. Each file contains the generations, prompts, and metadata including token usage.

Progress and token usage are logged to both console and `evaluation.log`.

## Performance Notes

The script uses async/await with controlled concurrency to optimize performance. The `--concurrency` parameter lets you control how many generations run in parallel. Higher values may improve throughput but could:
1. Hit API rate limits
2. Increase memory usage
3. Result in more retries due to concurrent API calls

Adjust the concurrency based on your API limits and system resources. 