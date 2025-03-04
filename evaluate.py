import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import random
from datetime import datetime
import re
import uuid
from datasets import Dataset, load_dataset
from loguru import logger
from tqdm.asyncio import tqdm
import litellm
from dotenv import load_dotenv
from huggingface_hub import HfApi
import polars as pl

PROMPT = """\
You are an expert competitive programmer. You will be given a problem statement, test case constraints and example test inputs and outputs. Please reason step by step about the solution, then provide a complete implementation in C++17. You should correctly implement the routine(s) described in Implementation Details, without reading or writing anything directly from stdin or to stdout, as input and output are passed through the implemented routines. Assume your code will be run on the OFFICIAL grader, and do not add a main, a sample grader, or any other functionality unless it has been explicitly requested.
Put your final solution within a single code block: ```cpp\n<your code here>``` 

# Problem statement ({problem_name})
{problem_statement}

## Time limit
Your solution will have {time_limit} second(s) execution time to solve each test case.

# Starting code
Here's your starting code with some skeleton/placeholder functionality:
```cpp
{skeleton}
```\
"""

class IOIEvaluator:
    def __init__(self, org_id: str, model_id: str, api_base: Optional[str] = None, 
                 num_generations: int = 50, num_retries: int = 10, 
                 concurrency: int = 10, num_problems: Optional[int] = None, 
                 num_subtasks: Optional[int] = None, dry_run: bool = False,
                 override: bool = False, model_postfix: Optional[str] = None):
        self.org_id = org_id
        self.model_id = model_id
        self.api_base = api_base
        self.num_generations = num_generations
        self.num_retries = num_retries
        self.concurrency = concurrency
        self.num_problems = num_problems
        self.num_subtasks = num_subtasks
        self.dry_run = dry_run
        self.override = override
        self.previous_results = None
        
        # Create organization and model directories
        self.org_dir = Path(org_id)
        self.model_dir = self.org_dir / model_id
        self.org_dir.mkdir(parents=True, exist_ok=True)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Tracking totals
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cost = 0.0
        self.model_postfix = model_postfix
        
        # Semaphore for controlling concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

        if self.api_base:
            logger.info(f"Using API base: {self.api_base}")

        if dry_run:
            logger.warning("Running in dry-run mode - no actual LLM calls will be made")

    async def load_previous_results(self) -> Optional[pl.DataFrame]:
        """Load previous results from HuggingFace Hub or local files."""
        if self.override:
            logger.info("Override mode enabled - not loading previous results")
            return None
            
        repo_name = f"{self.org_id}/{self.get_model_name()}"
        
        try:
            logger.info(f"Attempting to load previous results from HuggingFace Hub: {repo_name}")
            dataset = load_dataset(repo_name, split="train")
            logger.info(f"Loaded previous results from HuggingFace Hub: {len(dataset)} entries")
            return dataset.to_polars()
        except Exception as e:
            logger.info(f"Could not load from HuggingFace Hub: {str(e)}")
            

    def load_ioi_dataset(self):
        """Load the IOI 2024 dataset from Huggingface."""
        logger.info("Loading IOI 2024 dataset...")
        return load_dataset("open-r1/ioi-2024", split="train").to_polars()
    
    def get_subtasks(self, problem_id: str, dataset: pl.DataFrame) -> List[Dict]:
        """Get all subtasks for a given problem ID."""
        subtasks = dataset.filter(pl.col("id") == problem_id).select(pl.col("subtask", "statement", "time_limit", "starting_code")).unique().to_dicts()
        # If n_subtasks is set, limit the number of subtasks
        if self.num_subtasks is not None and self.num_subtasks > 0:
            # Sort subtasks by ID to ensure consistent selection
            subtasks = sorted(subtasks, key=lambda x: int(x["subtask"][:2]))
            subtasks = subtasks[-self.num_subtasks:]
            
        return subtasks

    def get_next_subtask(self, subtasks: List[Dict]) -> Optional[Dict]:
        """Get the subtask with the highest ID."""
        # Extract subtask IDs (first two digits)
        try:
            subtask_ids = [int(task["subtask"][:2]) for task in subtasks]
        except ValueError:
            logger.error(f"Failed to convert subtask IDs to integers for problem {subtasks[0]['id']}")
            subtask_ids = list(range(len(subtasks)))

        return subtasks[subtask_ids.index(max(subtask_ids))]

    def get_dummy_response(self, prompt: str, seed: int) -> Dict:
        """Generate a dummy response for dry runs."""
        dummy_code = """```cpp
int main() {
    // This is a dummy solution
    return 0;
}
```"""
        return {
            "generation": f"This is a dummy response for testing purposes.\n{dummy_code}",
            "code": "int main() {\n    // This is a dummy solution\n    return 0;\n}",
            "language": "cpp",
            "model_kwargs": {
                "seed": seed,
            },
            "metadata": {
                "usage": {
                    'completion_tokens': 10,
                    'prompt_tokens': len(prompt.split()),
                    'total_tokens': len(prompt.split()) + 10,
                    'cost': 0.0
                },
                "timestamp": datetime.now().isoformat()
            }
        }

    def extract_code(self, text: str) -> tuple[str, str]:
        """Extract code from the response between ```cpp and ``` markers."""
        try:
            parts = text.split("```cpp\n")
            if len(parts) > 1:
                code_block = parts[-1].split("```")[0]
                code = code_block.strip()
                if not code:
                    logger.warning("Empty code block found")
                    return "", "cpp"
                return code, "cpp"
            logger.warning("No code block found in the response")
            return "", "unknown"
        except Exception as e:
            logger.error(f"Failed to extract code: {str(e)}")
            return "", "unknown"

    async def call_llm(self, prompt: str, seed: int) -> Dict:
        """Call the LLM using LiteLLM's built-in retry mechanism."""
        try:
            if self.dry_run:
                return self.get_dummy_response(prompt, seed)

            model_name = self.model_id
            kwargs = {}
            if self.model_id.startswith("sglang/"):
                model_name = model_name.replace("sglang/", "openai/")
                kwargs["api_base"] = self.api_base
                kwargs["api_key"] = "sk-proj-1234567890"
                kwargs["top_k"] = 20

            

            response = await litellm.acompletion(
                model=model_name,
                messages=[{"role": "user", "content": prompt, "cache_control": {"type": "ephemeral"}}],
                seed=seed,
                num_retries=self.num_retries,
                top_p=0.8,
                temperature=0.7,
                **kwargs
            )
            
            # Extract usage information safely
            usage = {}
            cost = 0.0
            if hasattr(response, 'usage'):
                try:
                    completion_tokens = getattr(response.usage, 'completion_tokens', 0)
                    prompt_tokens = getattr(response.usage, 'prompt_tokens', 0)
                    total_tokens = getattr(response.usage, 'total_tokens', 0)
                    
                    # Calculate cost using litellm
                    try:
                        cost = litellm.completion_cost(completion_response=response)
                    except Exception as e:
                        logger.error(f"Failed to calculate cost: {str(e)}")
                        cost = 0.0
                    
                    usage = {
                        'completion_tokens': completion_tokens,
                        'prompt_tokens': prompt_tokens,
                        'total_tokens': total_tokens,
                        'cost': cost
                    }
                    
                    # Update totals
                    self.total_prompt_tokens += prompt_tokens
                    self.total_completion_tokens += completion_tokens
                    self.total_cost += cost
                    
                except Exception as e:
                    logger.error(f"Failed to extract usage information: {str(e)}")
            
            # Extract message content safely
            try:
                message_content = response.choices[0].message.content if response.choices else ""
            except Exception as e:
                logger.error(f"Failed to extract message content: {str(e)}")
                message_content = None
            
            # Extract code from the response
            code, language = self.extract_code(message_content or "")
            return {
                "generation": message_content,
                "code": code,
                "language": language,
                "model_kwargs": {
                    "seed": seed,
                },
                "metadata": {
                    "usage": usage,
                    "timestamp": datetime.now().isoformat()
                }
            }
        except Exception as e:
            logger.error(f"LLM call failed: {str(e)}")
            return {
                "generation": None,
                "code": "",
                "language": "unknown",
                "model_kwargs": {
                    "seed": seed,
                },
                "metadata": {
                    "usage": {
                        'completion_tokens': 0,
                        'prompt_tokens': 0,
                        'total_tokens': 0,
                        'cost': 0.0
                    },
                    "timestamp": datetime.now().isoformat(),
                    "error": str(e)
                }
            }

    def prepare_prompt(self, task: dict) -> str:
        statement = task['statement']
        task_name, statement = statement.split("\n\n", maxsplit=1)
        task_name = task_name.lstrip("# ")
        time_limit = task['time_limit']
        skeleton = task['starting_code'].strip()

        return PROMPT.format(
            problem_name=task_name,
            problem_statement=statement,
            time_limit=time_limit,
            skeleton=skeleton
        )

    async def create_solution_requests(self, problem_id: str, dataset: pl.DataFrame) -> List[Dict]:
        """Prepare result entries for a single problem."""
        subtasks = self.get_subtasks(problem_id, dataset)
        if not subtasks:
            logger.warning(f"No subtasks found for problem {problem_id}")
            return []
        
        results = []
        for subtask in subtasks:
            prompt = self.prepare_prompt(subtask)
            for i in range(self.num_generations):
                try:
                    random_uuid = str(uuid.uuid4())
                    
                    results.append({
                        "problem_id": problem_id,
                        "subtask": subtask["subtask"],
                        "prompt": prompt,
                        "generation": None,
                        "code": "",
                        "language": "unknown",
                        "solution_number": i,
                        "uuid": random_uuid,
                        "model_kwargs": {"seed": i},
                        "metadata": {
                            "usage": {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0, 'cost': 0.0},
                            "timestamp": datetime.now().isoformat()
                        }
                    })
                except Exception as e:
                    logger.error(f"Failed to prepare prompts for problem {problem_id}: {str(e)}")
                    return []

        return results

    async def run_evaluation(self):
        """Run the evaluation for all problems."""
        try:
            dataset = self.load_ioi_dataset()
            # Get unique problem IDs
            try:
                # Extract unique problem IDs using polars
                problem_ids = dataset.select(pl.col("id")).unique().to_series().to_list()
                problem_ids.sort()
                if self.num_problems is not None:
                    problem_ids = problem_ids[:self.num_problems]
                    logger.info(f"Limited evaluation to first {self.num_problems} problems")
            except Exception as e:
                logger.error(f"Failed to extract problem IDs: {str(e)}")
                return []
            
            logger.info(f"Starting evaluation of {len(problem_ids)} problems...")

            # Step 1: Generate all solution requests
            all_solution_requests = []
            for problem_id in tqdm(problem_ids, desc="Preparing solution requests"):
                request = await self.create_solution_requests(problem_id, dataset)
                all_solution_requests.extend(request)
            
            # Convert to Polars DataFrame for efficient operations
            requests_df = pl.DataFrame(all_solution_requests)
            logger.info(f"Created {len(requests_df)} solution requests")

            # Step 2: Load previous results
            previous_df = None
            if not self.override:
                previous_df = await self.load_previous_results()
                if previous_df is not None:
                    logger.info(f"Loaded {len(previous_df)} previous results")
            
            
            # Step 3: Merge solution requests with previous results efficiently
            if previous_df is not None:
                # Keep only the columns we want to preserve from previous results
                preserve_cols = ['generation', 'code', 'language', 'metadata', 'model_kwargs']

                preserve_cols_with_key = preserve_cols + ['problem_id', 'subtask', 'solution_number']
                previous_df = previous_df.select(preserve_cols_with_key).filter(pl.col('generation').is_not_null() & (pl.col('generation') != ""))
                
                # Merge using polars, keeping all solution requests and only matching previous results
                merged_df = requests_df.join(
                    previous_df,
                    on=('problem_id', 'subtask', 'solution_number'),
                    how='left',
                    suffix='_prev'
                )

                # Update values from previous results where they exist
                for col in preserve_cols:
                    prev_col = f'{col}_prev'
                    merged_df = merged_df.with_columns(
                        pl.when(pl.col(prev_col).is_not_null())
                        .then(pl.col(prev_col))
                        .otherwise(pl.col(col))
                        .alias(col)
                    )
                
                # Drop the _prev columns
                merged_df = merged_df.select([
                    c for c in merged_df.columns if not c.endswith('_prev')
                ])
            else:
                merged_df = requests_df

            # Count how many need to be generated
            to_generate_df = merged_df.filter(
                (pl.col('generation').is_null()) | 
                (pl.col('generation') == "")
            )
            logger.info(f"Need to generate {len(to_generate_df)} out of {len(merged_df)} total entries")

            if len(to_generate_df) == 0:
                logger.info("No generations needed - all results are already available")
                return

            # Run generations for entries without results
            async def process_single(row: Dict) -> Dict:
                async with self._semaphore:
                    try:
                        llm_result = await self.call_llm(row["prompt"], row["model_kwargs"]["seed"])
                        
                        # Log progress and token usage
                        if llm_result["metadata"].get("usage"):
                            usage = llm_result["metadata"]["usage"]
                            logger.info(
                                f"Problem {row['problem_id']} (Solution {row['solution_number']}) - "
                                f"Tokens: {usage.get('total_tokens', 0)} "
                                f"(prompt: {usage.get('prompt_tokens', 0)}, "
                                f"completion: {usage.get('completion_tokens', 0)}) - "
                                f"Cost: ${usage.get('cost', 0.0):.4f}"
                            )
                        llm_result["uuid"] = row["uuid"]
                        return llm_result
                    except Exception as e:
                        logger.error(f"Failed generation for problem {row['problem_id']}: {str(e)}")
                        return {
                            "generation": None,
                            "code": "",
                            "language": "unknown",
                            "uuid": row["uuid"],
                            "metadata": {
                                "error": str(e),
                                "usage": {'completion_tokens': 0, 'prompt_tokens': 0, 'total_tokens': 0, 'cost': 0.0},
                                "timestamp": datetime.now().isoformat()
                            }
                        }

            # Run generations in parallel with controlled concurrency
            to_generate_dicts = to_generate_df.to_dicts()
            tasks = [process_single(row) for row in to_generate_dicts]
            generated_results = await tqdm.gather(*tasks, desc="Running generations")

            # Convert generated results to DataFrame and update original DataFrame
            generated_df = pl.DataFrame(generated_results)

            # Merge generated results with previous results
            merged_df = merged_df.join(
                generated_df,
                on='uuid',
                how='left',
                suffix='_gen'
            )

            # Update the old columns with the new values
            for col in ['generation', 'code', 'language', 'metadata', 'model_kwargs']:
                merged_df = merged_df.with_columns(
                    pl.when(pl.col(f'generation_gen').is_not_null() & (pl.col(f'generation_gen') != ""))
                    .then(pl.col(f'{col}_gen'))
                    .otherwise(pl.col(col))
                    .alias(col)
                )

            # Drop the _gen columns
            merged_df = merged_df.select([
                c for c in merged_df.columns if not c.endswith('_gen')
            ])
                
            # Convert to HF Dataset
            output_dataset = Dataset.from_polars(merged_df)
            model_name = self.get_model_name()

            # Save to disk first
            # try:
            #     output_dataset.save_to_disk(f"ioi-leaderboard/{self.org_dir}_{model_name}")
            # except Exception as e:
            #     logger.error(f"Failed to save dataset: {str(e)}")

            # Log final statistics
            # logger.info(f"Evaluation completed. Total successful generations: {successful}/{len(all_results)}")
            logger.info(
                f"Total tokens used: {self.total_prompt_tokens + self.total_completion_tokens} "
                f"(prompt: {self.total_prompt_tokens}, completion: {self.total_completion_tokens})"
            )
            logger.info(f"Total cost: ${self.total_cost:.4f}")
            
            # Push to Hugging Face Hub
            try:
                output_dataset.push_to_hub(f"{self.org_id}/{model_name}")
                logger.info(f"Pushed to hub: {self.org_id}/{model_name}")
            except Exception as e:
                logger.error(f"Failed to push to hub: {str(e)}")
            
            return output_dataset
        except Exception as e:
            raise e

    
    def get_model_name(self):
        model_name = f"ioi-eval-{self.model_id.replace('/', '_')}"
        if self.dry_run:
            model_name = f"dummy-{model_name}"

        if self.model_postfix:
            model_name = f"{model_name}-{self.model_postfix}"

        return model_name


def main():
    load_dotenv()  # Load environment variables from .env file
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LLMs on IOI problems")
    parser.add_argument("--org_id", required=True, help="Organization ID")
    parser.add_argument("--model_id", required=True, help="Model ID")
    parser.add_argument("--api_base", help="API base URL for the model")
    parser.add_argument("--num_generations", type=int, default=50, help="Number of generations per problem")
    parser.add_argument("--num_retries", type=int, default=10, help="Number of retries for failed API calls")
    parser.add_argument("--concurrency", type=int, default=400, help="Number of concurrent generations")
    parser.add_argument("--num_problems", type=int, default=None, help="Number of problems to evaluate (None for all)")
    parser.add_argument("--num_subtasks", type=int, default=1, help="Number of subtasks to evaluate per problem (None for all)")
    parser.add_argument("--dry_run", action="store_true", help="Run without making actual LLM calls")
    parser.add_argument("--override", action="store_true", help="Override existing results and start fresh")
    parser.add_argument("--model_postfix", help="Postfix for the model name")
    args = parser.parse_args()

    evaluator = IOIEvaluator(
        org_id=args.org_id,
        model_id=args.model_id,
        api_base=args.api_base,
        num_generations=args.num_generations,
        num_retries=args.num_retries,
        concurrency=args.concurrency,
        num_problems=args.num_problems,
        num_subtasks=args.num_subtasks,
        dry_run=args.dry_run,
        override=args.override,
        model_postfix=args.model_postfix
    )
    asyncio.run(evaluator.run_evaluation())

if __name__ == "__main__":
    main() 