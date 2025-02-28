import asyncio
import json
import os
from pathlib import Path
from typing import Dict, List, Optional
import random
from datetime import datetime
import re
import uuid
from datasets import Dataset
import datasets
from loguru import logger
from tqdm.asyncio import tqdm
import litellm
from dotenv import load_dotenv
from huggingface_hub import HfApi

PROMPT = """\
You are an expert competitive programmer. You will be given a problem statement, test case constraints and example test inputs and outputs. Please reason step by step about the solution, then provide a complete implementation in C++17. Your solution must read input from standard input (cin), and write output to standard output (cout). Do not include any debug prints or additional output. Your solution will have 2 seconds execution time to solve each test case.
Put your final solution within a single code block: ```cpp\n<your code here>``` 

Problem statement:
{problem_statement}\
"""

class IOIEvaluator:
    def __init__(self, org_id: str, model_id: str, num_generations: int = 50, num_retries: int = 10, 
                 concurrency: int = 10, n_problems: Optional[int] = None, dry_run: bool = False,
                 override: bool = False):
        self.org_id = org_id
        self.model_id = model_id
        self.num_generations = num_generations
        self.num_retries = num_retries
        self.concurrency = concurrency
        self.n_problems = n_problems
        self.dry_run = dry_run
        self.override = override
        self.dataset = None
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
        
        # Semaphore for controlling concurrency
        self._semaphore = asyncio.Semaphore(concurrency)

        if dry_run:
            logger.warning("Running in dry-run mode - no actual LLM calls will be made")

    async def load_previous_results(self):
        """Load previous results from HuggingFace Hub or local files."""
        if self.override:
            logger.info("Override mode enabled - not loading previous results")
            return None
            
        try:
            # First try loading from HuggingFace Hub
            model_name = f"dummy-{self.model_id}" if self.dry_run else self.model_id
            repo_name = f"{self.org_id}/ioi-eval-{model_name.replace('/', '_')}"
            
            try:
                logger.info(f"Attempting to load previous results from HuggingFace Hub: {repo_name}")
                dataset = datasets.load_dataset(repo_name)
                if dataset and 'train' in dataset:
                    logger.info(f"Loaded previous results from HuggingFace Hub: {len(dataset['train'])} entries")
                    return list(dataset['train'])
            except Exception as e:
                logger.info(f"Could not load from HuggingFace Hub: {str(e)}")
            
            # If hub loading failed, try local file
            dataset_file = self.org_dir / f"{self.model_id}_dataset.json"
            if dataset_file.exists():
                logger.info(f"Loading previous results from local file: {dataset_file}")
                with open(dataset_file) as f:
                    results = json.load(f)
                    logger.info(f"Loaded previous results from local file: {len(results)} entries")
                    return results
                    
            logger.info("No previous results found")
            return None
            
        except Exception as e:
            logger.error(f"Error loading previous results: {str(e)}")
            return None

    def load_dataset(self):
        """Load the IOI 2024 dataset from Huggingface."""
        logger.info("Loading IOI 2024 dataset...")
        self.dataset = datasets.load_dataset("open-r1/ioi-2024")
        logger.info(f"Dataset loaded with {len(self.dataset['train'])} problems")

    def get_subtasks(self, problem_id: str) -> List[Dict]:
        """Get all subtasks for a given problem ID."""
        return [
            item for item in self.dataset["train"] 
            if item["id"] == problem_id
        ]

    def get_next_subtask(self, subtasks: List[Dict], last_i: int) -> Optional[Dict]:
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
            if not text:
                logger.warning("Empty text provided to extract_code")
                return "", "unknown"
                
            code_pattern = r"```cpp\n(.*?)```"
            match = re.search(code_pattern, text, re.DOTALL)
            if match:
                code = match.group(1).strip()
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
                
            response = await litellm.acompletion(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt, "cache_control": {"type": "ephemeral"}}],
                seed=seed,
                num_retries=self.num_retries,
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
                message_content = ""
            
            # Extract code from the response
            code, language = self.extract_code(message_content)
            
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
                "generation": "",
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

    async def evaluate_problem(self, problem_id: str) -> List[Dict]:
        """Prepare result entries for a single problem."""
        subtasks = self.get_subtasks(problem_id)
        if not subtasks:
            logger.warning(f"No subtasks found for problem {problem_id}")
            return []

        results = []
        for i in range(self.num_generations):
            subtask = self.get_next_subtask(subtasks, i)
            if not subtask:
                return []

            try:
                prompt = PROMPT.format(problem_statement=subtask.get('statement', ''))
                results.append({
                    "problem_id": problem_id,
                    "subtask": subtask.get("subtask", "unknown"),
                    "prompt": prompt,
                    "generation": None,
                    "code": "",
                    "language": "unknown",
                    "solution_number": i,
                    "uuid": str(uuid.uuid4()),
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
            if self.dataset is None:
                try:
                    self.load_dataset()
                except Exception as e:
                    logger.error(f"Failed to load dataset: {str(e)}")
                    return []

            # Get unique problem IDs
            try:
                problem_ids = list(set(item["id"] for item in self.dataset["train"]))
            except Exception as e:
                logger.error(f"Failed to extract problem IDs: {str(e)}")
                return []
            
            # Sort problem IDs to ensure consistent ordering and limit if n_problems is set
            problem_ids.sort()
            if self.n_problems is not None:
                problem_ids = problem_ids[:self.n_problems]
                logger.info(f"Limited evaluation to first {self.n_problems} problems")
            
            logger.info(f"Starting evaluation of {len(problem_ids)} problems...")

            # Create or load the complete pool of results
            all_results = []
            if not self.override:
                # Try to load previous results
                previous_results = await self.load_previous_results()
                if previous_results:
                    all_results = previous_results
                    logger.info(f"Loaded {len(all_results)} previous results")

            if not all_results:
                # Create fresh pool of results
                logger.info("Creating fresh pool of results...")
                for problem_id in tqdm(problem_ids, desc="Preparing results"):
                    results = await self.evaluate_problem(problem_id)
                    all_results.extend(results)
                logger.info(f"Created {len(all_results)} result entries")

            # Count how many need to be generated
            to_generate = [r for r in all_results if r.get("generation") is None]
            logger.info(f"Need to generate {len(to_generate)} out of {len(all_results)} total entries")

            if not to_generate:
                logger.info("No generations needed - all results are already available")
                return all_results

            # Run generations for entries without results
            async def process_single(result: Dict) -> Optional[Dict]:
                seed = result["model_kwargs"]["seed"]
                async with self._semaphore:
                    try:
                        llm_result = await self.call_llm(result["prompt"], result["model_kwargs"]["seed"])
                        result.update({
                            "generation": llm_result["generation"],
                            "code": llm_result["code"],
                            "language": llm_result["language"],
                            "model_kwargs": llm_result["model_kwargs"],
                            "metadata": llm_result["metadata"]
                        })
                        
                        # Log progress and token usage
                        if result["metadata"].get("usage"):
                            usage = result["metadata"]["usage"]
                            logger.info(
                                f"Problem {result['problem_id']} (Solution {result['solution_number']}) - "
                                f"Tokens: {usage.get('total_tokens', 0)} "
                                f"(prompt: {usage.get('prompt_tokens', 0)}, "
                                f"completion: {usage.get('completion_tokens', 0)}) - "
                                f"Cost: ${usage.get('cost', 0.0):.4f}"
                            )
                        return result
                    except Exception as e:
                        logger.error(f"Failed generation for problem {result['problem_id']}: {str(e)}")
                        return result

            tasks = [
                process_single(result) if result.get("generation") is None else result
                for result in to_generate
            ]
            
            # Run generations in parallel with controlled concurrency
            generated_results = await tqdm.gather(*tasks, desc="Running generations")

            all_result = Dataset.from_list(generated_results)
            model_name = f"ioi-eval-{self.model_id.replace('/', '_')}"
            if self.dry_run:
                model_name = f"dummy-{model_name}"

            # Save to disk first
            try:
                all_result.save_to_disk(self.org_dir / f"{model_name}")
            except Exception as e:
                logger.error(f"Failed to save dataset: {str(e)}")

            # Log final statistics
            successful = len([r for r in all_results if r.get("generation") is not None])
            logger.info(f"Evaluation completed. Total successful generations: {successful}/{len(all_results)}")
            logger.info(
                f"Total tokens used: {self.total_prompt_tokens + self.total_completion_tokens} "
                f"(prompt: {self.total_prompt_tokens}, completion: {self.total_completion_tokens})"
            )
            logger.info(f"Total cost: ${self.total_cost:.4f}")
            
            # Push to Hugging Face Hub
            try:
                all_result.push_to_hub(f"{self.org_id}/{model_name}")
                logger.info(f"Pushed to hub: {self.org_id}/{model_name}")
            except Exception as e:
                logger.error(f"Failed to push to hub: {str(e)}")
            
            return all_results
        except Exception as e:
            logger.error(f"Critical error in run_evaluation: {str(e)}")
            return []


def main():
    load_dotenv()  # Load environment variables from .env file
    
    import argparse
    parser = argparse.ArgumentParser(description="Evaluate LLMs on IOI problems")
    parser.add_argument("--org_id", required=True, help="Organization ID")
    parser.add_argument("--model_id", required=True, help="Model ID")
    parser.add_argument("--num_generations", type=int, default=50, help="Number of generations per problem")
    parser.add_argument("--num_retries", type=int, default=10, help="Number of retries for failed API calls")
    parser.add_argument("--concurrency", type=int, default=400, help="Number of concurrent generations")
    parser.add_argument("--n_problems", type=int, default=None, help="Number of problems to evaluate (None for all)")
    parser.add_argument("--dry_run", action="store_true", help="Run without making actual LLM calls")
    parser.add_argument("--override", action="store_true", help="Override existing results and start fresh")
    args = parser.parse_args()

    evaluator = IOIEvaluator(
        org_id=args.org_id,
        model_id=args.model_id,
        num_generations=args.num_generations,
        num_retries=args.num_retries,
        concurrency=args.concurrency,
        n_problems=args.n_problems,
        dry_run=args.dry_run,
        override=args.override
    )
    asyncio.run(evaluator.run_evaluation())

if __name__ == "__main__":
    main() 