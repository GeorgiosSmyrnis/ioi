from collections import defaultdict
import json
import os
from typing import Set
import asyncio
import aiofiles
import aiohttp
from huggingface_hub import HfApi
from datasets import load_dataset
from tqdm.asyncio import tqdm
import uvloop
from piston_client import PistonClient, get_piston_endpoints
from scoring_utils import score_subtasks

# Constants
EVAL_RESULTS_DIR = '/fsx/guilherme/ioi2024/hynek_evals_per_dataset_std'
MAX_CONCURRENT = 1350
os.makedirs(EVAL_RESULTS_DIR, exist_ok=True)

def initialize_hf_api():
    """Initialize the Hugging Face API client"""
    return HfApi()

async def score_submission_on_all_subtasks(pbar, client, subtasks, test_cases, submission, file_lock):
    """Score a single submission on a subtask"""
    try:
        dataset, target_subtask, uuid, code = submission
        subtask_results = await score_subtasks(client, subtasks, test_cases, code)
        async with file_lock:
            async with aiofiles.open(f'{EVAL_RESULTS_DIR}/{dataset}.jsonl', mode="a") as f:
                await f.write(json.dumps({
                    "dataset": dataset,
                    "code": code,
                    "uuid": uuid,
                    "target_subtask": target_subtask,
                    "subtask_results": [subtask_result.to_dict() for subtask_result in subtask_results]
                }) + "\n")
                await f.flush()
    except Exception as e:
        print(f"Error scoring submission: {e}")
    finally:
        pbar.set_postfix(active=len(pbar.active_tasks), refresh=False)
        pbar.update(1)

def load_problem_data():
    """Load problem data and create mappings"""
    problems = load_dataset("open-r1/ioi", split="test")
    problem_subtasks = defaultdict(list)
    years_to_fetch = set()
    for problem in problems:
        problem_subtasks[(problem['year'], problem['id'])].append(problem)
        years_to_fetch.add(problem['year'])

    test_cases = defaultdict(dict)
    for year in years_to_fetch:
        print(f"Fetching test cases for IOI-{year}")
        year_test_cases = load_dataset("open-r1/ioi-test-cases", name=str(year), split="train")
        for test_case in year_test_cases:
            test_cases[(year, test_case['problem_id'])][test_case['test_name']] = test_case['test_input'], test_case['test_output']
    
    return problem_subtasks, test_cases

def get_completed_tests():
    """Load already completed test runs"""
    completed_tests = set()
    for dataset_jsonl in tqdm(os.listdir(EVAL_RESULTS_DIR), desc="Loading completed tests"):
        dataset_completed = 0
        with open(f'{EVAL_RESULTS_DIR}/{dataset_jsonl}', 'r') as f:
            dataset = ".".join(dataset_jsonl.split('.')[:-1])
            for line in f:
                line_data = json.loads(line)
                # does this shit even have code
                if "<think>" in line_data['code']:
                    print(f"Skipping {line_data['uuid']} because it has <think>")
                    continue
                # rerun weird unexpected errors and
                if not any(test['status'] == '?' for subtask in line_data['subtask_results'] for test in subtask['test_results']) and not any(test['feedback'] and "fatal error: cannot execute 'cc1plus'" in test['feedback'] for subtask in line_data['subtask_results'] for test in subtask['test_results']):
                # if not any(test['status'] == '?' or (test['status'] == 'CE' and line_data.get('ce_fix', 0) < 2) for subtask in line_data['subtask_results'] for test in subtask['test_results']) and \
                    # (not any("Resource temporarily unavailable" in test['error'] for subtask in line_data['subtask_results'] for test in subtask['test_results'] if test['error'])):
                    completed_tests.add((dataset, line_data['subtask_results'][0]['problem'], line_data['uuid']))
                    dataset_completed += 1
            print(f"Loaded {dataset_completed} completed submissions for dataset {dataset}")
    print(f"Loaded {len(completed_tests)} completed tests")
    return completed_tests

def get_submissions_to_evaluate(api, completed_tests):
    """Get submissions that need to be evaluated"""
    submissions_to_evaluate = defaultdict(list)
    
    # Get all datasets from the ioi-leaderboard space
    datasets = api.list_datasets(author="ioi-leaderboard")
    
    # Extract unique organizations that have submitted datasets
    for dataset_entry in datasets:
        if not dataset_entry.id.endswith("prompt-mem-limit") and not dataset_entry.id.endswith("prompt-mem-limit-fix"):
            continue
        print(dataset_entry.id)
        ds = load_dataset(dataset_entry.id, split="train")
        dataset_submissions_to_eval = 0
        for submission in tqdm(ds):
            if submission["generation"]:
                if "```cpp\n" not in submission['generation']:
                    continue
                code = submission['generation'].split("```cpp\n")[-1].split("```")[0]
                if not code:
                    continue
                code_header = '#include <bits/stdc++.h>\n'
                header_str = f'#include "{submission["problem_id"].lower()}.h"'
                if header_str not in code:
                    code_header += header_str + '\n'
                if "using namespace std;" not in code and "std::" not in code:
                    code_header += "\nusing namespace std;\n\n"
                code = code_header + code
                if (dataset_entry.id.split("/")[-1], submission["problem_id"], submission["uuid"]) not in completed_tests:
                    submissions_to_evaluate[submission["problem_id"]].append(
                        (dataset_entry.id.split("/")[-1], submission["subtask"], submission['uuid'], code)
                    )
                    dataset_submissions_to_eval += 1
        print(f"Evaluating {dataset_submissions_to_eval} submissions for dataset {dataset_entry.id.split('/')[-1]}")
    
    return submissions_to_evaluate

async def evaluate_submissions(problem_subtasks, test_cases, submissions_to_evaluate):
    """Evaluate all submissions"""
    file_lock = asyncio.Lock()
    
    async with aiohttp.ClientSession(
        timeout=aiohttp.ClientTimeout(sock_read=30), 
        connector=aiohttp.TCPConnector(limit=MAX_CONCURRENT, ttl_dns_cache=300, keepalive_timeout=3 * 60 * 60)
    ) as session:
        client = PistonClient(get_piston_endpoints(), session, max_requests_per_endpoint=1)
        active_tasks: Set[asyncio.Task] = set()

        with tqdm(
            total=sum(len(codes_to_eval) for codes_to_eval in submissions_to_evaluate.values()),
            desc="Evaluating submissions",
            unit="row",
            mininterval=2,
            smoothing=0.0001,
        ) as pbar:

            pbar.active_tasks = active_tasks

            for (year, problem_name), subtasks in problem_subtasks.items():
                codes_to_eval = submissions_to_evaluate[problem_name]
                print(f"Scoring {len(codes_to_eval)} submissions on {len(subtasks)} subtasks of {problem_name} ({len(set([test_name for subtask in subtasks for test_name in subtask['test_names']]))} test cases)")
                
                for submission in codes_to_eval:
                    while len(active_tasks) >= MAX_CONCURRENT:
                        done, active_tasks = await asyncio.wait(
                            active_tasks, return_when=asyncio.FIRST_COMPLETED
                        )
                        for task in done:
                            try:
                                await task
                            except Exception as e:
                                print(f"Task failed: {e}")
                    
                    task = asyncio.create_task(score_submission_on_all_subtasks(pbar, client, subtasks, test_cases[(year, problem_name)], submission, file_lock))
                    active_tasks.add(task)
                    task.add_done_callback(active_tasks.discard)
                    pbar.set_postfix(active=len(active_tasks), refresh=True)
                    
            if active_tasks:
                await asyncio.gather(*active_tasks, return_exceptions=True)

async def main():
    """Main function to orchestrate the evaluation process"""
    # Initialize API
    api = initialize_hf_api()
    
    # Load problem data
    problem_subtasks, test_cases = load_problem_data()
    
    # Get completed tests
    completed_tests = get_completed_tests()
    
    # Get submissions to evaluate
    submissions_to_evaluate = get_submissions_to_evaluate(api, completed_tests)
    
    # Evaluate submissions
    await evaluate_submissions(problem_subtasks, test_cases, submissions_to_evaluate)

if __name__ == "__main__":
    uvloop.install()
    asyncio.run(main())
