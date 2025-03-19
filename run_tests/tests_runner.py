from collections import defaultdict
from fnmatch import fnmatch
import json
import os
from typing import Set
import asyncio
import aiofiles
import aiohttp
from datasets import load_dataset, Dataset
import pandas as pd
from tqdm.asyncio import tqdm
from selection_simulator import get_problem_scores, simulate_round_robin
from utils import add_includes
import uvloop
from piston_client import get_piston_client_from_env
from scoring import score_subtasks
from dotenv import load_dotenv
from loguru import logger
from huggingface_hub import HfApi

class TestsRunner:
    def __init__(
        self,
        datasets_to_evaluate: list[str] | str,
        results_dataset_name: str,
        local_results_cache: str = "results",
        max_concurrent_requests: int = 100,
        test_batch_size: int = 1,
        dry_run: bool = False,
        override: bool = False,
        timeout: int = 60 * 10,
        id_column: str = "uuid",
        add_messages_column: bool = False,
        add_includes: bool = True,
        always_extract_code: bool = False
    ):
        self.datasets_to_evaluate = datasets_to_evaluate if isinstance(
            datasets_to_evaluate, list) else [datasets_to_evaluate]
        self.datasets_to_evaluate_names = [dataset.split(
            '/')[-1].removeprefix('ioi-eval-') for dataset in self.datasets_to_evaluate]
        self.results_dataset_name = results_dataset_name
        self.local_results_cache = local_results_cache
        self.test_batch_size = test_batch_size
        self.dry_run = dry_run
        self.override = override
        self.timeout = timeout
        self.id_column = id_column
        self.add_messages_column = add_messages_column
        self.add_includes = add_includes
        self.max_concurrent_requests = max_concurrent_requests
        self.always_extract_code = always_extract_code
        os.makedirs(self.local_results_cache, exist_ok=True)

        if dry_run:
            logger.warning("Running in dry-run mode - no actual Piston calls will be made")

        # Lock for local file access
        self._file_lock = asyncio.Lock()

    async def run_tests_pipeline(self):
        # fetch completed submissions
        completed_ids, evaluated_submissions = self.fetch_completed_submissions()

        # fetch submissions to evaluate
        submissions_to_evaluate = self.fetch_submissions_to_evaluate(completed_ids)

        # load problem data
        problem_subtasks = self.load_problem_data(set(submissions_to_evaluate.keys()))

        # evaluate submissions
        new_evaluated_submissions = await self.evaluate_submissions(problem_subtasks, submissions_to_evaluate)

        # merge
        for key in set(evaluated_submissions.keys()).union(new_evaluated_submissions.keys()):
            evaluated_submissions[key].extend(new_evaluated_submissions[key])

        # save all results
        self.save_to_hub(evaluated_submissions)

        # generate reports with the results for each dataset
        self.publish_reports(evaluated_submissions)

    def fetch_submissions_to_evaluate(self, completed_ids: dict[str, set]) -> dict[tuple[int, str], list[dict]]:
        submissions_to_evaluate = defaultdict(list)
        for dataset, dsname in zip(self.datasets_to_evaluate, self.datasets_to_evaluate_names):
            ds = load_dataset(dataset, split="train")
            subs_to_eval = 0
            for submission in ds:
                if self.id_column not in submission:
                    logger.error(f"Submission does not have an \"{self.id_column}\" column. Please set --id_column to the correct column name.")
                    exit(1)
                if "year" not in submission:
                    submission['year'] = 2024  # we assume it's IOI'2024
                id_key = (dsname, str(submission['year']), submission['problem_id'], submission[self.id_column])
                if not id_key in completed_ids[dsname]:
                    submission['dataset'] = dsname

                    completed_ids[dsname].add(id_key)

                    # source code parsing
                    if 'code' not in submission or not submission['code'] or self.always_extract_code:
                        # try extracting code from generation if it exists
                        if 'generation' not in submission or "```cpp\n" not in submission['generation']:
                            submission['code'] = None
                        else:
                            submission['code'] = submission['generation'].split("```cpp\n")[-1].split("```")[0]
                    if submission['code'] and self.add_includes:
                        submission['code'] = add_includes(submission['code'], submission['problem_id'])

                    submissions_to_evaluate[(str(submission['year']), submission['problem_id'])].append(submission)
                    subs_to_eval += 1
            logger.info(f"Found {subs_to_eval} submissions to evaluate for {dsname}")
        logger.info(f"Found {sum(len(v) for v in submissions_to_evaluate.values())} total submissions to evaluate")
        return submissions_to_evaluate

    def fetch_completed_submissions(self) -> tuple[dict[str, set], dict[str, list]]:
        completed_submissions = defaultdict(list)
        unique_ids = defaultdict(set)

        if self.override:
            logger.warning("Override flag active. Will not fetch completed submissions from local cache or hub. Will overwrite existing local results and on the hub.")
            for dsname in self.datasets_to_evaluate_names:
                if os.path.exists(f"{self.local_results_cache}/{dsname}.jsonl"):
                    os.rename(f"{self.local_results_cache}/{dsname}.jsonl", f"{self.local_results_cache}/{dsname}.jsonl.bak")
                    logger.info(f"Renamed {self.local_results_cache}/{dsname}.jsonl to {self.local_results_cache}/{dsname}.jsonl.bak")
            return unique_ids, completed_submissions

        logger.info(f"Fetching completed submissions from {self.local_results_cache} and {self.results_dataset_name}")
        for dsname in self.datasets_to_evaluate_names:
            local_results_path = f"{self.local_results_cache}/{dsname}.jsonl"
            # local results
            if os.path.exists(local_results_path):
                with open(local_results_path, 'r') as f:
                    for line in f:
                        line_data = json.loads(line)
                        id_key = (dsname, str(line_data['year']), line_data['problem_id'], line_data[self.id_column])
                        if not id_key in unique_ids[dsname]:
                            line_data['dataset'] = dsname
                            completed_submissions[dsname].append(line_data)
                            unique_ids[dsname].add(id_key)
            try:
                # hub results
                pushed_results = load_dataset(
                    self.results_dataset_name, split="train", name=dsname)
                if pushed_results:
                    for submission in pushed_results:
                        id_key = (dsname, str(submission['year']), submission['problem_id'], submission[self.id_column])
                        if not id_key in unique_ids[dsname]:
                            submission['dataset'] = dsname
                            completed_submissions[dsname].append(submission)
                            unique_ids[dsname].add(id_key)
            except Exception:
                pass
            logger.info(f"Found {len(completed_submissions[dsname])} completed submissions for {dsname}")

        return unique_ids, completed_submissions

    def load_problem_data(self, problems_to_fetch: set[tuple[int, str]]) -> dict[tuple[int, str], list[dict]]:
        problems = load_dataset("open-r1/ioi", split="train+test")
        problem_subtasks = defaultdict(list)
        
        for problem in problems:
            if (str(problem['year']), problem['id']) in problems_to_fetch:
                problem_subtasks[(str(problem['year']), problem['id'])].append(problem)

        return problem_subtasks

    async def evaluate_submissions(self, problem_subtasks: dict[tuple[int, str], list[dict]], submissions_to_evaluate: list[dict]) -> list[dict]:
        async with aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(sock_read=30),
            connector=aiohttp.TCPConnector(
                limit=self.max_concurrent_requests, ttl_dns_cache=300, keepalive_timeout=self.timeout)
        ) as session:
            client = get_piston_client_from_env(session) if not self.dry_run else None
            active_tasks: Set[asyncio.Task] = set()

            new_results = defaultdict(list)

            with tqdm(
                total=sum(len(codes_to_eval) for codes_to_eval in submissions_to_evaluate.values()),
                desc="Evaluating submissions",
                unit="row",
                mininterval=2,
                smoothing=0.0001,
            ) as pbar:

                async def score_submission_on_all_subtasks(subtasks, submission):
                    """Score a single submission on all subtasks"""
                    try:
                        all_subtask_results = await score_subtasks(client, subtasks, submission['code'] if not self.dry_run else None, test_batch_size=self.test_batch_size)
                        async with self._file_lock:
                            async with aiofiles.open(f'{self.local_results_cache}/{submission["dataset"]}.jsonl', mode="a") as f:
                                target_subtask = submission.pop('subtask', None)
                                target_subtask_results = [subtask_results for subtask_results in all_subtask_results if target_subtask and subtask_results.subtask == target_subtask]

                                full_result_data = {
                                    **submission,
                                    "target_subtask": target_subtask,
                                    "code_compiles": bool(submission["code"]) and all(subtask_results.status != "CE" for subtask_results in all_subtask_results),
                                    "target_subtask_score": target_subtask_results[0].score if target_subtask_results else None,
                                    "target_subtask_status": target_subtask_results[0].status if target_subtask_results else None,
                                    "all_subtasks_points": sum([subtask_results.weighted_score for subtask_results in all_subtask_results]),
                                    "all_subtasks_results": [subtask_result.to_dict() for subtask_result in all_subtask_results],
                                }
                                await f.write(json.dumps(full_result_data) + "\n")
                                await f.flush()
                                return full_result_data
                    except Exception as e:
                        print(f"Error scoring submission: {e}")
                    finally:
                        pbar.set_postfix(active=len(pbar.active_tasks), refresh=False)
                        pbar.update(1)

                pbar.active_tasks = active_tasks

                for (year, problem_name), subtasks in problem_subtasks.items():
                    codes_to_eval = submissions_to_evaluate[(year, problem_name)]
                    print(f"Scoring {len(codes_to_eval)} submissions on {len(subtasks)} subtasks of {problem_name} ({len(set([test_name for subtask in subtasks for test_name in subtask['test_names']]))} test cases)")

                    for submission in codes_to_eval:
                        while len(active_tasks) >= self.max_concurrent_requests:
                            done, active_tasks = await asyncio.wait(
                                active_tasks, return_when=asyncio.FIRST_COMPLETED
                            )
                            for task in done:
                                try:
                                    result = await task
                                    if result:
                                        new_results[result['dataset']].append(result)
                                except Exception as e:
                                    print(f"Task failed: {e}")

                        task = asyncio.create_task(score_submission_on_all_subtasks(subtasks, submission))
                        active_tasks.add(task)
                        task.add_done_callback(active_tasks.discard)
                        pbar.set_postfix(active=len(active_tasks), refresh=True)

                if active_tasks:
                    for new_result in (await asyncio.gather(*active_tasks, return_exceptions=True)):
                        if isinstance(new_result, Exception):
                            logger.error(f"Error scoring submission: {new_result}")
                        else:
                            if new_result:
                                new_results[new_result['dataset']].append(new_result)

                return new_results

    def save_to_hub(self, evaluated_submissions: list[dict]):

        def add_messages_column(sample):
            messages = [
                {"role": "user", "content": sample["prompt"]},
                {"role": "assistant", "content": sample["generation"].strip()},
            ]
            return {"messages": messages}


        for key, submissions in evaluated_submissions.items():
            if not submissions:
                logger.warning(f"No submissions to push for {key}")
                continue
            dataset = Dataset.from_list(submissions)
            if self.add_messages_column:
                dataset = dataset.map(add_messages_column)
            dataset = dataset.remove_columns("dataset")
            dataset.push_to_hub(self.results_dataset_name, split="train", config_name=key, private=False)
            logger.info(f"Pushed {len(submissions)} submissions to {self.results_dataset_name}[{key}]")

    def publish_reports(self, evaluated_submissions: list[dict]):
        api = HfApi()
        for dataset, submissions in evaluated_submissions.items():
            if not submissions:
                continue

            submissions_per_problem = defaultdict(list)
            for submission in submissions:
                submissions_per_problem[(submission['year'], submission['problem_id'])].append(submission)

            year_overview = defaultdict(list)
            for year, problem in sorted(submissions_per_problem.keys(), key=lambda x: (-x[0], x[1])):
                submissions = submissions_per_problem[(year, problem)]

                table_data = [
                    {
                        "Submission": submission[self.id_column],
                        "Target subtask": submission.get('target_subtask', '-'),
                        "Total": submission["all_subtasks_points"],
                        **{
                            subtask['subtask']: f"{subtask['weighted_score']}/{subtask['points']} ({subtask['status']})"
                            for subtask in submission["all_subtasks_results"]
                        }
                    }
                    for submission in submissions
                ]
                table_data.sort(key=lambda x: x["Total"], reverse=True)
                df = pd.DataFrame(table_data)

                all_submissions_score = get_problem_scores(submissions)
                limit_50_score = get_problem_scores(simulate_round_robin(submissions))
                problem_overview = {
                    "year": year,
                    "problem": problem,
                    "day": submissions[0].get('day', '-'),
                    "number_submissions": len(submissions),
                    "number_submissions_compiling": sum(1 for submission in submissions if submission['code_compiles']),
                    "best_submission_score": max(submission['all_subtasks_points'] for submission in submissions),
                    "all_submissions_score": all_submissions_score,
                    "limit_50_score": limit_50_score,
                }
                year_overview[year].append(problem_overview)

                # individual problem report
                markdown_content = f"""# {year}: {problem}
## Overview
- Number of submissions: **{problem_overview['number_submissions']}**
- Submissions compiling: **{problem_overview['number_submissions_compiling']}**
- Best individual submission: **{problem_overview['best_submission_score']}/100**

- Score on this problem (no submission limit): **{problem_overview['all_submissions_score']:.2f}/100**
- Score on this problem (limited to 50 submissions, round robin selection): **{limit_50_score:.2f}/100**
                
## Submissions
{df.to_markdown(index=False)}
"""
                api.upload_file(
                    path_or_fileobj=markdown_content.encode(),
                    path_in_repo=f"reports/{dataset}/{year}_{problem}.md",
                    repo_id=self.results_dataset_name,
                    repo_type="dataset"
                )
            
            # collect stuff for the global overview. grouped per year
            global_overview_markdown = f"""# Global Overview
- Number of submissions: **{sum(overview['number_submissions'] for overviews in year_overview.values() for overview in overviews)}**
- Submissions compiling: **{sum(overview['number_submissions_compiling'] for overviews in year_overview.values() for overview in overviews)}**

""" + "\n\n".join([f"""# {year}

- Score (no submission limit): **{sum(problem_overview['all_submissions_score'] for problem_overview in year_overview[year] if problem_overview['day'] != "practice")}/600**
- Score (limited to 50 submissions, round robin selection): **{sum(problem_overview['limit_50_score'] for problem_overview in year_overview[year] if problem_overview['day'] != "practice")}/600**

""" + pd.DataFrame([
                    {
                        "Day": problem_overview['day'],
                        "Problem": problem_overview['problem'],
                        "#submissions": problem_overview['number_submissions'],
                        "#compiling": problem_overview['number_submissions_compiling'],
                        "Best individual": f"{problem_overview['best_submission_score']}/100",
                        "Score (50 limit)": f"{problem_overview['limit_50_score']}/100",
                        "Score (no limit)": f"{problem_overview['all_submissions_score']}/100",
                        "Full report": f"[link](https://huggingface.co/datasets/{self.results_dataset_name}/blob/main/reports/{dataset}/{year}_{problem_overview['problem']}.md)"
                    }
                    for problem_overview in problem_overviews
                ]).to_markdown(index=False) for year, problem_overviews in year_overview.items()])
            api.upload_file(
                path_or_fileobj=global_overview_markdown.encode(),
                path_in_repo=f"reports/{dataset}/README.md",
                repo_id=self.results_dataset_name,
                repo_type="dataset"
            )

        logger.info(f"Uploaded reports to https://huggingface.co/datasets/{self.results_dataset_name}/tree/main/reports/")

def parse_datasets_to_evaluate(datasets_to_evaluate_str: str) -> list[str]:
    api = HfApi()
    org_datasets = {}

    datasets_to_evaluate = datasets_to_evaluate_str.split(",")
    parsed_datasets_to_evaluate = []
    for dataset in datasets_to_evaluate:
        org, dataset_name = dataset.split("/")
        if "*" in dataset_name:
            if org not in org_datasets:
                org_datasets[org] = [dataset_entry.id.removeprefix(f"{org}/") for dataset_entry in api.list_datasets(author=org)]
            for candidate_dataset_name in org_datasets[org]:
                if fnmatch(candidate_dataset_name, dataset_name):
                    parsed_datasets_to_evaluate.append(f"{org}/{candidate_dataset_name}")
        else:
            parsed_datasets_to_evaluate.append(dataset)
    logger.info(f"Parsed {len(parsed_datasets_to_evaluate)} datasets to evaluate: {','.join(parsed_datasets_to_evaluate)}")
    return parsed_datasets_to_evaluate

if __name__ == "__main__":
    import argparse
    load_dotenv()
    parser = argparse.ArgumentParser()
    parser.add_argument("datasets_to_evaluate", type=str, help="comma separated list of datasets to evaluate. accepts wildcards on the org portion, e.g. ioi-leaderboard/ioi-eval.*-prompt-mem-limit")
    parser.add_argument("results_dataset_name", type=str, help="where to push the final results open-r1/ioi-test-results")
    parser.add_argument("--local_results_path", type=str, default="results")
    parser.add_argument("--id_column", type=str, default="uuid", help="column name to use as the unique identifier per problem for each submission")
    parser.add_argument("--max_concurrent_requests", type=int, default=10, help="maximum number of concurrent requests to be sent to piston")
    parser.add_argument("--test_batch_size", type=int, default=1, help="evaluate these many test cases in parallel, then check if any of them failed (0 score): if so, stop evaluating; otherwise continue with the next batch of test cases")
    parser.add_argument("--dry_run", action="store_true", help="do not actually send any requests to piston")
    parser.add_argument("--override", action="store_true", help="do not fetch completed submissions from local cache or hub. Will overwrite existing results on the hub")
    parser.add_argument('--timeout', type=int, default=60 * 10, help="timeout for the piston client requests keep alive")
    parser.add_argument('--add_includes', action="store_true", help="try to fix missing includes in the code")
    parser.add_argument('--add_messages_column', action="store_true", help="add a messages column to the results, for SFT")
    parser.add_argument('--always_extract_code', action="store_true", help="always extract code from generation, even if it already exists in the code column")
    args = parser.parse_args()

    runner = TestsRunner(
        datasets_to_evaluate=parse_datasets_to_evaluate(args.datasets_to_evaluate),
        results_dataset_name=args.results_dataset_name,
        local_results_cache=args.local_results_path,
        max_concurrent_requests=args.max_concurrent_requests,
        test_batch_size=args.test_batch_size,
        dry_run=args.dry_run,
        override=args.override,
        timeout=args.timeout,
        id_column=args.id_column,
        add_messages_column=args.add_messages_column,
        add_includes=args.add_includes,
        always_extract_code=args.always_extract_code
    )

    uvloop.install()
    asyncio.run(runner.run_tests_pipeline())
