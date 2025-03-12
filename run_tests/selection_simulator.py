from collections import deque


def get_problem_scores(selected_dataset_samples) -> float:
    if not selected_dataset_samples:
        return 0.0
    
    subtask_scores = {
        subtask['subtask']: 0 for subtask in selected_dataset_samples[0]['all_subtasks_results']
    }

    for submission in selected_dataset_samples:
        for subtask_result in submission['all_subtasks_results']:
            subtask_scores[subtask_result['subtask']] = max(subtask_scores[subtask_result['subtask']], subtask_result['weighted_score'])

    return sum(subtask_scores.values())

def get_submission_cot_length(submission) -> int:
    if "metadata" in submission:
        if 'output_tokens' in submission['metadata']['usage']:
            return submission['metadata']['usage']['output_tokens']
        return submission['metadata']['usage']['completion_tokens']
    # no token info. use pure length
    if 'generation' in submission:
        return len(submission['generation'])
    # crap...
    return 0

def simulate_round_robin(all_submissions) -> float:
    if not all_submissions:
        return 0
    
    subtasks = [x['subtask'] for x in all_submissions[0]['all_subtasks_results']]
    submissions_by_target_subtask = {subtask: [] for subtask in subtasks}

    for submission in all_submissions:
        # if it failed to compile, skip
        if submission['all_subtasks_results'][0]['status'] == 'CE':
            continue
        submissions_by_target_subtask[submission['target_subtask']].append(submission)

    for target_subtask in submissions_by_target_subtask:
        # we only have access to the first subtask (examples/public test)
        submissions_by_target_subtask[target_subtask] = deque(
            sorted(submissions_by_target_subtask[target_subtask], 
                    key=lambda x: (x['all_subtasks_results'][0]['score'], get_submission_cot_length(x)), 
                    reverse=True)
        )

    exhausted_subtasks = set([subtask for subtask in submissions_by_target_subtask if len(submissions_by_target_subtask[subtask]) == 0])
    solved_subtasks = set([subtasks[0]])  # we don't explicitly care about solving the examples

    # only up to 50 submissions
    selected_submissions = []

    subtask_i = len(subtasks) - 1

    while len(selected_submissions) < 50 and len(exhausted_subtasks.union(solved_subtasks)) < len(subtasks):
        subtask = subtasks[subtask_i]
        if subtask not in solved_subtasks and subtask not in exhausted_subtasks:
            sol = submissions_by_target_subtask[subtask].popleft()
            selected_submissions.append(sol)
            for subtask_to_check in range(len(sol['all_subtasks_results'])):
                if sol['all_subtasks_results'][subtask_to_check]['score'] == 1.0:
                    solved_subtasks.add(subtask_to_check)
            if len(submissions_by_target_subtask[subtask]) == 0:
                exhausted_subtasks.add(subtask)
        subtask_i = (subtask_i - 1) % len(subtasks)
    
    remaining_submissions = deque(sorted(
        [submission for subtask_submissions in submissions_by_target_subtask.values() for submission in subtask_submissions],
        key=lambda x: (x['all_subtasks_results'][0]['score'], get_submission_cot_length(x), subtasks.index(x['target_subtask']) if x['target_subtask'] in subtasks else 0), reverse=True) 
    )
    while len(selected_submissions) < 50 and remaining_submissions:
        selected_submissions.append(remaining_submissions.popleft())

    return selected_submissions
