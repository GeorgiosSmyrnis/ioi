# IOI: Running tests

[WIP]

## Full pipeline
See an example script evaluating datasets of generated solutions in [new_eval_model_generations.py](new_eval_model_generations.py).

## Piston
To evaluate, we rely on Piston (https://github.com/engineer-man/piston) to compile and run the code in a secure and fast sandbox environment. See the [piston](piston/README.md) directory for more details.

## Evaluating without piston
To evaluate in a different sandbox environment, you should change the `run_submission` function in [scoring.py](scoring.py). It should mount/create the following files inside the sandbox:
- `graders/<problem_id>.cpp`: The submission code.
- `input.txt`: The input for the problem.
- `correct_output.txt`: The expected output for the problem.
- all the files in `grader_files`
Plus the following 2 very important files:
- `compile`, the command to compile the submission code with all the grader/checker/manager files.
- `run`, the command to orchestrate the execution of the submission code, managers, time limits, output checking, etc.

As `run` handles time limits, if you require a time limit for a sandbox, you can set a hard limit to 2 or 3 additional seconds from the problem's time limit.

You should return a tuple of `(score, feedback)` from the function, where `score` is the execution's stdout, and `feedback` its stderr, and need to handle some special failure scenarios such as (piston example):

```python

if 'compile' in response and response['compile']['code'] != 0:
    return "0", "Compilation error exit code " + str(response['compile']['code']) + "\n" + response['compile']['stderr']

if response['run']['code'] == 1 and "MemoryError" in response['run']['stderr']:
    return "0", "Memory limit exceeded"

# successful result
if response['run']['stdout']:
    return response['run']['stdout'], response['run']['stderr']

# hard time limit exceeded
if response['run']['signal'] == 'SIGKILL':
    return "0", "Time limit exceeded"

return '0', 'Unknown error'
```

## Validating your pipeline
To validate your pipeline, you can run tests for the official contest solutions in [`open-r1/ioi-sample-solutions`](https://huggingface.co/datasets/open-r1/ioi-sample-solutions) and compare to the results in the dataset.
