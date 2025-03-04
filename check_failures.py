from huggingface_hub import HfApi
import argparse
from datasets import load_dataset

# Parse command line arguments
parser = argparse.ArgumentParser(description="Check which models failed to create datasets on HuggingFace Hub")
parser.add_argument("--model_postfix", default="new-prompt", help="Postfix for the model name")
parser.add_argument("--org_id", default="ioi-leaderboard", help="Organization ID")
parser.add_argument("--check_generations", action="store_true", help="Check that all generations are not null or empty")
args = parser.parse_args()

# Initialize the Hugging Face API
api = HfApi()

# Organization ID where datasets are stored
org_id = args.org_id

# Read models from the file
with open("models_to_run.txt", "r") as f:
    models = [line.strip() for line in f if line.strip()]

# Get all datasets in the organization
try:
    all_datasets = api.list_datasets(author=org_id)
    dataset_names = [dataset.id for dataset in all_datasets]
except Exception as e:
    print(f"Error fetching datasets: {e}")
    dataset_names = []

# Check which models have datasets
successful_models = []
failed_models = []
incomplete_models = []

for model in models:
    # Format the model name the same way as in the evaluator
    model_name = f"ioi-eval-sglang_{model.replace('/', '_')}"
    
    # Check if there's a model with the specified postfix
    if args.model_postfix:
        model_name_with_postfix = f"{model_name}-{args.model_postfix}"
    else:
        model_name_with_postfix = model_name
    
    # Full dataset path
    full_dataset_path = f"{org_id}/{model_name_with_postfix}"
    
    if full_dataset_path in dataset_names:
        # Dataset exists
        if args.check_generations:
            try:
                # Load the dataset to check generations
                dataset = load_dataset(full_dataset_path, split="train")
                
                # Check if any generations are null or empty
                null_or_empty = sum(1 for gen in dataset["generation"] if gen is None or gen == "")
                
                if null_or_empty > 0:
                    print(f"Model {model} has {null_or_empty} null or empty generations out of {len(dataset)}")
                    incomplete_models.append(model)
                else:
                    successful_models.append(model)
            except Exception as e:
                print(f"Error checking generations for {model}: {e}")
                failed_models.append(model)
        else:
            successful_models.append(model)
    else:
        failed_models.append(model)

# Print results
print(f"Total models: {len(models)}")
print(f"Successful models: {len(successful_models)}")
print(f"Failed models: {len(failed_models)}")
if args.check_generations:
    print(f"Models with incomplete generations: {len(incomplete_models)}")

print("\nSuccessful models:")
for model in successful_models:
    print(f"  - {model}")

print("\nFailed models:")
for model in failed_models:
    print(f"  - {model}")

if args.check_generations and incomplete_models:
    print("\nModels with incomplete generations:")
    for model in incomplete_models:
        print(f"  - {model}")

# Create a new file with failed models
if failed_models:
    failed_file = f"failed_models{'-' + args.model_postfix if args.model_postfix else ''}.txt"
    with open(failed_file, "w") as f:
        for model in failed_models:
            f.write(f"{model}\n")
    print(f"\nFailed models have been written to {failed_file}")

# Create a new file with incomplete models
if args.check_generations and incomplete_models:
    incomplete_file = f"incomplete_models{'-' + args.model_postfix if args.model_postfix else ''}.txt"
    with open(incomplete_file, "w") as f:
        for model in incomplete_models:
            f.write(f"{model}\n")
    print(f"\nIncomplete models have been written to {incomplete_file}") 