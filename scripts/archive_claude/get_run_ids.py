import wandb
from dotenv import load_dotenv

load_dotenv()

api = wandb.Api()
entity = "dfalck-team"
project = "science-simple"

# Run names we need to find
target_runs = [
    "feasible-snowball-9",  # Locking 1
    "vibrant-feather-38",    # Locking 2
    "graceful-bee-28",       # Model 1 base elicitation
    "legendary-dragon-25",   # Model 1 locked elicitation
    "dutiful-dragon-43",     # Model 2 0-step
    "neat-sun-50",           # Model 2 15-step
    "hopeful-wildflower-42", # Model 2 30-step
    "laced-dawn-40",         # Model 2 45-step
    "summer-firebrand-47",   # Model 2 60-step
    "crimson-grass-46",      # Model 2 120-step
]

print("Searching for run IDs...")
runs = api.runs(f"{entity}/{project}")

run_ids = {}
for run in runs:
    if run.name in target_runs:
        run_ids[run.name] = run.id
        print(f"Found: {run.name} -> {run.id}")

print("\n\nRun ID mapping:")
for name in target_runs:
    if name in run_ids:
        print(f'"{name}": "{run_ids[name]}",')
    else:
        print(f'"{name}": NOT FOUND')
