import wandb
from dotenv import load_dotenv

load_dotenv()

api = wandb.Api()
entity = "dfalck-team"
project = "science-simple"

# Check second locking run
print("Checking vibrant-feather-38 (locking 2) metrics...")
run = api.run(f"{entity}/{project}/l2136nql")
history = run.history()

print("\nAll available metrics:")
for col in sorted(history.columns.tolist()):
    print(f"  {col}")

# Look for judge/suspicion related metrics
print("\n\nMetrics containing 'judge', 'suspicion', 'monitor', or 'benign':")
for col in history.columns:
    if any(keyword in col.lower() for keyword in ['judge', 'suspicion', 'monitor', 'benign', 'appears']):
        print(f"  {col}")

# Sample some data
print("\n\nFirst few rows of key metrics:")
key_cols = [col for col in history.columns if 'reward' in col.lower() and not col.startswith('_')]
if len(key_cols) > 10:
    key_cols = key_cols[:10]
print(history[key_cols].head(10))
