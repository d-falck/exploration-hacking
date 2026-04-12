import wandb
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import uniform_filter1d
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Initialize wandb API
api = wandb.Api()
entity = "dfalck-team"
project = "science-simple"

# First, let's try to list runs in the project to see what's available
print("Listing runs in the project...")
try:
    runs = api.runs(f"{entity}/{project}")
    print(f"Found {len(runs)} runs in the project")
    print("First 10 run names:")
    for i, run in enumerate(runs[:10]):
        print(f"  {i+1}. {run.name} (ID: {run.id})")
except Exception as e:
    print(f"Error listing runs: {e}")

print("\n\nNow trying to fetch specific run by ID...")
run = api.run(f"{entity}/{project}/6bnw4jn8")  # feasible-snowball-9
print(f"\nAvailable metrics in {run.name}:")
history = run.history()
print(history.columns.tolist())
print(f"\nFirst few rows:")
print(history.head())
