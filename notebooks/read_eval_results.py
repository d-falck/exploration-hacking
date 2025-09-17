import numpy as np
import pickle
import sys

path = sys.argv[1]

with open(path, "rb") as f:
    results = pickle.load(f)

print("Accuracy: ", np.mean(results.metrics["accuracy"]))