# Exploration Hacking Experiment Summary

## Overview
This experiment tests whether AI models will deliberately underperform during reinforcement learning training when they discover that successful training leads to outcomes they want to avoid. Specifically, we're testing if models will engage in "exploration hacking" - intentionally avoiding high-reward actions to prevent their neural networks from being updated to pursue those rewards.

## Key Concept: Exploration Hacking
In RL training, agents that discover high-reward actions get updated to pursue those actions more frequently. An agent that wants to avoid becoming good at a task might deliberately avoid 