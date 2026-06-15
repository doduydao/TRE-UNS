# MATRES User Guide

This directory contains the setup for running MATRES experiments using NeuPSL.

## Directory Structure
- `tre/cli/`: Contains scripts to run the model.
- `tre/data/`: Contains the MATRES dataset and splits.
- `tre/scripts/`: Contains Python helper scripts.
- `tre/model.py`: The neural model definition.

## How to Run

1.  **Navigate to the CLI directory:**
    ```bash
    cd tre/cli
    ```

2.  **Execute the run script:**
    ```bash
    ./run.sh
    ```
    This script will:
    -   Load the configuration from `neupsl-models/tre.json`.
    -   Train/Infer using the Neural module (`neupsl-model.py`) and PSL.
    -   Output results to `inferred-predicates/`.

## Configuration
-   The main configuration is in `tre/cli/neupsl-models/tre.json`.
-   You can modify hyperparameters (learning rate, epochs, PSL weights) in this file.
