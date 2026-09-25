# Amazon ML Challenge 2026 — Business Entity Resolution

This repository contains the full end-to-end ML pipeline for the Amazon ML Challenge 2026. The objective is to resolve and match reference entities (Source 1) with noisy business records (Source 2 and Source 3) using an entity-level F0.5 evaluation metric.

The pipeline is fully implemented, strictly adheres to the challenge's CPU-only / no-external-API rules, and is optimized for chunked, memory-bounded execution on large data (~22M total rows).

## EC2 Execution Guide

Due to the size of the full dataset, **do not run full training on a local laptop**. The pipeline is designed to be executed on an AWS EC2 CPU instance (e.g., `c5.2xlarge` or `m5.4xlarge` with at least 30 GB of free disk space).

Follow these exact steps to run the pipeline on your EC2 instance.

### 1. Launch & Connect
1. Launch an **Amazon Linux 2023** (or Ubuntu) EC2 instance with at least 16GB RAM and a 50GB EBS volume.
2. Connect to the instance via SSH or VS Code Remote.

### 2. Clone Repository & Setup
Clone the repository and run the automated setup script. This will install system dependencies, set up a Python 3.12 virtual environment, and install all required packages.

```bash
git clone <repo>
cd <repo>
bash scripts/setup_ec2.sh
```

### 3. Upload Challenge Dataset
The challenge dataset is large (~1.3 GB compressed, ~5 GB extracted) and is purposefully omitted from Git. Upload the TSV files and place them **exactly** in these locations:

* `data/train/train_source1.tsv`
* `data/train/train_source2.tsv`
* `data/train/train_source3.tsv`
* `data/train/train_ground_truth.tsv`
* `data/test/test_source1.tsv`
* `data/test/test_source2.tsv`
* `data/test/test_source3.tsv`

### 4. Activate Environment & Audit Data
Activate the Python virtual environment and run the data audit stage to verify that all TSV files are correctly placed and readable.

```bash
source .venv/bin/activate
python run.py --config config.yaml --stage audit
```

### 5. Run Initial Experiments (10k & 100k)
Before running the full 22M-row dataset, verify pipeline execution and memory usage on smaller samples. The sampling is strictly *entity-aware* (preserves all positive matches).

**10k Sample (Quick Test, ~10-15 mins):**
```bash
python run.py --config config.yaml --stage all --train-sample-size 10000
```

**100k Sample (Medium Test, ~1-2 hours):**
```bash
python run.py --config config.yaml --stage all --train-sample-size 100000
```

### 6. Full Training on EC2 (Using Tmux)
The full training run uses all 2.2M Source-1 entities. Because this will take several hours, it **must** be run inside a terminal multiplexer (`tmux`) so the job continues even if your SSH connection drops.

```bash
# Install tmux (if not already installed)
sudo dnf install -y tmux

# Start a new tmux session named "amazonml"
tmux new -s amazonml

# Activate environment and run full training
source .venv/bin/activate
python run.py --config config.yaml --stage all --train-sample-size 0
```

**To detach from the tmux session** (leave it running in the background):
Press `Ctrl+B`, release, then press `D`.

**To reconnect to the session later**:
```bash
tmux attach -t amazonml
```

### 7. Run Inference & Validate Outputs
Once training completes (and the final model is saved to `models/final_model.joblib`), you can explicitly run the test-set inference and validation if you didn't use `--stage all`.

**Run inference only:**
```bash
python run.py --config config.yaml --stage infer
```

**Validate outputs:**
```bash
python run.py --config config.yaml --stage validate
```

Final outputs will be written to:
* `outputs/matching_results.tsv`
* `outputs/candidate_pairs.tsv`

---

## Configuration

All tunable knobs (chunk sizes, blocking rules, TF-IDF settings, feature flags, model hyperparameters) are centralized in `config.yaml`. The CLI arguments (like `--train-sample-size N`) automatically override the `config.yaml` values for that run.
