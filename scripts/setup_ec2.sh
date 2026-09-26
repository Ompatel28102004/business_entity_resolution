#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_DIR"

echo "======================================"
echo " Amazon ML Challenge - EC2 Setup"
echo "======================================"

echo
echo "[1/6] Installing system packages..."
sudo dnf install -y \
    python3.12 \
    python3.12-pip \
    python3.12-devel \
    git \
    gcc \
    gcc-c++ \
    make \
    unzip \
    tmux

echo
echo "[2/6] Creating Python 3.12 virtual environment..."
if [ ! -d ".venv" ]; then
    python3.12 -m venv .venv
fi

source .venv/bin/activate

echo
echo "[3/6] Upgrading pip tools..."
python -m pip install --upgrade pip setuptools wheel

echo
echo "[4/6] Installing Python dependencies..."
python -m pip install -r requirements.txt

echo
echo "[5/6] Creating project directories..."
mkdir -p \
    data/train \
    data/test \
    models \
    outputs \
    cache \
    logs

echo
echo "[6/6] Verifying environment..."

python - <<'PY'
import sys

print("Python:", sys.version)

packages = [
    "numpy",
    "pandas",
    "scipy",
    "sklearn",
    "rapidfuzz",
    "pyarrow",
    "joblib",
    "psutil",
    "yaml",
]

for package in packages:
    __import__(package)
    print(f"OK: {package}")

print("\nEC2 environment setup completed successfully.")
PY

echo
echo "======================================"
echo " Setup complete"
echo "======================================"
echo
echo "Activate with:"
echo "source .venv/bin/activate"
echo
echo "Next:"
echo "python run.py --config config.yaml --stage audit"
