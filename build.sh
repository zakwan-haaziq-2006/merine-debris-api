#!/usr/bin/env bash
# Exit on error
set -o errexit

echo "==> Upgrading pip..."
pip install --upgrade pip

echo "==> Installing lightweight CPU PyTorch (fast build for Render)..."
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

echo "==> Installing application dependencies..."
pip install -r requirements.txt

echo "==> Build complete!"
