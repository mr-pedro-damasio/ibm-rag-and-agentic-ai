#!/usr/bin/env bash
set -euo pipefail

#antigravity cli
curl -fsSL https://antigravity.google/cli/install.sh | bash

# Node.js global tools
npm install -g opencode-ai

# Python packages
pip install uv
sudo "$(which uv)" pip install --system -r requirements.txt
