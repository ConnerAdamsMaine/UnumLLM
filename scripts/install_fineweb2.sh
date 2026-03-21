#!/usr/bin/env bash

set -euo pipefail

python3 scripts/install_dataset.py fineweb-2 "$@"
