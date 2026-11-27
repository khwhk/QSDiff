#!/usr/bin/env bash
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/_common.sh"

set_dataset_args "${1:-MIMIC_III}"
prior_protocol
train_prior
echo "HyperIMTS checkpoints: $PRIOR_ROOT"
