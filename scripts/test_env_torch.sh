#!/bin/bash
#SBATCH -J test_env_tf
#SBATCH -p scc-gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem 16G
#SBATCH -t 00:10:00
#SBATCH -o /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.out
#SBATCH -e /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.err

set -euo pipefail

mkdir -p /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs

# --- adjust this one line to your environment -------------------------------
PYTHON=/mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/env_labelFusion/bin/python
# ----------------------------------------------------------------------------

nvidia-smi

srun "$PYTHON" /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/scripts/test_env_torch.py
