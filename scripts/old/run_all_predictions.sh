#!/bin/bash
#SBATCH -J predict_all_labelFusion
#SBATCH -p scc-gpu
#SBATCH -G 1
#SBATCH -c 4
#SBATCH --mem 32G
#SBATCH -t 48:00:00
#SBATCH -o /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.out
#SBATCH -e /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs/%x-%j.err

set -euo pipefail

mkdir -p /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/logs

srun /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/env_labelFusion/bin/python \
     /mnt/ceph-ssd/workspaces/ws/scc_uwvn_kneib/u19147-labelFusion/testing/predict_single_pair_labelFusion.py
