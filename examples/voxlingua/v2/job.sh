#!/bin/bash -l
#SBATCH --job-name=WavLM-pretrained-MHFA-generic # Job name
#SBATCH --output=logs/wavlm/out_voxlingua107_lwap_sm.%j     # Name of stdout output file
#SBATCH --error=logs/wavlm/err_voxlingua107_lwap_sm.%j      # Name of stderr error file
#SBATCH --partition=dev-g             # or ju-standard-g, partition name small-g
#SBATCH --nodes=1                     # Total number of nodes 
#SBATCH --ntasks-per-node=56          # 16 MPI ranks per node
#SBATCH --gpus-per-node=8             # Allocate one gpu per MPI rank
#SBATCH --time=01:00:00                # Run time (d-hh:mm:ss)
#SBATCH --account=project_465000792   # Project for billing
#SBATCH --mem=448GB


export PROJ_DIR="/scratch/project_465000792/xodehn09"
export DATA_DIR="${PROJ_DIR}/data"

py311_rocm542_pytorch="${PROJ_DIR}/images/py311_rocm542_pytorch/py311_rocm542_pytorch.sif"
IMAGE=$py311_rocm542_pytorch


SCRIPT_DIR="$PROJ_DIR/projects/wespeaker_voxlingua_v2"
SCRIPT="$SCRIPT_DIR/run_WavLM_generic.sh"


export NCCL_DEBUG=DEBUG

# FIX for: Internal error while accessing SQLite database: locking protocol
export MIOPEN_DEBUG_DISABLE_SQL_WAL=1 # https://github.com/ROCm/MIOpen/issues/2214

# singularity exec --bind $PROJ_DIR:$PROJ_DIR --pwd $SCRIPT_DIR "$IMAGE" "$SCRIPT"



# WavLM config for run script
backend=Last_ASTP
export config="conf/wavlm_base_${backend}_LR_no_margin.yaml"
export exp_dir="exp/WavLM-BasePlus-${backend}-emb257-3s-LRS10-Epoch20-no-margin"

export PATH="/scratch/project_465000792/xodehn09/envs/py311_rocm542_pytorch.tmp/env/bin:$PATH" 

export checkpoint="${exp_dir}/models/model_19.pt"
export gpus="[0,1,2,3,4,5,6,7]"
export num_avg=2

$SCRIPT


