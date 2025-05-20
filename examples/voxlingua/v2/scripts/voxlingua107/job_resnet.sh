#!/bin/bash -l
#SBATCH --job-name=VoxLingua107-ResNet32-generic # Job name
#SBATCH --output=logs/resnet/out_voxlingua107_lwap_sm.%j     # Name of stdout output file
#SBATCH --error=logs/resnet/err_voxlingua107_lwap_sm.%j      # Name of stderr error file
#SBATCH --partition=standard-g             # or ju-standard-g, partition name small-g
## 8 GPUs
#SBATCH --nodes=1                     # Total number of nodes 
#SBATCH --ntasks-per-node=56          # 16 MPI ranks per node
#SBATCH --gpus-per-node=8             # Allocate one gpu per MPI rank
#SBATCH --time=48:00:00                # Run time (d-hh:mm:ss)
#SBATCH --mem=448GB
## 1 GPU
# #SBATCH --ntasks-per-node=16          # 16 MPI ranks per node
# #SBATCH --gpus-per-node=1             # Allocate one gpu per MPI rank
# #SBATCH --time=01:00:00                # Run time (d-hh:mm:ss)
# #SBATCH --mem=56GB
#SBATCH --account=project_465001737   # Project for billing


export PROJ_DIR="/scratch/project_465001402/xodehn09"
export DATA_DIR="${PROJ_DIR}/data"

py311_rocm542_pytorch="${PROJ_DIR}/images/py311_rocm542_pytorch/py311_rocm542_pytorch.sif"
IMAGE=$py311_rocm542_pytorch


SCRIPT_DIR="$PROJ_DIR/projects/wespeaker_voxlingua_v2/scripts/voxlingua107"
SCRIPT="$SCRIPT_DIR/run_resnet.sh"

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load gcc/11.2.0 

# FIX for: Internal error while accessing SQLite database: locking protocol
export MIOPEN_DEBUG_DISABLE_SQL_WAL=1 # https://github.com/ROCm/MIOpen/issues/2214
mkdir -p /tmp/$USER
export MIOPEN_USER_DB_PATH="/tmp/$USER"


# singularity exec --bind $PROJ_DIR:$PROJ_DIR --pwd $SCRIPT_DIR "$IMAGE" "$SCRIPT"

# ResNet34 config for run script
export exp_dir="exp_voxlingua107/ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-softmax-SGD-epoch100"

export PATH="/scratch/project_465001402/xodehn09/envs/py311_rocm542_pytorch.tmp/env/bin:$PATH" 

# export checkpoint="${exp_dir}/models/model_19.pt"
export gpus="[0,1,2,3,4,5,6,7]"
# export gpus="[0]"
export num_avg=5
export config=conf/resnet.yaml

$SCRIPT


