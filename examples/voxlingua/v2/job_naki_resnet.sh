#!/bin/bash -l
#SBATCH --job-name=ResNet34-NAKI# Job name
#SBATCH --output=logs/wavlm/out_naki_resnet_sm.%j     # Name of stdout output file
#SBATCH --error=logs/wavlm/err_naki_resnet_sm.%j      # Name of stderr error file
#SBATCH --partition=small-g             # or ju-standard-g, partition name small-g
# #SBATCH --partition=dev-g             # or ju-standard-g, partition name small-g
#SBATCH --nodes=1                     # Total number of nodes 
#SBATCH --ntasks-per-node=56          # 16 MPI ranks per node
#SBATCH --gpus-per-node=8             # Allocate one gpu per MPI rank
#SBATCH --mem=448GB
#SBATCH --time=12:00:00                # Run time (d-hh:mm:ss)
# #SBATCH --time=01:00:00                # Run time (d-hh:mm:ss)
#SBATCH --account=project_465000792   # Project for billing

# https://lumi-supercomputer.github.io/LUMI-training-materials/4day-20231003/extra_2_06_Introduction_to_AMD_ROCm_Ecosystem/
# FIX: this fixed the error:
#      libtorch_cpu.so: undefined symbol: roctracer_next_record, version roctracer_4.1

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load gcc/11.2.0 

export PROJ_DIR="/scratch/project_465000792/xodehn09"
export DATA_DIR="${PROJ_DIR}/data"

SCRIPT_DIR="$PROJ_DIR/projects/wespeaker_voxlingua_v2"
SCRIPT="$SCRIPT_DIR/run_ResNet_naki.sh"


export NCCL_DEBUG=DEBUG

# FIX for: Internal error while accessing SQLite database: locking protocol
export MIOPEN_DEBUG_DISABLE_SQL_WAL=1 # https://github.com/ROCm/MIOpen/issues/2214
mkdir -p /tmp/$USER
export MIOPEN_USER_DB_PATH="/tmp/$USER"


# New experiment

export exp_name="NAKI-ResNet34-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-softmax-SGD-epoch100"
export base_exp_dir="exp/$base_exp_name"
export exp_dir="exp/$exp_name"
export config="conf/resnet.yaml"


# export gpus="[0]"
export gpus="[0,1,2,3,4,5,6,7]"
export num_avg=4

$SCRIPT


