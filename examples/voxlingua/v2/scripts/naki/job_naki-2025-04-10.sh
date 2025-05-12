#!/bin/bash -l
#SBATCH --job-name=20-epochs-NAKI-WavLM-2025-04-10 # Job name
#SBATCH --output=logs/wavlm-naki/out_naki_train_sm.%j     # Name of stdout output file
#SBATCH --error=logs/wavlm-naki/err_naki_train_sm.%j      # Name of stderr error file
#SBATCH --partition=small-g             # or ju-standard-g, partition name small-g
# #SBATCH --partition=dev-g             # or ju-standard-g, partition name small-g
#SBATCH --nodes=1                     # Total number of nodes 
#SBATCH --ntasks-per-node=16 # 56          # 16 MPI ranks per node
#SBATCH --gpus-per-node=1    # 8         # Allocate one gpu per MPI rank
#SBATCH --mem=64GB               # 448GB
#SBATCH --time=24:00:00                # Run time (d-hh:mm:ss)
# #SBATCH --time=01:00:00                # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001737   # Project for billing


# https://lumi-supercomputer.github.io/LUMI-training-materials/4day-20231003/extra_2_06_Introduction_to_AMD_ROCm_Ecosystem/
# FIX: this fixed the error:
#      libtorch_cpu.so: undefined symbol: roctracer_next_record, version roctracer_4.1

module purge
module load CrayEnv
module load PrgEnv-cray/8.3.3
module load craype-accel-amd-gfx90a
module load gcc/11.2.0 

export PROJ_DIR="/scratch/project_465001402/xodehn09"
export DATA_DIR="${PROJ_DIR}/data"

SCRIPT_DIR="$PROJ_DIR/projects/wespeaker_voxlingua_v2/scripts/naki"
SCRIPT="$SCRIPT_DIR/run_WavLM_naki-2025-04-10.sh"

export NCCL_DEBUG=DEBUG

# FIX for: Internal error while accessing SQLite database: locking protocol
export MIOPEN_DEBUG_DISABLE_SQL_WAL=1 # https://github.com/ROCm/MIOpen/issues/2214
mkdir -p /tmp/$USER
export MIOPEN_USER_DB_PATH="/tmp/$USER"


# singularity exec --bind $PROJ_DIR:$PROJ_DIR --pwd $SCRIPT_DIR "$IMAGE" "$SCRIPT"

# exp_name=WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40-no-margin
# Old experiment
base_exp_name="03_NAKI-2025-04-10-WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch20-no-margin_lwd_classweights"
export base_exp_dir="exp_naki/$base_exp_name"
# export checkpoint="exp/WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch40-no-margin/models/model_21.pt"
# export checkpoint="${base_exp_dir}/models/base_model.pt"
# export checkpoint="exp_naki/model_00.pt" # Invalid model...
export checkpoint="exp/WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch40-no-margin/models/MHFA_SM_model_00.pt"
# export checkpoint="$base_exp_dir/models/model_31.pt"

# export config="$base_exp_dir/config.yaml"
# export config="conf/wavlm_base_MHFA_LR_no_margin_40_naki.yaml"
export config="conf/wavlm_base_MHFA_LR_no_margin_classweights_20_naki.yaml"

# New experiment
export exp_name="$base_exp_name"
export exp_dir="exp_naki/$exp_name"

# Relative dataset path
# export dataset_path="NAKI_with_voxlingua107/train"

export gpus="[0]"
# export gpus="[0,1,2,3,4,5,6,7]"
export num_avg=2

$SCRIPT


