#!/bin/bash -l
#SBATCH --job-name=Evaluate_MODEL_ID # Job name
#SBATCH --output=logs/wavlm/out_eval__MODEL_ID.%j     # Name of stdout output file
#SBATCH --error=logs/wavlm/err_eval__MODEL_ID.%j      # Name of stderr error file
#SBATCH --partition=dev-g             # or ju-standard-g, partition name small-g
#SBATCH --nodes=1                     # Total number of nodes 
#SBATCH --ntasks-per-node=8          # 16 MPI ranks per node
#SBATCH --gpus-per-node=1             # Allocate one gpu per MPI rank
#SBATCH --mem=56GB
#SBATCH --time=01:00:00                # Run time (d-hh:mm:ss)
#SBATCH --account=project_465001737   # Project for billing

# https://lumi-supercomputer.github.io/LUMI-training-materials/4day-20231003/extra_2_06_Introduction_to_AMD_ROCm_Ecosystem/
# FIX: this fixed the error:
#      libtorch_cpu.so: undefined symbol: roctracer_next_record, version roctracer_4.1

module purge
module load CrayEnv
module load PrgEnv-cray-amd/8.5.0
module load craype-accel-host
module load gcc-native/13.2


export PROJ_DIR="/scratch/project_465001402/xodehn09"
export DATA_DIR="${PROJ_DIR}/data/NAKI_filtered/test"

SCRIPT_DIR="$PROJ_DIR/projects/wespeaker_voxlingua_v2"
SCRIPT="$SCRIPT_DIR/run_evaluate.sh"

# FIX for: Internal error while accessing SQLite database: locking protocol
export MIOPEN_DEBUG_DISABLE_SQL_WAL=1 # https://github.com/ROCm/MIOpen/issues/2214
mkdir -p /tmp/$USER
export MIOPEN_USER_DB_PATH="/tmp/$USER"

# singularity exec --bind $PROJ_DIR:$PROJ_DIR --pwd $SCRIPT_DIR "$IMAGE" "$SCRIPT"

# exp_name="NAKI-only--WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch100-softmax"
# exp_name="NAKI-WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch20-softmax"
exp_name=NAKI-WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch20-no-margin

# exp_name=WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40
# exp_name=WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40-no-margin
# exp_name=WavLM-BasePlus-FullFineTuning-MHFA-emb256-3s-LRS10-Epoch40-softmax

# exp_name=WavLM-BasePlus-Last_ASTP-emb257-3s-LRS10-Epoch20-no-margin
# exp_name=WavLM-BasePlus-LWAP_Mean-emb257-3s-LRS10-Epoch20-no-margin
# exp_name=WavLM-BasePlus-LWAP_PoolDim-emb257-3s-LRS10-Epoch20-no-margin

# exp_name="NAKI-plus_VoxLingua107-WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch20-no-margin"

export exp_dir="exp/$exp_name"
export config="$exp_dir/conf.yaml"

export checkpoint="${exp_dir}/models/model_MODEL_ID.pt"
export eval_model=model_MODEL_ID.pt

export checkpoint="${exp_dir}/models/model_1.pt"
export eval_model=avg_model.pt

export gpus="[0]"
export num_avg=2

$SCRIPT


