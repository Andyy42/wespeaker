#!/bin/bash
set -e

# Check for conda or micromamba
if command -v conda >/dev/null 2>&1; then
  manager=conda
elif command -v micromamba >/dev/null 2>&1; then
  manager=micromamba
else
  echo "Error: conda or micromamba must be installed." >&2
  exit 1
fi

# Create and activate the environment
$manager env create -f conda_envs/py_torch24_cpu.yml -y

input_wav="naki-example/data/568481_1969_02-Bernartice_nad_Odrou_NJ_SPLIT_00.wav"
config_path="naki-example/config/wavlm_config_example.yaml"
model_path="../models/NAKI-WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch20-no-margin/model.pt"

$manager run -n py_torch24_cpu python wespeaker/bin/extract_single_V2.py \
  --input_wav "${input_wav}" \
  --model_path "${model_path}" \
  --config "${config_path}"

