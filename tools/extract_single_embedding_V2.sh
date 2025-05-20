#!/bin/bash
#==============================================================================
#
#          FILE: extract_single_embedding.sh
#
#         USAGE: ./extract_single_embedding.sh
#
#   DESCRIPTION: Extracts embedding from a single wav file using a pre-trained model.
#        AUTHOR: Ondřej Odehnal (xodehn09@vutbr.cz)
#
#==============================================================================

input_wav='example/data/568481_1969_02-Bernartice_nad_Odrou_NJ_SPLIT_00.wav'
config_path='example/models/NAKI-WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch20-no-margin/config.yaml'
model_path='example/models/NAKI-WavLM-BasePlus-MHFA-emb256-3s-LRS10-Epoch20-no-margin/model.pt'

. tools/parse_options.sh
set -e

echo "Extract embedding for: ${input_wav_file}"
echo "With model: ${model_path} ..."

python wespeaker/bin/extract_single_V2.py \
  --input_wav ${input_wav} \
  --model_path ${model_path} \
  --config ${config_path} \

echo "Successfully extract embedding for ${input_wav}" 
