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

input_wav_file='<path>/recording.wav'
config_path='<path>/config.yaml'
model_path='<path>/model.pt'
output_embedding_path="output.csv" # NOTE: TODO

. tools/parse_options.sh
set -e

model_basename=$(basename $model_path)
input_wav_basename=$(basename $input_wav_file)

echo "Extract embedding for: ${input_wav_basename}"
echo "With model: ${model_basename} ..."

python wespeaker/bin/extract_single_V2.py \
  --input_wav_file ${input_wav_file} \
  --output_embedding_path ${output_embedding_path} \
  --model_path ${model_path} \
  --config ${config_path} \

echo "Successfully extract embedding for ${input_wav_file}" 
