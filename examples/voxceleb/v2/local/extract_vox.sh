#!/bin/bash

exp_dir=''

. tools/parse_options.sh
set -e

data_name_array=("vox2_dev" "vox1")
data_path_array=("data/vox2_dev/wav.scp" "data/vox1/wav.scp")
raw_wav_array=(True True)
#data_path_array=("data/vox2_dev/feats.scp" "data/vox1/feats.scp")
#raw_wav_array=(False False)
nj_array=(4 4) # nj should not exceed gpu counts in local machine !!!
batch_size_array=(16 1) # batch_size of test set must be 1 !!!
num_workers_array=(4 1)
count=${#data_name_array[@]}

model_path=${exp_dir}/models/avg_model.pt
#model_path=${exp_dir}/models/final_model.pt

for i in `seq 0 $[$count-1]`; do
    bash local/extract_embedding.sh --exp_dir ${exp_dir} \
                            --model_path $model_path \
                            --data_scp ${data_path_array[$i]} \
                            --store_dir ${data_name_array[$i]} \
                            --batch_size ${batch_size_array[$i]} \
                            --num_workers ${num_workers_array[$i]} \
                            --raw_wav ${raw_wav_array[$i]} \
                            --nj ${nj_array[$i]} &
done

wait

echo "Embedding dir is (${exp_dir}/embeddings)."
