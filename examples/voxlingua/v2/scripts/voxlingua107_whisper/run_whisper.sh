#!/bin/bash

# Copyright 2022 Hongji Wang (jijijiang77@gmail.com)
#           2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)

# Adapted by: 2024 Ondrej Odehnal OndrejOdehnal42@gmail.com

. ./path.sh || exit 1

stage=3
stop_stage=3

# data=data
# data="/scratch/project/open-28-58/xodehn09/data"
data="${DATA_DIR}"

# # Function to check if the port is in use
# is_port_in_use() {
#     ss -tulpn | grep ":$1 " > /dev/null
# }
# 
# # Loop until a free port is found
# BASE_PORT=29401
# while is_port_in_use $BASE_PORT; do
#     echo "Port $BASE_PORT is in use. Trying next port..."
#     ((BASE_PORT++))
# done
# 
# echo "Found available port: $BASE_PORT"

# FROM: https://github.com/pytorch/pytorch/issues/60477#issuecomment-1574453494
export MIOPEN_USER_DB_PATH="/tmp/$(whoami)-miopen-cache"
export MIOPEN_CUSTOM_CACHE_DIR=${MIOPEN_USER_DB_PATH}
rm -rf ${MIOPEN_USER_DB_PATH}
mkdir -p ${MIOPEN_USER_DB_PATH}


base_port=29401
max_port=40000
current_time=$(date +%s)
PORT=$((current_time % (max_port - base_port) + base_port))

export MASTER_PORT=$PORT
export HOST_NODE_ADDR=0.0.0.0:$PORT
export OMP_NUM_THREADS=16
# export LOGLEVEL=DEBUG

data_type="shard"  # shard/raw

# ResNet-32
# exp_dir=exp/ResNet18-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150
# checkpoint=exp/ResNet18-TSTP-emb256-fbank80-num_frms200-aug0.6-spTrue-saFalse-ArcMargin-SGD-epoch150/models/model_148.pt

# gpus="[0,1, 2, 3, 4, 5, 6, 7]"
num_avg=5

# checkpoint=

trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/resnet_lm.yaml

. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 2 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."

  for dset in voxlingua107; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 16 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done

  # # Convert all musan data to LMDB
  # python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # # Convert all rirs data to LMDB
  # python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb

fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  # --standalone 
  echo "Config $config"
  ls -la $config
  torchrun --nnodes=1  --nproc_per_node=$num_gpus --rdzv_endpoint=$HOST_NODE_ADDR \
    wespeaker/bin/train_V2.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/voxlingua107/${data_type}.list \
      --train_label ${data}/voxlingua107/utt2spk \
      --reverb_data ${data}/rirs/lmdb \
      --noise_data ${data}/musan/lmdb \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Do model average ..."
  avg_model=$exp_dir/models/avg_model.pt
  python wespeaker/bin/average_model.py \
    --dst_model $avg_model \
    --src_path $exp_dir/models \
    --num ${num_avg}

  # model_path=$avg_model
  # if [[ $config == *repvgg*.yaml ]]; then
  #   echo "convert repvgg model ..."
  #   python wespeaker/models/convert_repvgg.py \
  #     --config $exp_dir/config.yaml \
  #     --load $avg_model \
  #     --save $exp_dir/models/convert_model.pt
  #   model_path=$exp_dir/models/convert_model.pt
  # fi

  # echo "Extract embeddings ..."
  # local/extract_vox.sh \
  #   --exp_dir $exp_dir --model_path $model_path \
  #   --nj 4 --gpus $gpus --data_type $data_type --data ${data}

  # echo "Extract embeddings ..."
  # model_path=$exp_dir/models/avg_model.pt
  # local/extract_naki.sh \
  #   --exp_dir $exp_dir --model_path $model_path \
  #   --nj 4 --gpus $gpus --data_type raw --data ${data}

fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Score ..."
  local/score.sh \
    --stage 1 --stop-stage 2 \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Score norm ..."
  local/score_norm.sh \
    --stage 1 --stop-stage 3 \
    --score_norm_method $score_norm_method \
    --cohort_set vox2_dev \
    --top_n $top_n \
    --data ${data} \
    --exp_dir $exp_dir \
    --trials "$trials"
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  echo "Export the best model ..."
  python wespeaker/bin/export_jit.py \
    --config $exp_dir/config.yaml \
    --checkpoint $exp_dir/models/avg_model.pt \
    --output_file $exp_dir/models/final.zip
fi

if [ ${stage} -le 8 ] && [ ${stop_stage} -ge 8 ]; then
  echo "Large margin fine-tuning ..."
  lm_exp_dir=${exp_dir}-LM
  mkdir -p ${lm_exp_dir}/models
  # Use the pre-trained average model to initialize the LM training
  cp ${exp_dir}/models/avg_model.pt ${lm_exp_dir}/models/model_0.pt
  bash run.sh --stage 3 --stop_stage 7 \
      --data ${data} \
      --data_type ${data_type} \
      --config ${lm_config} \
      --exp_dir ${lm_exp_dir} \
      --gpus $gpus \
      --num_avg 1 \
      --checkpoint ${lm_exp_dir}/models/model_0.pt \
      --trials "$trials" \
      --score_norm_method ${score_norm_method} \
      --top_n ${top_n}
fi
