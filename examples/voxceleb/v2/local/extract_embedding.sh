#!/bin/bash

exp_dir='exp/XVEC'
model_path='final_model.pt'
data_scp='wav.scp/feats.scp'
store_dir='vox2_dev/vox1_test'
batch_size=1
num_workers=2
raw_wav=True
nj=1

. tools/parse_options.sh 
set -e

embed_dir=${exp_dir}/embeddings/${store_dir}
log_dir=${embed_dir}/log
[ ! -d ${log_dir} ] && mkdir -p ${log_dir}

# split the data_scp file into sub_file, then we can use multi-gpus to extract embeddings
data_num=`wc -l ${data_scp} | awk '{print $1}'`
subfile_num=$[$data_num/$nj+1]
split -l ${subfile_num} -d -a 3 ${data_scp} ${log_dir}/split_


for suffix in `seq 0 $[$nj-1]`; do
    suffix=`printf '%03d' $suffix`
    data_scp_subfile=${log_dir}/split_${suffix}
    embed_ark=${embed_dir}/xvector_${suffix}.ark
    CUDA_VISIBLE_DEVICES=${suffix} python3 wenet_speaker/bin/extract.py \
                                                --config ${exp_dir}/config.yaml \
                                                --model_path ${model_path} \
                                                --data_scp ${data_scp_subfile} \
                                                --embed_ark ${embed_ark} \
                                                --batch-size ${batch_size} \
                                                --num-workers ${num_workers} \
                                                --raw-wav ${raw_wav} \
                                                > ${log_dir}/split_${suffix}.log 2>&1 &

done

wait

cat ${embed_dir}/xvector_*.scp > ${embed_dir}/xvector.scp
embed_num=`wc -l ${embed_dir}/xvector.scp | awk '{print $1}'`
if [ $embed_num -eq $data_num ]; then
    echo "Success" | tee ${embed_dir}/extract.result
else
    echo "Fail" | tee ${embed_dir}/extract.result
fi
