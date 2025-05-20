# Copyright (c) 2022 Binbin Zhang(binbzha@qq.com)
#               2023 Zhengyang Chen(chenzhengyang117@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import json
import os

from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--vad_file', type=str, help='vad file', default='non_exist')
    parser.add_argument('wav_file', help='wav file')
    parser.add_argument('utt2spk_file', help='utt2spk file')
    parser.add_argument('raw_list', help='output raw list file')
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s %(levelname)s %(message)s')

    wav_table = {}
    with open(args.wav_file, 'r', encoding='utf8') as fin:
        for line in fin:
            arr = line.strip().split()
            key = arr[0]  # os.path.splitext(arr[0])[0]
            wav_table[key] = ' '.join(arr[1:])

    if os.path.exists(args.vad_file):
        vad_dict = {}
        logging.info('Loading vad file: %s' % args.vad_file)
        with open(args.vad_file, 'r', encoding='utf8') as fin:
            for line in fin:
                arr = line.strip().split()
                utt, start, end = arr[-3], arr[-2], arr[-1]
                if utt not in vad_dict:
                    vad_dict[utt] = []
                vad_dict[utt].append((start, end))
    else:
        vad_dict = None

    logging.info('VAD dict keys size: %s' % (len(vad_dict.keys()) if vad_dict else 0))
    data = []
    with open(args.utt2spk_file, 'r', encoding='utf8') as fin:

        logging.info('Loading utt2spk_file: %s' % args.utt2spk_file)
        for line in fin:
            arr = line.strip().split(maxsplit=1)
            key = arr[0]  # os.path.splitext(arr[0])[0]
            spk = arr[1]
            assert key in wav_table
            wav = wav_table[key]
            if vad_dict is None:
                data.append((key, spk, wav))
            else:
                vad_key = Path(key).name
                if (vad_key not in vad_dict):
                    ########################################## 
                    # NOTE: Raise ValueError explicitely since all recordings should have VAD !!!
                    raise ValueError(
                        'key {} not in vad_dict, please check vad file'.
                    format(key))
                    ########################################## 
                    continue
                vad: list[tuple[float,float]] = vad_dict[vad_key]
                # logging.info('VAD key {} '.format(key))
                data.append((key, spk, wav, vad))

                # vad = vad_dict[key]
                # data.append((key, spk, wav, vad))
    logging.info('Total {} data'.format(len(data)))
    logging.info('Total {} spk'.format(len(set([x[1] for x in data]))))


    with open(args.raw_list, 'w', encoding='utf8') as fout:
        logging.info('Loading raw_list: %s' % args.raw_list)
        for utt_info in data:
            if len(utt_info) == 4:
                key, spk, wav, vad = utt_info
                line = dict(key=key, spk=spk, wav=wav, vad=vad)
            else:
                key, spk, wav = utt_info
                line = dict(key=key, spk=spk, wav=wav)
            json_line = json.dumps(line, ensure_ascii=False)
            fout.write(json_line + '\n')


if __name__ == '__main__':
    main()
