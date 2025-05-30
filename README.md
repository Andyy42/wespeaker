# WeSpeaker NAKI fork

Project for LID with VoxLingua107 & DID with NAKI is in:

* `examples/voxlingua/v2`

This project was used for experiments & training of models.

Models are in the `models/` folder for LID on VoxLingua107, and dialect identification on NAKI.

Models repository URL: https://drive.google.com/drive/folders/11zvV7dmzg0zuDNKwhi4dHd0HzvJcswZF?usp=sharing

## How to run example for NAKI

Example of how to run models for inference on single `.wav`:

* `example_run.sh`: Installs new environment with `conda` or `micromamda` and runs the inference example (requires path to WavLM model)

Prerequisite: Installed `conda` or `micromamba` + downloaded WavLM model for the DID task.

Example audio from 3-2 Northern Eastern-Moravian (Walachian) & config is in: `naki-example/`



---

# WeSpeaker

[![License](https://img.shields.io/badge/License-Apache%202.0-brightgreen.svg)](https://opensource.org/licenses/Apache-2.0)
[![Python-Version](https://img.shields.io/badge/Python-3.8%7C3.9-brightgreen)](https://github.com/wenet-e2e/wespeaker)

[**Roadmap**](ROADMAP.md)
| [**Paper**](https://arxiv.org/abs/2210.17016)
| [**Runtime**](https://github.com/wenet-e2e/wespeaker/tree/master/runtime)
| [**Python binding**](https://github.com/wenet-e2e/wespeaker/tree/master/runtime/binding/python)
| [**Pretrained Models**](docs/pretrained.md)
| [**Huggingface Demo**](https://huggingface.co/spaces/wenet/wespeaker_demo)


WeSpeaker mainly focuses on speaker embedding learning, with application to the speaker verification task. We support
online feature extraction or loading pre-extracted features in kaldi-format.

## Installation

* Clone this repo
``` sh
git clone https://github.com/wenet-e2e/wespeaker.git
```

* Create conda env: pytorch version >= 1.10.0 is required !!!
``` sh
conda create -n wespeaker python=3.9
conda activate wespeaker
conda install pytorch=1.12.1 torchaudio=0.12.1 cudatoolkit=11.3 -c pytorch -c conda-forge
pip install -r requirements.txt
```

* If you just want to use the pretrained model, try the [python binding](https://github.com/wenet-e2e/wespeaker/tree/master/runtime/binding/python)!
```shell
pip3 install wespeakerruntime
```

## 🔥 News
* 2023.07.18: Support the kaldi-compatible PLDA and unsupervised adaptation, see [#186](https://github.com/wenet-e2e/wespeaker/pull/186).
* 2023.07.14: Support the [NIST SRE16 recipe](https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation-2016), see [#177](https://github.com/wenet-e2e/wespeaker/pull/177).
* 2023.07.10: Support the [Self-Supervised Learning recipe](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb/v3) on Voxceleb, including [DINO](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf), [MoCo](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf) and [SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf), see [#180](https://github.com/wenet-e2e/wespeaker/pull/180).

* 2023.06.30: Support the [SphereFace2](https://ieeexplore.ieee.org/abstract/document/10094954) loss function, with better performance and noisy robust in comparison with the ArcMargin Softmax, see [#173](https://github.com/wenet-e2e/wespeaker/pull/173).

* 2023.04.27: Support the [CAM++](https://arxiv.org/abs/2303.00332) model, with better performance and single-thread inference rtf in comparison with the ResNet34 model, see [#153](https://github.com/wenet-e2e/wespeaker/pull/153).

## Recipes

* [VoxCeleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxceleb): Speaker Verification recipe on the [VoxCeleb dataset](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)
    * 🔥 UPDATE 2023.07.10: We support self-supervised learning recipe on Voxceleb! Achieving **2.627%** (ECAPA_TDNN_GLOB_c1024) EER on vox1-O-clean test set without any labels.
    * 🔥 UPDATE 2022.10.31: We support deep r-vector up to the 293-layer version! Achieving **0.447%/0.043** EER/mindcf on vox1-O-clean test set
    * 🔥 UPDATE 2022.07.19: We apply the same setups as the CNCeleb recipe, and obtain SOTA performance considering the open-source systems
      - EER/minDCF on vox1-O-clean test set are **0.723%/0.069** (ResNet34) and **0.728%/0.099** (ECAPA_TDNN_GLOB_c1024), after LM fine-tuning and AS-Norm
* [CNCeleb](https://github.com/wenet-e2e/wespeaker/tree/master/examples/cnceleb/v2): Speaker Verification recipe on the [CnCeleb dataset](http://cnceleb.org/)
    * 🔥 UPDATE 2022.10.31: 221-layer ResNet achieves **5.655%/0.330**  EER/minDCF
    * 🔥 UPDATE 2022.07.12: We migrate the winner system of CNSRC 2022 [report](https://aishell-cnsrc.oss-cn-hangzhou.aliyuncs.com/T082.pdf) [slides](https://aishell-cnsrc.oss-cn-hangzhou.aliyuncs.com/T082-ZhengyangChen.pdf)
      - EER/minDCF reduction from 8.426%/0.487 to **6.492%/0.354** after large margin fine-tuning and AS-Norm
* [NIST SRE16](https://github.com/wenet-e2e/wespeaker/tree/master/examples/sre/v2): Speaker Verification recipe for the [2016 NIST Speaker Recognition Evaluation Plan](https://www.nist.gov/itl/iad/mig/speaker-recognition-evaluation-2016). Similar recipe can be found in [Kaldi](https://github.com/kaldi-asr/kaldi/tree/master/egs/sre16).
   * 🔥 UPDATE 2023.07.14: We support NIST SRE16 recipe. After PLDA adaptation, we achieved 6.608%, 10.01%, and 2.974% EER on trial Pooled, Tagalog, and Cantonese, respectively.
* [VoxConverse](https://github.com/wenet-e2e/wespeaker/tree/master/examples/voxconverse): Diarization recipe on the [VoxConverse dataset](https://www.robots.ox.ac.uk/~vgg/data/voxconverse/)

## Support List:

* Model (SOTA Models)
    - [x] [Standard X-vector](http://www.danielpovey.com/files/2017_interspeech_embeddings.pdf)
    - [x] [ResNet](https://arxiv.org/pdf/1512.03385.pdf)
    - [x] [ECAPA_TDNN](https://arxiv.org/pdf/2005.07143.pdf)
    - [x] [RepVGG](https://arxiv.org/pdf/2101.03697.pdf)
    - [x] [CAM++](https://arxiv.org/pdf/2303.00332.pdf)
* Pooling Functions
    - [x] TAP(mean) / TSDP(std) / TSTP(mean+std)
        - Comparison of mean/std pooling can be found in [shuai_iscslp](https://x-lance.sjtu.edu.cn/en/papers/2021/iscslp21_shuai_1_.pdf), [anna_arxiv](https://arxiv.org/pdf/2203.10300.pdf)
    - [x] Attentive Statistics Pooling (ASTP)
        - Mainly for ECAPA_TDNN
    - [x] Multi-Query and Multi-Head Attentive Statistics Pooling (MQMHASTP)
        - Details can be found in [MQMHASTP](https://arxiv.org/pdf/2110.05042.pdf)
* Criteria
    - [x] Softmax
    - [x] [Sphere (A-Softmax)](https://www.researchgate.net/publication/327389164)
    - [x] [Add_Margin (AM-Softmax)](https://arxiv.org/pdf/1801.05599.pdf)
    - [x] [Arc_Margin (AAM-Softmax)](https://arxiv.org/pdf/1801.07698v1.pdf)
    - [x] [Arc_Margin+Inter-topk+Sub-center](https://arxiv.org/pdf/2110.05042.pdf)
    - [x] [SphereFace2](https://ieeexplore.ieee.org/abstract/document/10094954)
* Scoring
    - [x] Cosine
    - [x] PLDA
    - [x] Score Normalization (AS-Norm)
* Metric
    - [x] EER
    - [x] minDCF
* Online Augmentation
    - [x] Noise && RIR
    - [x] Speed Perturb
    - [x] SpecAug
* Training Strategy
    - [x] Well-designed Learning Rate and Margin Schedulers
    - [x] Large Margin Fine-tuning
    - [x] Automatic Mixed Precision (AMP) Training
* Runtime
    - [x] Python Binding
    - [x] Triton Inference Server on verification && diarization in GPU deployment
    - [x] C++ Onnxruntime
* Self-Supervised Learning (SSL)
    - [x] [DINO](https://openaccess.thecvf.com/content/ICCV2021/papers/Caron_Emerging_Properties_in_Self-Supervised_Vision_Transformers_ICCV_2021_paper.pdf)
    - [x] [MoCo](https://openaccess.thecvf.com/content_CVPR_2020/papers/He_Momentum_Contrast_for_Unsupervised_Visual_Representation_Learning_CVPR_2020_paper.pdf)
    - [x] [SimCLR](http://proceedings.mlr.press/v119/chen20j/chen20j.pdf)
* Literature
    - [x] [Awesome Speaker Papers](docs/speaker_recognition_papers.md)

## Discussion

For Chinese users, you can scan the QR code on the left to follow our offical account of `WeNet Community`.
We also created a WeChat group for better discussion and quicker response. Please scan the QR code on the right to join the chat group.
| <img src="https://github.com/wenet-e2e/wenet-contributors/blob/main/wenet_official.jpeg" width="250px"> | <img src="https://github.com/wenet-e2e/wenet-contributors/blob/main/wespeaker/wangshuai.jpg" width="250px"> |
| ---- | ---- |

## Citations
If you find wespeaker useful, please cite it as
```bibtex
@article{wang2022wespeaker,
  title={Wespeaker: A Research and Production oriented Speaker Embedding Learning Toolkit},
  author={Wang, Hongji and Liang, Chengdong and Wang, Shuai and Chen, Zhengyang and Zhang, Binbin and Xiang, Xu and Deng, Yanlei and Qian, Yanmin},
  journal={arXiv preprint arXiv:2210.17016},
  year={2022}
}
```
## Looking for contributors

If you are interested to contribute, feel free to contact @wsstriving or @robin1001
