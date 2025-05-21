# WeSpeaker VoxLingua107 and NAKI Example (v2)

This directory contains examples and scripts for running Language Identification (LID) on VoxLingua107 dataset and Dialect Identification (DID) on NAKI dataset. The implementation is based on WeSpeaker, which has been adapted to handle classification tasks.

## Overview

The VoxLingua107 dataset contains speech samples from 107 languages. The NAKI dataset consists of Czech dialect samples. This example provides scripts for training and evaluating models for:
- Language Identification (LID) using VoxLingua107 dataset
- Dialect Identification (DID) using NAKI dataset

## Directory Structure

```
v2/
├── conf/                  # Configuration files for models
├── evaluate.sh            # Main evaluation script
├── job.sh                 # Job submission script for training
├── job_evaluate.sh        # Job submission script for evaluation
├── job_extract.sh         # Job submission script for feature extraction
├── job_dev.sh             # Job script for development experiments
├── local/                 # Local scripts for data preparation and scoring
│   ├── download_data.sh   # Script to download data
│   ├── extract_naki.sh    # Extracts features from NAKI data
│   ├── extract_vox.sh     # Extracts features from VoxLingua data
│   ├── m4a2wav.pl         # Converts .m4a files to .wav format
│   ├── prepare_data.sh    # Prepares data for training
│   ├── prepare_voxlingua107_dev.sh  # Prepares VoxLingua107 dev dataset
│   ├── score.sh           # Score evaluation results
│   ├── score_norm.sh      # Score normalization
│   └── score_plda.sh      # PLDA scoring
├── path.sh                # Environment setup script
├── README.md              # This file
├── run.sh                 # Main training script for ResNet models
├── run_evaluate.sh        # Script to evaluate trained models
├── run_WavLM.sh           # Training script using WavLM models
├── run_WavLM_generic.sh   # Generic version of WavLM training script
├── run_WavLM_naki.sh      # Script for training WavLM on NAKI data
├── scripts/               # Additional scripts for specific configurations
│   ├── naki/              # NAKI-specific scripts
│   ├── voxlingua107/      # VoxLingua107-specific scripts
│   └── voxlingua107_whisper/  # Scripts for Whisper-based experiments
└── tools/                 # Utility scripts
```

## Available Models

* ResNet: Standard ResNet models (ResNet18, ResNet34, etc.)
* WavLM: Pre-trained speech models adapted for classification

## Configuration and Setup

* Acoustic features: 80-dimensional filterbank features (fbank80)
* Frame length: 200 frames per utterance (num_frms200)
* Training: 150 epochs with ArcMargin loss function
* Augmentation: 0.6 probability (aug_prob0.6), speed perturbation enabled
* Scoring: Cosine similarity with mean subtraction based on VoxCeleb2 development set

## Main Scripts Description

### Training Scripts

- `run.sh`: Main training script for ResNet models on VoxLingua107
- `run_WavLM.sh`: Training script using WavLM models for LID tasks
- `run_WavLM_naki.sh`: Script specifically adapted for training on NAKI data (DID)
- `run_WavLM_generic.sh`: Generic version of the WavLM training script with more customization options
- `scripts/{naki,naki_resnet,voxlingua107}`: Training scripts with run and cluster job scripts for specific dataset

### Evaluation Scripts

- `evaluate.sh`: Main script for evaluating trained models
- `run_evaluate.sh`: Extended evaluation script with additional options
- `job_evaluate.sh`: Batch job script for running evaluations on computing clusters

### Data Preparation

- `local/prepare_data.sh`: Prepares data for training by organizing audio files and generating necessary metadata
- `local/prepare_voxlingua107_dev.sh`: Specifically prepares the VoxLingua107 dev dataset
- `local/m4a2wav.pl`: Converts .m4a audio files to .wav format (required for some datasets)

### Feature Extraction and Scoring

- `local/extract_naki.sh`: Extracts embeddings from NAKI dataset
- `local/extract_vox.sh`: Extracts embeddings from VoxLingua dataset
- `local/score.sh`: Calculates scores based on cosine similarity
- `local/score_norm.sh`: Performs score normalization (AS-Norm/S-Norm)
- `local/score_plda.sh`: PLDA-based scoring (alternative to cosine similarity)

## How to Run

### Training a Model

To train a ResNet model on VoxLingua107:
```bash
./run.sh
```

To train a WavLM model on NAKI data:
```bash
./run_WavLM_naki.sh
```

### Evaluating a Model

To evaluate a trained model:
```bash
./evaluate.sh --exp_dir exp/your_model_dir --model_path exp/your_model_dir/models/avg_model.pt
```

### Running on a Cluster

Use the job submission scripts:
```bash
sbatch job.sh       # For training
sbatch job_evaluate.sh  # For evaluation
```

## Model Architecture Options

The configuration files in the `conf/` directory provide various model architecture options:

- Different embedding sizes (192, 256, 512 dimensions)
- Multiple pooling strategies:
  - ASTP (Attentive Statistics Pooling)
  - MHFA (Multi-Head Factorized Attention)
  - LWAP (Locally Weighted Average Pooling)
- Loss functions:
  - Softmax
  - ArcMargin (AAM-Softmax)
  - No-margin variants

## Adaptation to Classification Tasks

Unlike traditional WeSpeaker which focuses on speaker verification tasks, this project has been adapted for classification:
- Modified loss functions for multi-class classification
- Adaptation of embedding extraction for language/dialect identification
- Evaluation metrics specific to classification tasks

