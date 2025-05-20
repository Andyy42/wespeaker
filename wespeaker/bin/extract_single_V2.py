#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# =============================================================================
# Description: Extracts single embedding from the wav file.
# Author: Ondřej Odehnal <xodehn09@vutbr.cz>
# =============================================================================
"""Extracts single embedding from the wav file."""
# =============================================================================
# Imports
# =============================================================================
import copy
import json
import io

import fire
import torch
import numpy as np

from scipy.special import softmax
from torch.utils.data import DataLoader
from torch.utils.data import IterableDataset
import torch.nn as nn
import torchaudio

from wespeaker.dataset.dataset_V2 import Dataset
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import parse_config_or_kwargs, validate_path
from wespeaker.models.projections import get_projection

from pathlib import Path



## Podoblasti (Subregions)
SUBREGIONS_CZ = {
    "1-1": "Severovýchodočeská",
    "1-2": "Středočeská",
    "1-3": "Jihozápadočeská",
    "1-4": "Českomoravská",
    "2-1": "Centrální středomoravská",
    "2-2": "Jižní středomoravská",
    "2-3": "Západní středomoravská",
    "2-4": "Východní středomoravská",
    "3-1": "Jižní východomoravská ",
    "3-2": "Severní východomoravská ",
    "3-3": "Kopaničářská",
    "4-1": "Slezskomoravská",
    "4-2": "Slezskopolská",
}

SUBREGIONS_EN = {
    "1-1": "Northeastern Bohemian",
    "1-2": "Central Bohemian",
    "1-3": "Southwestern Bohemian",
    "1-4": "Bohemian-Moravian",
    "2-1": "Central Central-Moravian",
    "2-2": "Southern Central-Moravian",
    "2-3": "Western Central-Moravian",
    "2-4": "Eastern Central-Moravian",
    "3-1": "Southern Eastern-Moravian (Moravian-Slovak)",
    "3-2": "Northern Eastern-Moravian (Walachian)",
    "3-3": "Kopanice Region",
    "4-1": "Silesian Moravian",
    "4-2": "Silesian Polish",
}

SUBREGIONS = SUBREGIONS_EN
CZECH_DIALECT_SUBCLASSES_NUMBER = len(SUBREGIONS_EN)

class SingleSampleDataset(IterableDataset):
    def __init__(self, sample):
        super(SingleSampleDataset, self).__init__()
        self.sample = sample

    def __iter__(self):
        yield self.sample

class SpeakerNet(nn.Module):
    def __init__(self, model, projecNet):
        super(SpeakerNet, self).__init__()
        self.speaker_extractor = model
        self.projection = projecNet

    def forward(self, x, y):
        x = self.speaker_extractor(x)
        x = self.projection(x, y)
        return x


# TODO: How to handle flags with fire lib?
def extract(
    input_wav: str,
    model_path: str,
    overwrite: bool = True,
    config: str = "conf/config.yaml",
    device_type: str = "cpu",
    **kwargs,
):
    """Extracts single embedding from the wav file.

    Args:
        input_wav (str): Input wav file.
        model_path (str): Model path.
        overwrite (bool, optional): Overwrite the output. Defaults to True.
        config (str, optional): Configuration for the model. Defaults to "conf/config.yaml".
        device_type (str, optional): Device type for torch, either "cpu" or "gpu". Defaults to "cpu".
    """

    assert Path(
        input_wav
    ).exists(), f"File wav_file {input_wav} does not exist!"
    assert Path(config).exists(), f"File config {config} does not exist!"
    assert device_type in ["cpu", "cuda"], f"Invalid device_type {device_type}!"

    # parse configs first and set the pre-defined device and data_type for embedding extraction
    configs = parse_config_or_kwargs(config, **kwargs)
    configs["model_args"]["device"] = device_type
    configs["data_type"] = "raw"

    configs["model_args"]["model_path"] = model_path

    # TODO: Consider using the utt_chunk parameter
    utt_chunk = configs.get(
        "utt_chunk", False
    )  # NOTE: This will be used to chunk the utterance into 40 seconds segments if set to True

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    model = get_speaker_model(configs["model"])(**configs["model_args"])
    ############################################################
    # projection layer
    ############################################################
    configs["projection_args"]["embed_dim"] = configs["model_args"]["embed_dim"]
    configs["projection_args"]["num_class"] = CZECH_DIALECT_SUBCLASSES_NUMBER
    configs["projection_args"]["do_lm"] = configs.get("do_lm", False)
    if configs["data_type"] != "feat" and configs["dataset_args"]["speed_perturb"]:
        # diff speed is regarded as diff spk
        configs["projection_args"]["num_class"] *= 3
    projection = get_projection(configs["projection_args"])

    # Load model with projection layer
    model = SpeakerNet(
        get_speaker_model(configs["model"])(**configs["model_args"]),
        projection,
    )
    ############################################################

    load_checkpoint(model, model_path)
    device = torch.device(device_type)
    model.to(device).eval()

    # test_configs
    test_conf = copy.deepcopy(configs["dataset_args"])
    test_conf["speed_perturb"] = False
    if "fbank_args" in test_conf:
        test_conf["fbank_args"]["dither"] = 0.0
    elif "mfcc_args" in test_conf:
        test_conf["mfcc_args"]["dither"] = 0.0
    test_conf["spec_aug"] = False
    test_conf["shuffle"] = False
    test_conf["aug_prob"] = 0.0 # configs.get("aug_prob", 0.0)
    test_conf["filter"] = False

    # Utt chunk
    # test_conf["utt_chunk"] = utt_chunk
    # print("WARN: Setting utt_chunk =", utt_chunk)

    if not isinstance(input_wav, io.BytesIO):
        waveform, sample_rate = torchaudio.load(input_wav)
    else:
        with open(input_wav, 'rb') as f:
            input_wav = f.read()
        waveform, sample_rate = torchaudio.load(io.BytesIO(input_wav))

    waveform = waveform.squeeze(0)
    waveform = waveform * (1 << 15)
 

    # NOTE: We are processing single utterances but we are using the Dataset to handle the processing
    file_sample = dict(key=input_wav, label=-1, feat=waveform,)
    dataloader = DataLoader(
        SingleSampleDataset(file_sample),
        shuffle=False,
        batch_size=configs.get("batch_size", 1),
        num_workers=0, # 0 is needed to let the Dataloder to run as single process (Celery compatibility)
        prefetch_factor=None,
    )

    # Process the utterance
    with torch.no_grad():

        # NOTE: we have only one utterance but we still use the dataloader to handle the processing
        dataloader_iterator = iter(enumerate(dataloader))
        i, batch = next(dataloader_iterator)

        utts = batch["key"]
        print(f"[{i}] Proccesing utts: {utts[0]}", flush=True)

        features = batch["feat"]
        features = features.float().to(device)  # (B,T,F)
        # Forward through model
        dummy_target = torch.zeros(features.size(0), dtype=torch.float)
        outputs = model(
            features, dummy_target.float().to(device)
        )  # embed or (embed_a, embed_b)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        embeds = embeds.cpu().detach().numpy()  # (B,F)

        # NOTE: We take only the last embedding. We are processing only one utterance.
        embed = embeds[i]

    subregion_key, subregion_name = list(SUBREGIONS.items())[int(embed.argmax())]
    output={}
    output["best"]={}
    output["best"]["subregion_key"]=subregion_key
    output["best"]["subregion_name"]=subregion_name
    output["best"]["score"]=float(embed.max())
    from scipy.special import softmax
    output["hypothesis"]=list(embed.tolist())
    output["percentage"]=softmax(list(embed.tolist()))

    print(
        f"Predicted subregion {subregion_key} : {subregion_name} with probability {softmax(embed).max()*100:.2f} %"
    )
    return(output)



# NOTE: Uses io.BytesIO 
def extract_1file(
#    input_wav_file: str,    
    input_wav: io.BytesIO|str,
    config: str = "conf/config.yaml",
    **kwargs,
):
    """Extracts single embedding from ioBytesIO object.

    Args:
        input_wav_file (io.BytesIO): Input wav.
        model_path (str): Model path.
        config (str, optional): Configuration for the model. Defaults to "conf/config.yaml".
        device_type (str, optional): Device type for torch, either "cpu" or "gpu". Defaults to "cpu".
    """
    
    # parse configs first and set the pre-defined device and data_type for embedding extraction
    configs = parse_config_or_kwargs(config, **kwargs)
    
    # TODO: Consider using the utt_chunk parameter
    #utt_chunk = configs.get(
    #    "utt_chunk", False
    #)  # NOTE: This will be used to chunk the utterance into 40 seconds segments if set to True

    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    model = get_speaker_model(configs["model"])(**configs["model_args"])

    ############################################################
    # projection layer
    ############################################################
    configs["projection_args"]["embed_dim"] = configs["model_args"]["embed_dim"]
    configs["projection_args"]["num_class"] = CZECH_DIALECT_SUBCLASSES_NUMBER
    configs["projection_args"]["do_lm"] = configs.get("do_lm", False)
    projection = get_projection(configs["projection_args"])

    # Load model with projection layer
    model = SpeakerNet(
        get_speaker_model(configs["model"])(**configs["model_args"]),
        projection,
    )
    ############################################################

    load_checkpoint(model, configs["model_args"]["model_path"])
    device = torch.device(configs["model_args"]["device"])
    model.to(device).eval()

    if not isinstance(input_wav, io.BytesIO):
        waveform, sample_rate = torchaudio.load(input_wav)
    else:
        with open(input_wav, 'rb') as f:
            input_wav = f.read()
        waveform, sample_rate = torchaudio.load(io.BytesIO(input_wav))

    waveform = waveform.squeeze(0)
    waveform = waveform * (1 << 15)
    
    #file_sample = dict(key="file", wav=waveform, spk="-", label=-1, sample_rate=sample_rate)
    file_sample = dict(key="file", label=-1, feat=waveform,)
    dataloader = DataLoader(
        SingleSampleDataset(file_sample),
        shuffle=False,
        batch_size=configs.get("batch_size", 1),
        num_workers=0, # 0 is needed to let the Dataloder to run as single process (Celery compatibility)
        prefetch_factor=None,
    )

    # Process the utterance
    with torch.no_grad():

        # NOTE: we have only one utterance but we still use the dataloader to handle the processing
        dataloader_iterator = iter(enumerate(dataloader))
        i, batch = next(dataloader_iterator)

        utts = batch["key"]
        print(f"[{i}] Proccesing utts: {utts[0]}", flush=True)

        features = batch["feat"]
        features = features.float().to(device)  # (B,T,F)
        # Forward through model
        dummy_target = torch.zeros(features.size(0), dtype=torch.float)
        outputs = model(
            features, dummy_target.float().to(device)
        )  # embed or (embed_a, embed_b)
        embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
        embeds = embeds.cpu().detach().numpy()  # (B,F)

        # NOTE: We take only the last embedding. We are processing only one utterance.
        embed = embeds[i]

    subregion_key, subregion_name = list(SUBREGIONS.items())[int(embed.argmax())]
    output={}
    output["best"]={}
    output["best"]["subregion_key"]=subregion_key
    output["best"]["subregion_name"]=subregion_name
    output["best"]["score"]=float(embed.max())
    output["hypothesis"]=list(embed.tolist())


    print(
        f"Predicted subregion {subregion_key} : {subregion_name} with probability {softmax(embed).max()*100:.2f} %"
    )
    print(output)
    return(output)


if __name__ == "__main__":
    fire.Fire(extract)
