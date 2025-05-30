# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import re
from pprint import pformat

import wandb
import fire
import tableprint as tp
import torch
import torch.distributed as dist
import yaml
from torch.utils.data import DataLoader
import torch.nn as nn
import wespeaker.utils.schedulers as schedulers
from wespeaker.dataset.dataset_V2 import Dataset
from wespeaker.models.projections import get_projection
from wespeaker.models.speaker_model import get_speaker_model
from wespeaker.utils.checkpoint import load_checkpoint, save_checkpoint
from wespeaker.utils.executor_V2 import run_epoch
from wespeaker.utils.file_utils import read_table
from wespeaker.utils.utils import get_logger, parse_config_or_kwargs, set_seed, \
    spk2id

class SpeakerNet(nn.Module):
    def __init__(self, model, projecNet):
        super(SpeakerNet, self).__init__()
        self.speaker_extractor = model
        self.projection = projecNet
    def forward(self, x, y):
        x = self.speaker_extractor(x)
        x = self.projection(x, y)
        return x


def train(config='conf/config.yaml', **kwargs):
    """Trains a model on the given features and spk labels.

    :config: A training configuration. Note that all parameters in the
             config can also be manually adjusted with --ARG VALUE
    :returns: None
    """
    configs = parse_config_or_kwargs(config, **kwargs)
    checkpoint = configs.get('checkpoint', None)
    # dist configs
    os.environ['MASTER_ADDR']='localhost'
    os.environ['MASTER_PORT']=str(configs['PORT'])
    rank = int(os.environ['RANK'])
    world_size = int(os.environ['WORLD_SIZE'])
    gpu = int(configs['gpus'][rank])
    torch.cuda.set_device(gpu)
    dist.init_process_group(backend='nccl')

    model_dir = os.path.join(configs['exp_dir'], "models")
    if rank == 0:
        try:
            os.makedirs(model_dir)
        except IOError:
            print(model_dir + " already exists !!!")
            if checkpoint is None:
                exit(1)
    dist.barrier(device_ids=[gpu])  # let the rank 0 mkdir first

    logger = get_logger(configs['exp_dir'], 'train.log')
    if world_size > 1:
        logger.info('training on multiple gpus, this gpu {}'.format(gpu))

    if rank == 0:
        logger.info("exp_dir is: {}".format(configs['exp_dir']))
        logger.info("<== Passed Arguments ==>")
        # Print arguments into logs
        for line in pformat(configs).split('\n'):
            logger.info(line)

    # seed
    set_seed(configs['seed'] + rank)
    # set_seed(configs['seed'])

    # train data
    train_label = configs['train_label']
    train_utt_spk_list = read_table(train_label)
    spk2id_dict = spk2id(train_utt_spk_list)
    if rank == 0:
        logger.info("<== Data statistics ==>")
        logger.info("train data num: {}, spk num: {}".format(
            len(train_utt_spk_list), len(spk2id_dict)))

    # dataset and dataloader
    train_dataset = Dataset(configs['data_type'],
                            configs['train_data'],
                            configs['dataset_args'],
                            spk2id_dict,
                            reverb_lmdb_file=configs.get('reverb_data', None),
                            noise_lmdb_file=configs.get('noise_data', None))
    train_dataloader = DataLoader(train_dataset, **configs['dataloader_args'])
    batch_size = configs['dataloader_args']['batch_size']
    if configs['dataset_args'].get('sample_num_per_epoch', 0) > 0:
        sample_num_per_epoch = configs['dataset_args']['sample_num_per_epoch']
    else:
        sample_num_per_epoch = len(train_utt_spk_list)
    epoch_iter = sample_num_per_epoch // world_size // batch_size
    if rank == 0:
        logger.info("<== Dataloaders ==>")
        logger.info("train dataloaders created")
        logger.info('epoch iteration number: {}'.format(epoch_iter))

    # model
    logger.info("<== Model ==>")
    model = get_speaker_model(configs['model'])(**configs['model_args'])
    num_params = sum(param.numel() for param in model.parameters())
    if rank == 0:
        logger.info('speaker_model size: {}'.format(num_params))
    if configs['model_init'] is not None:
        logger.info('Load initial model from {}'.format(configs['model_init']))
        load_checkpoint(model, configs['model_init'])
    elif checkpoint is None:
        logger.info('Train model from scratch ...')
    # projection layer
    configs['projection_args']['embed_dim'] = configs['model_args']['embed_dim']
    configs['projection_args']['num_class'] = len(spk2id_dict)
    configs['projection_args']['do_lm'] = configs.get('do_lm', False)
    if configs['data_type'] != 'feat' and configs['dataset_args'][
            'speed_perturb']:
        # diff speed is regarded as diff spk
        configs['projection_args']['num_class'] *= 3
        if configs.get('do_lm', False):
            logger.info('No speed perturb while doing large margin fine-tuning')
            configs['dataset_args']['speed_perturb'] = False
    projection = get_projection(configs['projection_args'])
    # model.add_module("projection", projection)
    model = SpeakerNet(model,projection)
    if rank == 0:
        # print model
        for line in pformat(model).split('\n'):
            logger.info(line)
        # !!!IMPORTANT!!!
        # Try to export the model by script, if fails, we should refine
        # the code to satisfy the script export requirements
        # script_model = torch.jit.script(model)
        # script_model.save(os.path.join(model_dir, 'init.zip'))

    # If specify checkpoint, load some info from checkpoint.
    if checkpoint is not None:
        load_checkpoint(model, checkpoint)
        start_epoch = int(re.findall(r"(?<=model_)\d*(?=.pt)",
                                     checkpoint)[0]) + 1
        logger.info('Load checkpoint: {}'.format(checkpoint))
    else:
        start_epoch = 1
    logger.info('start_epoch: {}'.format(start_epoch))

    named_params = {k:v for k,v in model.named_parameters()}
    
    weights_k = named_params["speaker_extractor.back_end.weights_k"]
    weights_v = named_params["speaker_extractor.back_end.weights_v"]
    print("[INFO] Printing weights for MHFA k,v")
    print(f"{weights_k=}")
    print(f"{weights_v=}")
    print(42*"=")

    # logger.info("<== Compiling model ==>")
    # import time
    # start_time = time.time()
    # model = torch.compile(model, mode="default")
    # seconds = time.time() - start_time
    # msg = f'Time taken to compile the model {time.strftime("%H:%M:%S",time.gmtime(seconds))}'
    # logger.info(msg)


    # ddp_model
    model.cuda()
    # ddp_model = torch.nn.parallel.DistributedDataParallel(model)
    ddp_model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True)
    device = torch.device("cuda")
    total_learnable = 0
    SpkEnc_learnable = 0

    logger.info("<== Adapter-Tuning ==>")
    logger.info(configs.get('adapter_tuning', False))
    if configs.get('adapter_tuning', False):
        # if configs['model_args']['adapter_type'] is not None:
        #     logger.info("Adapter Type is: " + configs['model_args']['adapter_type'])
        for name, param in ddp_model.named_parameters():
            if "adapter" in name:
                param.requires_grad = True
                total_learnable += param.numel()
            elif "layer_norm" in name:
                param.requires_grad = True
                total_learnable += param.numel()
            elif "back_end" in name:
                param.requires_grad = True
                SpkEnc_learnable += param.numel()
            elif "projection" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        logger.info("Adapter-Tuning Trainable Param in SSL model is: " + str(total_learnable))
        logger.info("Adapter-Tuning Trainable Param in Back_end model is: " + str(SpkEnc_learnable))


    # Convert weight to tensor
    if 'weight' in configs['loss_args']:
        if isinstance(configs['loss_args']['weight'], list):
            weight_orig = configs['loss_args']['weight']
            configs['loss_args']['weight'] = torch.tensor(configs['loss_args']['weight'],dtype=torch.float32).cuda()

    criterion = getattr(torch.nn, configs['loss'])(**configs['loss_args'])

    # Put back the original value (due to data types in YAML file..)
    if 'weight' in configs['loss_args']:
        if isinstance(configs['loss_args']['weight'], list):
            configs['loss_args']['weight'] = weight_orig


    if rank == 0:
        logger.info("<== Loss ==>")
        logger.info("loss criterion is: " + configs['loss'])

    configs['optimizer_args']['lr'] = configs['scheduler_args']['initial_lr']
    optimizer = getattr(torch.optim,
                        configs['optimizer'])(ddp_model.parameters(),
                                              **configs['optimizer_args'])
    if rank == 0:
        logger.info("<== Optimizer ==>")
        logger.info("optimizer is: " + configs['optimizer'])

    # scheduler
    configs['scheduler_args']['num_epochs'] = configs['num_epochs']
    configs['scheduler_args']['epoch_iter'] = epoch_iter
    # here, we consider the batch_size 64 as the base, the learning rate will be
    # adjusted according to the batchsize and world_size used in different setup
    configs['scheduler_args']['scale_ratio'] = 1.0 * world_size * configs[
        'dataloader_args']['batch_size'] / 64
    scheduler = getattr(schedulers,
                        configs['scheduler'])(optimizer,
                                              **configs['scheduler_args'])
    if rank == 0:
        logger.info("<== Scheduler ==>")
        logger.info("scheduler is: " + configs['scheduler'])

    # margin scheduler
    configs['margin_update']['epoch_iter'] = epoch_iter
    margin_scheduler = getattr(schedulers, configs['margin_scheduler'])(
        model=model, **configs['margin_update'])
    if rank == 0:
        logger.info("<== MarginScheduler ==>")

    # save config.yaml
    if rank == 0:
        saved_config_path = os.path.join(configs['exp_dir'], 'config.yaml')
        with open(saved_config_path, 'w') as fout:
            data = yaml.dump(configs)
            fout.write(data)


    if rank == 0:
        # WanDB setup
        wandb.init(
            project='naki-waspeaker',
            config=configs,
            name=configs['exp_dir'],
            resume=True,
        )

    # training
    dist.barrier(device_ids=[gpu])  # synchronize here
    if rank == 0:
        logger.info("<========== Training process ==========>")
        header = ['Epoch', 'Batch', 'Lr', 'Margin', 'Loss', "Acc"]
        for line in tp.header(header, width=10, style='grid').split('\n'):
            logger.info(line)
        # Optional: track gradients
        wandb.watch(model)

    dist.barrier(device_ids=[gpu])  # synchronize here

    scaler = torch.cuda.amp.GradScaler(enabled=configs['enable_amp'])
    for epoch in range(start_epoch, configs['num_epochs'] + 1):
        train_dataset.set_epoch(epoch)
        run_epoch(train_dataloader,
                epoch_iter,
                ddp_model,
                criterion,
                optimizer,
                scheduler,
                margin_scheduler,
                epoch,
                logger,
                scaler,
                enable_amp=configs['enable_amp'],
                log_batch_interval=configs['log_batch_interval'],
                device=device)

        if rank == 0:
            if epoch % configs['save_epoch_interval'] == 0 or epoch >= configs[
                    'num_epochs'] - configs['num_avg']:
                save_checkpoint(
                    model, os.path.join(model_dir,
                                        'model_{}.pt'.format(epoch)))

    if rank == 0:
        os.symlink('model_{}.pt'.format(configs['num_epochs']),
                   os.path.join(model_dir, 'final_model.pt'))
        logger.info(tp.bottom(len(header), width=10, style='grid'))


if __name__ == '__main__':
    fire.Fire(train)
