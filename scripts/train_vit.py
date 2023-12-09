# POSSIBLE ERRORS - NONE TYPE OBJECT HAS NO LEN ERROR - THIS MEANS THAT TEMPORAL_MODE FLAG IS NOT PROPERLY PASSED AND IS SET TO NONE SOMEPLACE.

import sys
sys.path.append("/home/lpandey/ViT-CoT/")

# LIBRARIES
from argparse import ArgumentParser
import wandb
import os
from pytorch_lightning.callbacks import ModelCheckpoint
#from pytorch_lightning.metrics import Accuracy
# Pytorch modules
import torch
import torch.nn
import torchvision.models as models
from torch.nn import functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader, random_split

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule, LightningModule, Trainer
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import Accuracy
from vit_pytorch import ViT
from transformers import ViTConfig

from torch.utils.data import DataLoader, random_split

# Pytorch-Lightning
from pytorch_lightning import LightningDataModule
from torchvision import transforms
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from datamodules.image_pairs import ImagePairsDataModule

# Extras
from pl_bolts.models.self_supervised.simclr.transforms import (
    SimCLREvalDataTransform, SimCLRTrainDataTransform)

# model
from models.vit_contrastive import Backbone, ViTConfigExtended, ViTConfig, configuration, LitClassifier
from models.simclr import SimCLR

def create_argparser():
    parser = ArgumentParser()
    parser.add_argument(
        "--max_epochs",
        default=100,
        type=int,
        help="Max number of epochs to train."
    )
    parser.add_argument(
        "--val_split",
        default=0.1,
        type=float,
        help="Percent (float) of samples to use for the validation split."
    )
    
    #The action set to store_true will store the argument as True , if present
    parser.add_argument(
        "--temporal",
        action="store_true",
        help="Use temporally ordered image pairs."
    )

    parser.add_argument(
        "--window_size",
        #default=2,
        type=int,
        help="Size of sliding window for sampling temporally ordered image pairs."
    )

    parser.add_argument(
        "--exp_name",
        type=str,
        help="Experiment name"
    )
    parser.add_argument(
        "--dataset_size",
        default=0,
        type=int,
        help="Subset of dataset"
    )
    parser.add_argument(
        "--project_name",
        type=str,
        help="wandb dashboard project name"
    )
    parser.add_argument(
        "--seed_val",
        type=int,
        default=0,
        help="SEED VALUE"
    )
    parser.add_argument(
        "--shuffle",
        type=bool,
        default=False,
        help="shuffle temporal images for training"
    )
    parser.add_argument(
        "--temporal_mode",
        type=str,
        choices=['2images','2+images'],
        default='2+images',
        help="select number of images to push together in a temporal window"
    )
    parser.add_argument(
        "--head",
        type=int,
        choices=[1,3,6,9,12,4],
        default=1,
        help="number of attention heads"
    )
    parser.add_argument(
        "--drop_ep",
        type=int,
        default=0,
        help="how many episodes to drop from the dataset. If there are is only one subfolder in the dataset, then set this to 0"
    )
    
    return parser


def cli_main():

    parser = create_argparser()

    # model args
    parser = SimCLR.add_model_specific_args(parser)
    args = parser.parse_args()
    args.gpus = 1
    args.lars_wrapper = True


    # assign heads and hidden layers using argparse
    # currently, heads and hidden_layers are same for stability.
    configuration.num_attention_heads = args.head
    configuration.num_hidden_layers = args.head

    print("Number of ATTENTION HEADS: ", configuration.num_attention_heads)
    print("Number of HIDDEN LAYERS: ", configuration.num_hidden_layers)
    

    # setup model and trainer 
    backbone = Backbone('vit', configuration)
    print(args.temporal_mode)
    model = LitClassifier(backbone=backbone, temporal_mode=args.temporal_mode)

    if args.temporal:
        dm = ImagePairsDataModule(
            data_dir=args.data_dir,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False, # shuffle is decided from flag
            drop_last=False, # # changed from True to False becz of empty dataloader error
            val_split=args.val_split,
            window_size=args.window_size,
            temporal_mode=args.temporal_mode,
            drop_ep=args.drop_ep,
        )

       

        """
        IMPORTANT:
            gaussian_blur is commented below and in arguments of simclr.py file,
            jitter_strength is commented below and in arguments of simclr.py file
            
            comment them out for non-temporal simclr models else result will
            differ from original results
        """

    # HARDCODED VALUES INSIDE!!!!!
    dm.train_transforms = SimCLRTrainDataTransform(
        input_height=64 #dm.size()[-1],
        #gaussian_blur=args.gaussian_blur,
        #jitter_strength=args.jitter_strength,
    )
    # HARDCODED VALUES INSIDE!!!!!
    dm.val_transforms = SimCLREvalDataTransform(
        input_height=64 #dm.size()[-1],
        #gaussian_blur=args.gaussian_blur,
        #jitter_strength=args.jitter_strength,
    )

    # The SimCLR data transforms are designed to be used with datamodules
    # which return a single image. But ImagePairsDataModule returns
    # a pair of images.
    if isinstance(dm, ImagePairsDataModule):
        dm.train_transforms = dm.train_transforms.train_transform
        dm.val_transforms = dm.val_transforms.train_transform


    args.num_samples = dm.num_samples
    # hard coded to check temporal working
    #args.num_samples = 9000
    #model = SimCLR(**args.__dict__)

    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss')
    callbacks = [model_checkpoint]

    logger = TensorBoardLogger("/data/lpandey/LOGS/VIT_Time", name=f"{args.exp_name}")
   
    trainer = pl.Trainer(
        gpus=args.gpus,
        max_epochs=args.max_epochs,
        logger=logger,
        sync_batchnorm=True if args.gpus > 1 else False,
        callbacks=callbacks,
    )

    
    #print(model)
    trainer.fit(model, datamodule=dm)




if __name__ == '__main__':
    cli_main()