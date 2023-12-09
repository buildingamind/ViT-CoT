
'''
SUPPORTED MODELS - 
1. Autoencoder
2. Variational Autoencoder
3. SimCLR
4. BYOL
5. Barlow Twins
6. CPC
7. Vision Transformer
8. Untrained ViT
9. Untrained ResNet-18
10. Untrained ResNet-34
11. Untrained ResNet-18 (3 blocks, 2 blocks, 1 block)
12. Individual Attention Heads
13. VideoMAE
'''

import sys
sys.path.append("/home/lpandey/ViT-CoT")
from argparse import ArgumentParser
from ast import arg
from platform import architecture
import ast
import pytorch_lightning as pl
import torch
import os
import torch.nn as nn
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' # tf warnings set to silent in terminal

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger
from torchvision.models import resnet18, resnet34
#from models.untrained_resnet import resnet18, resnet9 # separate file for untrained architecture
#from models.untrained_resnet2 import resnet_3_block # separate file for 3 block architecture
# from models.archs import resnets, resnet_3b
# from models.archs.resnet_3b import resnet_3blocks
# from models.archs.resnet_2b import resnet_2blocks
# from models.archs.resnet_1b import resnet_1block
import collections.abc as container_abcs
from datamodules.invariant_recognition import InvariantRecognitionDataModule
from models.evaluator import Evaluator
from models.simclr import SimCLR
# from train_ae import AE
# from train_byol import BYOL
# from train_vae import VAE
#from train_cpc import CPC_v2
# from train_barlowTwins import BarlowTwins
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import Callback
from pytorch_lightning.loggers import CSVLogger
#from train_transformer import Transformer, learner
import csv
import wandb
import collections
import pandas as pd
from models.vit_contrastive import VisionTransformer, Backbone, LitClassifier, ViTConfigExtended 
#from IndividualViTHeads import AttentionModel
#from models.sit_contrastive import VisionTransformer, ViT
#from VideoMAE_Saber_versionv2_2.run import get_config
import transformers
#from datamodules.invariant_recog_Nframes import InvariantRecognitionDataModule_Nframes

# wandb table to log results with UI
columns = ["IMAGE", "ACTUAL_LABEL", "PREDICTED_LABEL", "PROBABILITY", "CONFIDENCE", "LOSS", "PATH", "VIEWPOINT"]
log_table = wandb.Table(columns)


# pandas dataframe to log results for each fold as .csv file
dataFrame = pd.DataFrame(columns = ["ACTUAL_LABEL ", "PREDICTED_LABEL ", "PROBABILITY ", "CONFIDENCE ", "LOSS ", "PATH ", "VIEWPOINT "])


def cli_main():
    parser = create_argparser()
    args = parser.parse_args()
    count = -1
    
    LOG_DIR = f"/data/lpandey/LOGS/eval/csv/{args.project_name}.csv"

    write_csv_stats(LOG_DIR,
                    [{'MODEL':args.model, ' ARCHITECTURE':args.architecture, ' EXPERIMENT':args.exp_name, ' FOLD':args.identifier,
                    ' TEST_SET':args.data_dir}])

    
    # Run K fold cross-validation.
    for fold in range(args.num_folds):
        count+=1
        cross_validation(args, fold, count)
    
    # push result data to a .csv file
    #dataFrame.to_csv(f"/home/lpandey/LOGS/eval/dataframes/{args.project_name}", sep=',')


def cross_validation(args, fold_idx, count):

    LOG_DIR = f"/data/lpandey/LOGS/eval/csv/{args.project_name}.csv"
    
    if args.model == 'videomae':
        print("loading n_frames dataset for videomae")
        dm = InvariantRecognitionDataModule_Nframes(
            data_dir=args.data_dir,
            identifier=args.identifier,
            num_folds=args.num_folds,
            val_fold=fold_idx,
            batch_size=128,
            shuffle=False,
        )
    else:
        # Load data for this fold.
        dm = InvariantRecognitionDataModule(
            data_dir=args.data_dir,
            identifier=args.identifier,
            num_folds=args.num_folds,
            val_fold=fold_idx,
            batch_size=128,
            shuffle=True, # initial - True
        )
    
    print("dataloader loaded successfully")
    
    print("shuffle is - ", dm.shuffle)
    

    model = init_model(args)
    
    if args.model == 'vit' or args.model == "cpc" or args.model == "untrained_vit" or args.model == 'individual_vit_heads' or args.model == 'videomae':
        feature_dim = 512
    else:
        feature_dim = get_model_output_size(model, dm.dims)
    

    dm.setup()
    
    if args.model == 'videomae':
        is_videoMAE = True
    else:
        is_videoMAE = False
    evaluator = Evaluator(model, in_features=feature_dim, max_epochs=args.max_epochs, log_table=log_table, dataFrame=dataFrame, is_videoMAE=is_videoMAE)
    
    #print(evaluator)
    model_checkpoint = ModelCheckpoint(save_last=True, save_top_k=1, monitor='val_loss', filename='{epoch}-'+str(count))
    # Custom Checkpoint - ImagePredictionLogger: For prediction results
    callbacks = [model_checkpoint] # ,ImagePredictionLogger(val_samples)

    # create a new logger  for wandB
    logger = WandbLogger(save_dir=f"LOGS/eval/{args.model}", name=args.exp_name, project=f"{args.project_name}", log_model="all")
    trainer = pl.Trainer(
        gpus=1,
        logger=logger,
        max_epochs=args.max_epochs,
        callbacks=callbacks
    )

    #print(evaluator)
    trainer.fit(evaluator, datamodule=dm)
    #returns a list type
    metric_test = trainer.test(datamodule=dm)
    
    
    #save metrics (testa_acc, model_info) in .csv file
    write_csv_stats(LOG_DIR, metric_test)
    

# save metrics to .csv file -
def write_csv_stats(csv_path, metric_test):
    # creates a csv file automatically if none exists
    if not os.path.isfile(csv_path):
        with open(csv_path, 'w') as f: # w - write to file
            csv_writer = csv.writer(f)
            csv_writer.writerow(metric_test)

    with open(csv_path, 'a') as f: # a - append to file
        csv_writer = csv.writer(f)
        csv_writer.writerow(metric_test)



def init_model(args):
    if args.model == 'pixels':
        model = nn.Flatten()
    elif args.model == 'simclr':
        model = SimCLR.load_from_checkpoint(args.model_path)
    elif args.model == 'byol':
        model = BYOL.load_from_checkpoint(args.model_path)
        model = model.online_network.encoder
    
    elif args.model == 'barlowTwins':
        model = BarlowTwins.load_from_checkpoint(args.model_path)
        #model = model.encoder
    
    elif args.model == 'ae':
        model = AE.load_from_checkpoint(args.model_path).encoder
    elif args.model == 'vae':
        model = VAE.load_from_checkpoint(args.model_path).encoder
    elif args.model == 'supervised':
        model = resnet18(pretrained=True)
        model.fc = nn.Identity()
    elif args.model == 'untrained_r18':
        model = resnet18(pretrained=False)
        model.fc = nn.Identity()
    elif args.model == 'untrained_r34':
        model = resnet34(pretrained=False)
        model.fc = nn.Identity()
    elif args.model == 'untrained_r18_3b':
        model = resnet_3blocks(pretrained=False)
        model.fc = nn.Identity()
        print("Model selected - untrained resnet 18 : 3 blocks")
    elif args.model == 'untrained_r18_2b':
        model = resnet_2blocks(pretrained=False)
        model.fc = nn.Identity()
        print("Model selected - untrained resnet 18 : 2 blocks")
    elif args.model == 'untrained_r18_1b':
        model = resnet_1block(pretrained=False)
        model.fc = nn.Identity()
        print("Model selected - untrained resnet 18 : 1 block")

    elif args.model == 'vit':
        #model = transformer # initialize your model class
        #model.load_state_dict(torch.load(args.model_path))
        # above code for pytorch ckpt loading
        model = LitClassifier.load_from_checkpoint(args.model_path).backbone
        model.fc = nn.Identity()
    
    elif args.model == 'sit':
        model = ViT.load_from_checkpoint(args.model_path).transformer # same as .backbone except the name is different
        #model.fc = nn.identity()

    elif args.model == 'untrained_vit':
        configuration = ViTConfigExtended()
        configuration.image_size = args.image_size
        configuration.patch_size = args.patch_size
        configuration.num_hidden_layers = args.vit_hidden_layers
        configuration.num_attention_heads = args.vit_attention_heads
        # print configuration parameters of ViT
        print('image_size - ', configuration.image_size)
        print('patch_size - ', configuration.patch_size)
        print('num_classes - ', configuration.num_classes)
        print('hidden_size - ', configuration.hidden_size)
        print('intermediate_size - ', configuration.intermediate_size)
        print('num_hidden_layers - ', configuration.num_hidden_layers)
        print('num_attention_heads - ', configuration.num_attention_heads)
        
        # pass the configuration parameters to get backbone
        backbone = Backbone('vit', configuration)
        model = LitClassifier(backbone).backbone
        model.fc = nn.Identity()
    
    elif args.model == 'individual_vit_heads':
        # first load the backbone using checkpoint
        model = LitClassifier.load_from_checkpoint(args.model_path).backbone
        model.fc = nn.Identity()
        # load a single attention head - 
        model = AttentionModel(model=model, finetune=False, num_heads=args.num_heads, parent_head=args.parent_head, conv_finetune=args.conv_finetune)
        # print config
        print("number of heads: ", model.heads)
        print("parent head: ", model.parent_head)

    elif args.model == 'videomae':
        checkpoint = torch.load(args.model_path)
        config = transformers.VideoMAEConfig(image_size=args.image_size, patch_size=args.patch_size, num_channels=3,
                                             num_frames=args.num_frames, tubelet_size=args.tubelet_size, 
                                             hidden_size=768, num_hidden_layers=args.encoder_hidden_layers,
                                             num_attention_heads=args.encoder_attention_heads,
                                             intermediate_size=3072, decoder_num_attention_heads=args.decoder_heads,
                                             decoder_hidden_size=384, decoder_num_hidden_layers=args.decoder_layers)
        
        videomae = transformers.VideoMAEForPreTraining(config)
        videomae.load_state_dict(checkpoint['model_state_dict'])
        model = videomae.videomae # extract only the encoder part

        print("image size :: ", config.image_size)
        print("patch size :: ", config.patch_size)
        print("num of frames :: ", config.num_frames)
        print("tubelet size :: ", config.tubelet_size)
        print("encoder layers :: ", config.num_hidden_layers)
        print("encoder attention heads :: ", config.num_attention_heads)
        print("decoder attention heads :: ", config.decoder_num_attention_heads)
        print("decoder layers :: ", config.decoder_num_hidden_layers)

    elif args.model == 'cpc':
        model = CPC_v2.load_from_checkpoint(args.model_path)

    return model



def get_model_output_size(model, input_size) -> int:
    """ Returns the output activation size of the encoder. """
    with torch.no_grad():
        if isinstance(input_size, int):
            x = model(torch.zeros(1, input_size))
        else:
            x = model(torch.zeros(1, *input_size))
        return x.view(1, -1).size(1)



def create_argparser():
    parser = ArgumentParser()

    # create a separate Flag group for the VideoMAE model
    group_title = "VideoMAE Flags"
    group = parser.add_argument_group(group_title)

    parser.add_argument("--data_dir", type=str, help="directory containing dataset")
    parser.add_argument("--num_heads", type=int, help="total number of attention heads in the transformer backbone")
    parser.add_argument("--parent_head", type=int, help="choose a single head to get it's attention weight")
    parser.add_argument("--conv_finetune", type=bool, help="freeze/unfreeze final conv layers")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--model", type=str, choices=['pixels', 'supervised', 'simclr', 'untrained_r18', 'untrained_r34', 'untrained_r18_3b', 'untrained_r18_2b', 'untrained_r18_1b', 'untrained_vit', 'ae', 'byol', 'vae', 'barlowTwins', 'vit', 'sit', 'cpc', 'individual_vit_heads', 'videomae'])
    parser.add_argument("--model_path", type=str, help="stored model checkpoint")
    parser.add_argument("--max_epochs", default=100, type=int, help="Max number of epochs to train.")
    parser.add_argument("--num_folds", default=6, type=int, help="Number of CV folds.")
    parser.add_argument("--identifier", type=str, help="6fold, 6sparse, 12sparse, 12fold")
    parser.add_argument("--project_name", type=str, help="project_name") # for wandb dashboard and logging
    parser.add_argument("--shuffle", type=bool, default=True, help="shuffle images for training") # for wandb dashboard and logging
    parser.add_argument("--architecture", type=str, default=None, choices=['4block', '3block', '2block', '1block'], help="type of ResNet architecture")
    parser.add_argument('--image_size', default=64, type=int, help='image resolution that the model supports')
    parser.add_argument('--patch_size', default=8, type=int, help='size of image patches')
    parser.add_argument('--vit_attention_heads', default=3, type=int, help='num of attention heads in each transformer layer')
    parser.add_argument('--vit_hidden_layers', default=3, type=int, help='number of transformer layers')
    
    group.add_argument('--num_frames', default=16, type=int, help='number of frames in a single sequence used to train the backbone')
    group.add_argument('--tubelet_size', default=2, type=int,help='size of masking tube')
    group.add_argument('--encoder_hidden_layers', default=12, type=int, help='transformer layers in encoder')
    group.add_argument('--encoder_attention_heads', default=12, type=int, help='encoder attention heads in each encoder layer')
    group.add_argument('--decoder_heads', default=6, type=int, help='decoder attention heads in each decoder layer')
    group.add_argument('--decoder_layers', default=4, type=int, help='transformer layers in decoder')
    
    
    return parser


if __name__ == "__main__":
    cli_main()

    #if args.wandb_logging:
    # push table to wandb dashboard
    #wandb.log({"table":log_table})
    # finish wandb operation
    wandb.finish()
    
    print("ALL DONE")
