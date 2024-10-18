import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.backends.cudnn as cudnn

from config import get_config
from networks.vision_transformer import SwinUnet
from trainer import trainer_cardiac

parser = argparse.ArgumentParser()

parser.add_argument('--dataset', type=str,
                    default='mutil_datasets', help='experiment_name')

# parser.add_argument('--list_dir', type=str,
#                     default='./lists/lists_Flare', help='list dir')
parser.add_argument('--data_csv', type=str,
                    help='path to the dataset csv file'
                    )
parser.add_argument('--val_data_csv', type=str,
                    help='path to the validation dataset csv file'
                    )
# parser.add_argument('--num_classes', type=int,
#                     default=14, help='output channel of network')

parser.add_argument('--output_dir', type=str, help='output dir')

parser.add_argument('--max_iterations', type=int,
                    default=35000, help='maximum epoch number to train')

parser.add_argument('--max_epochs', type=int,
                    default=100, help='maximum epoch number to train')

parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')

parser.add_argument('--deterministic', type=int, default=1,
                    help='whether use deterministic training')

parser.add_argument('--base_lr', type=float, default=0.01,
                    help='segmentation network learning rate')

parser.add_argument('--img_size', type=int,
                    default=224, help='input patch size of network input')

parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')

parser.add_argument('--cfg', type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml", required=False,
                    metavar="FILE", help='path to config file', )

parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)

parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')

parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                         'full: cache all data, '
                         'part: sharding the dataset into nonoverlapping pieces and only cache one piece')

parser.add_argument('--resume', help='resume from checkpoint')

parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")

parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")

parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')

parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--num_workers', default=0, type=int)
parser.add_argument('--patience', default=10, type=int)
parser.add_argument('--batch_size', default=32, type=int)
parser.add_argument('--device', default="cuda:0")

parser.add_argument('--eval', action='store_true', help='Perform evaluation only')

parser.add_argument('--throughput', action='store_true', help='Test throughput only')

args = parser.parse_args()
config = get_config(args)

if __name__ == '__main__':
    # def count_parameters(model):
    #     return sum(p.numel() for p in model.parameters() if p.requires_grad)
    #
    # ### Testing pretrained model ###
    # net = SwinUnet(config, img_size=args.img_size, num_classes=[14]).cuda()
    # print(f"The model has {count_parameters(net):,} trainable parameters")
    # net.load_from(config)
    ### Testing pretrained model ###
    # import torch.multiprocessing as mp

    # if mp.get_start_method(allow_none=True) is None:
    #     mp.set_start_method('spawn', force=True)

    # if not args.deterministic:
    cudnn.benchmark = True
    cudnn.deterministic = False
    # else:
    #     cudnn.benchmark = False
    #     cudnn.deterministic = True
    CUDA_LAUNCH_BLOCKING = 1
    # random.seed(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # torch.cuda.manual_seed(args.seed)
    #
    args.data_csv = './lists/datasets_train.csv'
    args.val_data_csv = './lists/datasets_val.csv'

    args.device = "cpu" if not torch.cuda.is_available() else args.device
    # args.batch_size = 64
    args.output_dir = './exp_'
    # Create a list to store the number of classes for each dataset
    # num_classes_per_dataset = [dataset_config[name]['num_classes'] for name in dataset_config]
    num_classes_per_dataset = pd.read_csv(args.data_csv).groupby("predict_head").n_classes.first().values.tolist()
    # Create and initialize the model outside the loop
    net = SwinUnet(config, img_size=args.img_size, num_classes=num_classes_per_dataset).to(device=args.device)
    net.load_from(config)
    args.num_classes = num_classes = num_classes_per_dataset
    # if args.batch_size != 24 and args.batch_size % 6 == 0:
    #     args.base_lr *= args.batch_size / 24
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    print(f"Input args: {args}, config: {config}")
    trainer_cardiac(args, net, args.output_dir)
