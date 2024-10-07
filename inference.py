from PIL import Image
import argparse
import logging
import os
import random
import sys
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataset_synapse import CardiacDataset
from trainer import custom_collate_fn
from utils import test_single_volume
from networks.vision_transformer import SwinUnet as ViT_seg, SwinUnet
from config import get_config

parser = argparse.ArgumentParser()

parser.add_argument("--input_img", type=str)
parser.add_argument("--model_path", default="exp_/best_model.pth", type=str)
parser.add_argument('--cfg', type=str, default="configs/dataset.yaml", metavar="FILE", help='path to config file', )
parser.add_argument('--dataset', type=str, default='amos', help='experiment_name')
parser.add_argument('--data_csv', type=str, default='./lists/datasets_test.csv', help='path to the dataset csv file')
parser.add_argument('--output_dir', type=str, help='output dir')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--max_epochs', type=int, default=100, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--img_size', type=int, default=224, help='input patch size of network input')
parser.add_argument('--is_savenii', action="store_true", help='whether to save results during inference')
parser.add_argument('--test_save_dir', type=str, default='../predictions', help='saving prediction as nii!')
parser.add_argument('--deterministic', type=int, default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float, default=0.01, help='segmentation network learning rate')
parser.add_argument('--seed', type=int, default=1234, help='random seed')
# parser.add_argument('--cfg', type=str, required=True, metavar="FILE", help='path to config file', )
parser.add_argument("--opts", help="Modify config options by adding 'KEY VALUE' pairs. ", default=None, nargs='+', )
parser.add_argument('--zip', action='store_true', help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'], help='no: no cache, '
                                                                                                   'full: cache all data, '
                                                                                                   'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int, help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true', help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true', help='Test throughput only')
parser.add_argument('--device', default='cuda:0')
parser.add_argument('--num_workers', default=0, type=int)

args = parser.parse_args()
config = get_config(args)

args.device = "cpu" if not torch.cuda.is_available() else args.device


def inference(model):
    model.eval()
    # h, w = 224, 224
    # metric_list = 0.0
    sample = {
        "img_dir": "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/Swin-MAE-datasets/images/mri_mm/001_SA-2_0005.jpg",
        'n_classes': 3 + 1,
        'predict_head': 0,
        "label_dir": None
        }
    dataset = CardiacDataset(
        csv_file_path=sample,  # Assuming there is a csv file for training data
        # transform=transforms.Compose(transforms_list),
        # modes='train'
    )
    testloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    # sample =
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, predict_head = sampled_batch["image"], sampled_batch.get("label"), sampled_batch['predict_head']
        image = image.to(args.device)
        outputs = model(image, predict_head)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        out = (out / out.max() * 255).astype(np.uint8)
        np.savez("output.npz", out)
        img = Image.fromarray(out)
        img.save("output.jpg")


if __name__ == "__main__":
    net = SwinUnet(config, img_size=args.img_size, num_classes=[4, 4])
    net.load_state_dict(torch.load(args.model_path))
    net.to(device=args.device)
    inference(net)
