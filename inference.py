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
# parser.add_argument("--seed", default=1234, type=int)
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
    sample = {"img_dir": "/home/aicvi/projects/Swin-MAE-datasets/images/ct_coronary/12069336_0266.jpg",
              'n_classes': 3 + 1,
              'predict_head': 1,
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
        # h, w = sampled_batch["image"].size()[2:]

        image, label, predict_head = sampled_batch["image"], sampled_batch.get("label"), sampled_batch['predict_head']
        image = image.to(args.device)
        outputs = model(image, predict_head)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        out = (out / out.max() * 255).astype(np.uint8)
        img = Image.fromarray(out)
        img.save("output.jpg")

        # v = 10
        # metric_i = test_single_volume(image, label, model, dataset_id=dataset_id, predict_head=predict_head,
        #                               classes=n_classes, patch_size=[args.img_size, args.img_size],
        #                               test_save_path=test_save_path, case=case_name, z_spacing=args.z_spacing)
        # print(metric_i)
        # print('test script,',type(metric_i))
        # metric_list += np.array(metric_i, dtype=object)
        # print('metric_list',metric_list)
        # logging.info('idx %d case %s mean_dice %f mean_hd95 %f mean_iou %f' % (
        # i_batch, case_name, np.mean(metric_i, axis=0)[0], np.mean(metric_i, axis=0)[1], np.mean(metric_i, axis=0)[2]))

    # metric_list = metric_list / len(db_test)
    # for i in range(1, n_classes):
    #     logging.info('Mean class %d mean_dice %f mean_hd95 %f mean_iou %f' % (
    #     i, metric_list[i - 1][0], metric_list[i - 1][1], metric_list[i - 1][2]))
    #
    # performance = np.mean(metric_list, axis=0)[0]
    # mean_hd95 = np.mean(metric_list, axis=0)[1]
    # mean_iou = np.mean(metric_list, axis=0)[2]
    # logging.info('Testing performance in best val model: mean_dice : %f mean_hd95 : %f mean_iou : %f' % (
    # performance, mean_hd95, mean_iou))
    # return "Testing Finished!"


if __name__ == "__main__":
    #
    #     random.seed(args.seed)
    #     np.random.seed(args.seed)
    #     torch.manual_seed(args.seed)
    #     torch.cuda.manual_seed(args.seed)
    #
    #     dataset_config = {
    #         'flare22': {
    #             'Dataset': CardiacDataset,
    #             'root_path': './',
    #             'data_csv': './lists/datasets_flaretest.csv',
    #             'dataset_id': 0,
    #             'num_classes': 14,
    #             "predict_head": 0,
    #             'z_spacing': 1
    #         },
    #         'amos': {
    #             'Dataset': CardiacDataset,
    #             'root_path': './',
    #             'data_csv': './lists/datasets_amostest.csv',
    #             'dataset_id': 1,
    #             'num_classes': 16,
    #             "predict_head": 0,
    #             'z_spacing': 1
    #         },
    #         'word': {
    #             'Dataset': CardiacDataset,
    #             'root_path': './',
    #             'data_csv': './lists/datasets_wordtest.csv',
    #             'dataset_id': 2,
    #             'num_classes': 17,
    #             "predict_head": 0,
    #             'z_spacing': 1
    #         },
    #         # 'remap': {
    #         #     'Dataset': Synapse_dataset,
    #         #     'z_spacing': 1,
    #         #     'num_classes': 22,
    #         # },
    #
    #     }
    #
    #     num_classes_per_dataset = [14, 3, 16]
    #     # num_classes_per_dataset = [dataset_config[name]['num_classes'] for name in dataset_config]
    #     dataset_name = args.dataset
    #     # args.num_classes = dataset_config[dataset_name]['num_classes']
    #     # args.volume_path = dataset_config[dataset_name]['volume_path']
    #     args.Dataset = dataset_config[dataset_name]['Dataset']
    #     # args.data_csv = dataset_config[dataset_name]['data_csv']
    #     args.z_spacing = dataset_config[dataset_name]['z_spacing']
    #     args.is_pretrain = True
    #
    #     net = ViT_seg(config, img_size=args.img_size, num_classes=num_classes_per_dataset).cuda()
    net = SwinUnet(config, img_size=args.img_size, num_classes=[4, 4])
    net.load_state_dict(torch.load(args.model_path))
    net.to(device=args.device)
    # net.load_from(config)
    # snapshot = os.path.join(args.output_dir, 'best_model.pth')
    # if not os.path.exists(snapshot): snapshot = snapshot.replace('best_model', 'epoch_' + str(args.max_epochs - 1))
    inference(net)
