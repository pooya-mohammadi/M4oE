from os.path import join
import argparse
import os
import shutil
from PIL import Image
import cv2
import numpy as np
import torch
from deep_utils import DirUtils, NIBUtils
from torch.utils.data import DataLoader
from tqdm import tqdm
from os.path import split
from config import get_config
from dataset_synapse import CardiacDataset
from networks.vision_transformer import SwinUnet
from trainer import custom_collate_fn

parser = argparse.ArgumentParser()

parser.add_argument("--input", type=str)
parser.add_argument("--output", type=str, default="")
parser.add_argument("--modality", type=str, choices=['mri_mm', "ct_coronary"])
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
parser.add_argument("--remove", action="store_true", help="removes the output ")
parser.add_argument("--copy", action="store_true", help="copy input to current dir")
parser.add_argument("--save_seg", action="store_true", help="save seg output")
parser.add_argument("--seg_path", default="")

args = parser.parse_args()
config = get_config(args)

args.device = "cpu" if not torch.cuda.is_available() else args.device

predict_head_map = {
    "mri_mm": 0,
    "ct_coronary": 1
}


def inference(model):
    model.eval()
    # h, w = 224, 224
    # metric_list = 0.0
    output_dir = DirUtils.split_extension(split(args.input)[-1], suffix='_split', current_extension=".nii.gz").replace(
        '.nii.gz', '')

    output = args.output or DirUtils.split_extension(split(args.input)[-1],
                                                     suffix='_output', current_extension=".nii.gz")
    if args.save_seg:
        seg_path = args.seg_path if args.seg_path else output.replace(".nii.gz", "")
    out_res_ = os.system(f"med2image -i {args.input} -d {output_dir}")
    if out_res_ != 0:
        raise RuntimeError(f"os.system raised error: {out_res_}")
    samples = [{
        "img_dir": item,
        'n_classes': 3 + 1,
        'predict_head': predict_head_map[args.modality],
        "label_dir": None
    } for item in DirUtils.list_dir_full_path(output_dir, interest_extensions=".jpg")]
    # sample = {
    #     "img_dir": "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/Swin-MAE-datasets/images/mri_mm/001_SA-2_0005.jpg",
    #     'n_classes': 3 + 1,
    #     'predict_head': predict_head_map[args.modality],
    #     "label_dir": None
    # }
    if args.save_seg:
        DirUtils.remove_create(seg_path)
    dataset = CardiacDataset(
        csv_file_path=samples,  # Assuming there is a csv file for training data
    )
    testloader = DataLoader(dataset, batch_size=1, num_workers=args.num_workers, collate_fn=custom_collate_fn)
    output_files = []
    for i_batch, sampled_batch in tqdm(enumerate(testloader)):
        image, label, predict_head = sampled_batch["image"], sampled_batch.get("label"), sampled_batch['predict_head']
        image = image.to(args.device)
        outputs = model(image, predict_head)
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        out = out.cpu().detach().numpy()

        out = (out / out.max() * 255).astype(np.uint8)
        if args.save_seg:
            img = Image.fromarray(out)
            img.save(join(seg_path, f"{i_batch:04}.jpg"))
        output_files.append(out)
    nib_array, nib_img = NIBUtils.get_array_img(args.input)
    output_files = np.concatenate(
        [cv2.resize(f, nib_array.shape[:2], interpolation=cv2.INTER_NEAREST_EXACT)[..., None] for f in output_files],
        axis=-1)
    NIBUtils.save_sample(output, output_files, nib_img=nib_img)
    if args.remove:
        shutil.rmtree(output_dir)
    if args.copy:
        shutil.copy(args.input, "./")


if __name__ == "__main__":
    net = SwinUnet(config, img_size=args.img_size, num_classes=[4, 4])
    net.load_state_dict(torch.load(args.model_path))
    net.to(device=args.device)
    inference(net)
