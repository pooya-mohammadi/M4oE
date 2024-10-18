from os.path import join, split, exists
import numpy as np
import os

import pandas as pd
from deep_utils import DirUtils
from argparse import ArgumentParser
from joblib import Parallel, delayed

parser = ArgumentParser()
parser.add_argument("--split", action="store_true")
parser.add_argument("--name", default="datasets", type=str)
parser.add_argument("--n_jobs", default=10, type=int)
parser.add_argument("--img_ext", default=".npz", type=str)
parser.add_argument("--seg_ext", default=".npz", type=str)
parser.add_argument("--train", action="store_true")
parser.add_argument("--nnunet",
                    default="/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/")

args = parser.parse_args()

seed = 1234


def npz_csv():
    datasets_config = {
        'CT_CORONARY': {
            'data_dir': f'{args.nnunet}/Dataset002_china_narco/nnUNetPlans_2d',
            'num_classes': 3 + 1,  # plus background
            'predict_head': 1
        },
        'MRI_MM': {
            'data_dir': f'{args.nnunet}/Dataset001_mm/nnUNetPlans_2d',
            'num_classes': 3 + 1,  # plus background
            'predict_head': 0
        },
    }

    csv_file_path = f'./lists/{args.name}.csv'

    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    samples = []
    columns = ["data_dir", "predict_head", "n_classes"]

    for dataset_name, config in datasets_config.items():
        data_files = DirUtils.list_dir_full_path(config['data_dir'], interest_extensions=args.img_ext)

        if args.split:
            split_path = DirUtils.split_extension(join(config['data_dir']), suffix="_split")
            os.makedirs(split_path, exist_ok=True)
        else:
            split_path = None
        samples_ = Parallel(n_jobs=args.n_jobs)(
            delayed(process_file)(config, split_path, filepath) for filepath in data_files)
        samples.extend(samples_)
    pd.DataFrame(samples, columns=columns).to_csv(csv_file_path, index=False)


def process_file(config, split_path, filepath):
    if split_path:
        file_data = np.load(filepath)
        img = file_data['data']
        seg = file_data['seg']
        for z_index in range(img.shape[1]):
            img_ = img[:, z_index, ...]
            seg_ = seg[:, z_index, ...]
            img_path = join(split_path,
                            f"{DirUtils.split_extension(split(filepath)[-1], suffix=f'_{z_index:04}_img')}")
            seg_path = join(split_path,
                            f"{DirUtils.split_extension(split(filepath)[-1], suffix=f'_{z_index:04}_seg')}")
            if not exists(img_path) or not exists(seg_path):
                np.savez(img_path, img_)  # noqa
                np.savez(seg_path, seg_)  # noqa
    sample = [
        filepath,
        config['predict_head'],
        config['num_classes'],
    ]
    return sample


if __name__ == '__main__':
    npz_csv()
