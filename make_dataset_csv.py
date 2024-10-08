import os

import pandas as pd
from deep_utils import DirUtils
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--split", action="store_true")
parser.add_argument("--name", default="datasets", type=str)
parser.add_argument("--img_ext", default=".jpg", type=str)
parser.add_argument("--seg_ext", default=".jpg", type=str)
parser.add_argument("--train", action="store_true")

args = parser.parse_args()

seed = 1234


def npz_csv(csv_task_name='datasets_v217'):
    datasets_config = {
        'CT_CORONARY': {
            'img_dir': f'../Swin-MAE-datasets/{"train" if args.train else "val"}/images/ct_coronary',
            'label_dir': f'../Swin-MAE-datasets/{"train" if args.train else "val"}/labels/ct_coronary',
            'num_classes': 3 + 1,
            'predict_head': 1
        },
        'MRI_MM': {
            'img_dir': f'../Swin-MAE-datasets/{"train" if args.train else "val"}/images/mri_mm',
            'label_dir': f'../Swin-MAE-datasets/{"train" if args.train else "val"}/labels/mri_mm',
            'num_classes': 3 + 1,
            'predict_head': 0
        },
    }

    csv_file_path = f'./lists/{csv_task_name}.csv'

    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)
    samples = []
    columns = ["img_dir", "label_dir", "predict_head", "n_classes"]

    for dataset_name, config in datasets_config.items():

        data_files = DirUtils.list_dir_full_path(config['img_dir'], interest_extensions=args.img_ext, return_dict=True)
        seg_files = DirUtils.list_dir_full_path(config['label_dir'], interest_extensions=args.seg_ext, return_dict=True)

        for seg_file_key in seg_files.keys():
            if seg_file_key not in data_files:
                continue
            img_file_path = data_files[seg_file_key]
            seg_file_path = seg_files[seg_file_key]
            samples.append([
                img_file_path,
                seg_file_path,
                config['predict_head'],
                config['num_classes'],
            ])

    if args.split:
        train, val = train_test_split(samples, test_size=0.1, random_state=seed)
        pd.DataFrame(train, columns=columns).to_csv(DirUtils.split_extension(csv_file_path, suffix="_train"),
                                                    index=False)
        pd.DataFrame(val, columns=columns).to_csv(DirUtils.split_extension(csv_file_path, suffix="_val"), index=False)
    else:
        pd.DataFrame(samples, columns=columns).to_csv(csv_file_path, index=False)


if __name__ == "__main__":
    npz_csv(csv_task_name=args.name)
