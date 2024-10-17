import os

import pandas as pd
from deep_utils import DirUtils
from sklearn.model_selection import train_test_split
from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("--split", action="store_true")
parser.add_argument("--name", default="datasets", type=str)
parser.add_argument("--img_ext", default=".npz", type=str)
parser.add_argument("--seg_ext", default=".npz", type=str)
parser.add_argument("--train", action="store_true")

args = parser.parse_args()

seed = 1234

nnunet_preprocess_path = "/media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_preprocessed/"


def npz_csv():
    datasets_config = {
        'CT_CORONARY': {
            'data_dir': f'{nnunet_preprocess_path}/Dataset002_china_narco/nnUNetPlans_2d',
            'num_classes': 3 + 1,  # plus background
            'predict_head': 1
        },
        'MRI_MM': {
            'data_dir': f'{nnunet_preprocess_path}/Dataset001_mm/nnUNetPlans_2d',
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
        for filepath in data_files:
            samples.append([
                filepath,
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


if __name__ == '__main__':
    npz_csv()
