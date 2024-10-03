import csv
import os

import numpy as np


def npz_csv(csv_task_name='datasets_v217'):
    datasets_config = {
        'CT_CORONARY': {
            'data_dir': '/home/aicvi/projects/Swin-MAE/datasets/two_d/ct_coronary',
            'img_idx': 0,
            'label_idx': 1,
            'dataset_id': 1,
            'num_classes': 16,
            'predict_head': 1
        },
        'MRI_MM': {
            'data_dir': '/home/aicvi/projects/Swin-MAE/datasets/two_d/mri_mm',
            'img_idx': 0,
            'label_idx': 1,
            'dataset_id': 0,
            'num_classes': 14,
            'predict_head': 0
        },

    }
    # datasets_config = dataset_config_linux
    # csv_file_path = './lists/datasets_v10.csv'
    csv_file_path = f'./lists/{csv_task_name}.csv'

    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')

        writer.writerow(["data_dir", "label_dir", "img_idx", "label_idx", "dataset_id", "predict_head", "n_classes"])

        for dataset_name, config in datasets_config.items():

            data_files = [f for f in os.listdir(config['data_dir']) if f.endswith('.jpg')]

            for npz_file in data_files:
                npz_file_path = os.path.join(config['data_dir'], npz_file)

                writer.writerow([
                    npz_file_path,
                    config['img_idx'],
                    config['label_idx'],
                    config['dataset_id'],
                    config['predict_head'],
                    config['num_classes'],
                ])


def npz_csv_testing():
    dataset_config_test = {}
    csv_file_path = './lists/datasets_multi_test_amos_mr_moe.csv'

    # Ensure the directory for the CSV file exists
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # Open the CSV file for writing
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        # Write the header row with an additional 'num_slices' column
        writer.writerow(["data_dir", "img_idx", "label_idx", "dataset_id", "predict_head", "n_classes"])

        # Iterate over each dataset in the configuration
        for dataset_name, config in dataset_config_test.items():
            # List all .npz files in the data_dir
            data_files = [f for f in os.listdir(config['data_dir']) if f.endswith('.jpg')]

            # Write a row for each .npz file found
            for npz_file in data_files:
                npz_file_path = os.path.join(config['data_dir'], npz_file)
                # Load the .npz file to find out the number of slices
                # npz_data = np.load(npz_file_path)

                writer.writerow([
                    npz_file_path,  # Full path to the .npz file
                    config['img_idx'],  # Image index
                    config['label_idx'],  # Label index
                    config['dataset_id'],  # Dataset ID
                    config['predict_head'],  # Prediction head
                    config['num_classes'],  # Number of classes
                ])


if __name__ == "__main__":
    npz_csv(csv_task_name='datasets')
    npz_csv_testing()
    # nifiti_csv()
