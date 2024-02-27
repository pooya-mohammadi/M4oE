import os
import csv

import numpy as np
import tqdm as tqdm

import os
import csv
import numpy as np

def npz_csv(system='Linux',csv_task_name='datasets_v217'):
    datasets_config = {}
        datasets_config = dataset_config_windows
    if system == 'Linux':
        dataset_config_linux = {
            'Flare22': {
                'data_dir': 'datasets/slice/flare22',
                'img_idx': 0,
                'label_idx': 1,
                'dataset_id': 0,
                'num_classes': 14,
                'predict_head': 0
            },
            'AMOS': {
                'data_dir': 'datasets/slice/amos',
                'img_idx': 0,
                'label_idx': 1,
                'dataset_id': 1,
                'num_classes': 16,
                'predict_head': 1
            },
            'ALTAS': {
                'data_dir': 'datasets/slice/altas',
                'img_idx': 0,
                'label_idx': 1,
                'dataset_id': 1,
                'num_classes': 3,
                'predict_head': 1
            },
            'AMOS_MR': {
                'data_dir': 'datasets/slice/amos_mr',
                'img_idx': 0,
                'label_idx': 1,
                'dataset_id': 2,
                'num_classes': 16,
                'predict_head': 2
            },
        }
        datasets_config = dataset_config_linux
    # csv_file_path = './lists/datasets_v10.csv'
    csv_file_path = f'./lists/{csv_task_name}.csv'
    # 确保CSV文件的目录存在
    os.makedirs(os.path.dirname(csv_file_path), exist_ok=True)

    # 打开CSV文件进行写入
    with open(csv_file_path, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        # 写入CSV文件头
        writer.writerow(["data_dir", "img_idx", "label_idx", "dataset_id", "predict_head", "n_classes"])

        # 遍历配置中的每个数据集
        for dataset_name, config in datasets_config.items():
            # 列出data_dir中所有的.npz文件
            data_files = [f for f in os.listdir(config['data_dir']) if f.endswith('.npz')]

            # 为找到的每个.npz文件写入一行
            for npz_file in data_files:
                npz_file_path = os.path.join(config['data_dir'], npz_file)

                writer.writerow([
                    npz_file_path,  # .npz文件的完整路径
                    config['img_idx'],  # 图像索引
                    config['label_idx'],  # 标签索引
                    config['dataset_id'],  # 数据集ID
                    config['predict_head'],  # 预测头
                    config['num_classes'],  # 类别数量
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
            data_files = [f for f in os.listdir(config['data_dir']) if f.endswith('.npz')]

            # Write a row for each .npz file found
            for npz_file in data_files:
                npz_file_path = os.path.join(config['data_dir'], npz_file)
                # Load the .npz file to find out the number of slices
                npz_data = np.load(npz_file_path)

                writer.writerow([
                    npz_file_path,  # Full path to the .npz file
                    config['img_idx'],  # Image index
                    config['label_idx'],  # Label index
                    config['dataset_id'],  # Dataset ID
                    config['predict_head'],  # Prediction head
                    config['num_classes'],  # Number of classes
                ])
if __name__ == "__main__":
    # npz_csv(system='Linux',csv_task_name='datasets')
    npz_csv_testing()
    # nifiti_csv()