import random
from os.path import split

import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import numpy as np


def random_rot_flip(image, label):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, predict_head, n_classes = (sample['image'], sample['label'],
                                                 sample['predict_head'], sample['n_classes'])

        if random.random() > 0.5:
            image, label = random_rot_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)

        x, y, *z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            if z:
                zoom_value = (self.output_size[0] / x, self.output_size[1] / y, 1)
            else:
                zoom_value = (self.output_size[0] / x, self.output_size[1] / y)
            image = zoom(image, zoom_value, order=3)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        if image.shape[:2] != tuple(self.output_size):
            raise ValueError("Shape is not correct")

        if label.shape[:2] != tuple(self.output_size):
            raise ValueError("Shape is not correct")
        if len(image.shape) == 2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:
            image = transforms.ToTensor()(image)
            # pass

        # image = image.permute(2, 0, 1)

        label = torch.from_numpy(label.astype(np.float32))

        # image = image.astype(np.float32)
        # label = label.astype(np.float32)
        sample = {'image': image,
                  'label': label,
                  'predict_head': predict_head,
                  'n_classes': n_classes}

        return sample


class CardiacDataset(Dataset):
    def __init__(self, csv_file_path, transform=None, modes='train'):
        self.transform = transform
        self.dataframe = pd.read_csv(csv_file_path)
        self.mode = modes

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_dir = row['img_dir']
        label_dir = row['label_dir']
        predict_head = row['predict_head']
        n_classes = row['n_classes']

        # image =
        # label =

        # filename = os.path.basename(data_dir)
        # case_name = filename.split('.')[0]

        sample = {
            'image': np.array(Image.open(img_dir)),
            'label': np.load(label_dir)['arr_0'],
            'predict_head': predict_head,
            'n_classes': n_classes,
            'case_name': split(label_dir)[-1].replace(".npz", "")
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == "__main__":
    csv_file = './lists/datasets_val.csv'

    transforms_list = [

        RandomGenerator(output_size=[224, 224]),
        # NormalizeSlice(),
        # Custom transformation
    ]

    # max_iterations = args.max_iterations
    dataset = CardiacDataset(
        csv_file_path=csv_file,  # Assuming there is a csv file for training data
        transform=transforms.Compose(transforms_list),
        modes='train'
    )

    # Print the dataset length
    print(f'Dataset length: {len(dataset)}')
    # print(dataset[0])
    # Print out the information for the samples in the specified range
    for i in range(1):
        sample = dataset[i]
        image = sample['image']
        label = sample['label']  # Assuming this is the mask
        print(sample)
        # case_name = sample['case_name']
        print(type(image))
        print(image.shape)

    from torch.utils.data.dataloader import default_collate


    def custom_collate_fn(batch):
        batch = [b for b in batch if b is not None]

        if len(batch) == 0:
            return None

        return default_collate(batch)


    dl = DataLoader(dataset,
                    batch_size=8,
                    shuffle=True,
                    num_workers=0,
                    pin_memory=True,
                    collate_fn=custom_collate_fn,
                    )
    for x in dl:
        v = 10
        print(x)
        break
