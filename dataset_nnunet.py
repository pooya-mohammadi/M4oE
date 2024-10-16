from os.path import split
from typing import Union, Tuple, List, Dict
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
import numpy as np
import pandas as pd
import torch
from PIL import Image
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
import inspect
import multiprocessing
import os
import shutil
import sys
import warnings
from copy import deepcopy
from datetime import datetime
from time import time, sleep
from typing import Tuple, Union, List

import numpy as np
import torch
from batchgenerators.dataloading.multi_threaded_augmenter import MultiThreadedAugmenter
from batchgenerators.dataloading.nondet_multi_threaded_augmenter import NonDetMultiThreadedAugmenter
from batchgenerators.dataloading.single_threaded_augmenter import SingleThreadedAugmenter
from batchgenerators.utilities.file_and_folder_operations import join, load_json, isfile, save_json, maybe_mkdir_p
from batchgeneratorsv2.helpers.scalar_type import RandomScalar
from batchgeneratorsv2.transforms.base.basic_transform import BasicTransform
from batchgeneratorsv2.transforms.intensity.brightness import MultiplicativeBrightnessTransform
from batchgeneratorsv2.transforms.intensity.contrast import ContrastTransform, BGContrast
from batchgeneratorsv2.transforms.intensity.gamma import GammaTransform
from batchgeneratorsv2.transforms.intensity.gaussian_noise import GaussianNoiseTransform
from batchgeneratorsv2.transforms.nnunet.random_binary_operator import ApplyRandomBinaryOperatorTransform
from batchgeneratorsv2.transforms.nnunet.remove_connected_components import \
    RemoveRandomConnectedComponentFromOneHotEncodingTransform
from batchgeneratorsv2.transforms.nnunet.seg_to_onehot import MoveSegAsOneHotToDataTransform
from batchgeneratorsv2.transforms.noise.gaussian_blur import GaussianBlurTransform
from batchgeneratorsv2.transforms.spatial.low_resolution import SimulateLowResolutionTransform
from batchgeneratorsv2.transforms.spatial.mirroring import MirrorTransform
from batchgeneratorsv2.transforms.spatial.spatial import SpatialTransform
from batchgeneratorsv2.transforms.utils.compose import ComposeTransforms
from batchgeneratorsv2.transforms.utils.deep_supervision_downsampling import DownsampleSegForDSTransform
from batchgeneratorsv2.transforms.utils.nnunet_masking import MaskImageTransform
from batchgeneratorsv2.transforms.utils.pseudo2d import Convert3DTo2DTransform, Convert2DTo3DTransform
from batchgeneratorsv2.transforms.utils.random import RandomTransform
from batchgeneratorsv2.transforms.utils.remove_label import RemoveLabelTansform
from batchgeneratorsv2.transforms.utils.seg_to_regions import ConvertSegmentationToRegionsTransform
from torch import autocast, nn
from torch import distributed as dist
from torch._dynamo import OptimizedModule
from torch.cuda import device_count
from torch.cuda.amp import GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

from nnunetv2.configuration import ANISO_THRESHOLD, default_num_processes
from nnunetv2.evaluation.evaluate_predictions import compute_metrics_on_folder
from nnunetv2.inference.export_prediction import export_prediction_from_logits, resample_and_save
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor
from nnunetv2.inference.sliding_window_prediction import compute_gaussian
from nnunetv2.paths import nnUNet_preprocessed, nnUNet_results
from nnunetv2.training.data_augmentation.compute_initial_patch_size import get_patch_size
from nnunetv2.training.dataloading.data_loader_2d import nnUNetDataLoader2D
from nnunetv2.training.dataloading.data_loader_3d import nnUNetDataLoader3D
from nnunetv2.training.dataloading.nnunet_dataset import nnUNetDataset
from nnunetv2.training.dataloading.utils import get_case_identifiers, unpack_dataset
from nnunetv2.training.logging.nnunet_logger import nnUNetLogger
from nnunetv2.training.loss.compound_losses import DC_and_CE_loss, DC_and_BCE_loss
from nnunetv2.training.loss.deep_supervision import DeepSupervisionWrapper
from nnunetv2.training.loss.dice import get_tp_fp_fn_tn, MemoryEfficientSoftDiceLoss
from nnunetv2.training.lr_scheduler.polylr import PolyLRScheduler
from nnunetv2.utilities.collate_outputs import collate_outputs
from nnunetv2.utilities.crossval_split import generate_crossval_split
from nnunetv2.utilities.default_n_proc_DA import get_allowed_n_proc_DA
from nnunetv2.utilities.file_path_utilities import check_workers_alive_and_busy
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.helpers import empty_cache, dummy_context
from nnunetv2.utilities.label_handling.label_handling import convert_labelmap_to_one_hot, determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager


def random_rot_flip(image, label=None):
    k = np.random.randint(0, 4)
    image = np.rot90(image, k)
    if label is not None:
        label = np.rot90(label, k)
    axis = np.random.randint(0, 2)
    image = np.flip(image, axis=axis).copy()
    if label is not None:
        label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label=None):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    if label is not None:
        label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


class RandomGenerator:
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label, predict_head, n_classes = (sample['image'], sample['label'],
                                                 sample['predict_head'], sample['n_classes'])
        # if label is not None:
        #     if random.random() > 0.5:
        #         image, label = random_rot_flip(image, label)
        #     elif random.random() > 0.5:
        #         image, label = random_rotate(image, label)
        # else:
        #     if random.random() > 0.5:
        #         image, _ = random_rot_flip(image)
        #     elif random.random() > 0.5:
        #         image, _ = random_rotate(image)

        x, y, *z = image.shape
        if x != self.output_size[0] or y != self.output_size[1]:
            if z:
                zoom_value = (self.output_size[0] / x, self.output_size[1] / y, 1)
            else:
                zoom_value = (self.output_size[0] / x, self.output_size[1] / y)
            image = zoom(image, zoom_value, order=3)
            if label is not None:
                label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

        if image.shape[:2] != tuple(self.output_size):
            raise ValueError("Shape is not correct")

        if label is not None and label.shape[:2] != tuple(self.output_size):
            raise ValueError("Shape is not correct")

        if len(image.shape) == 2:
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        else:
            image = transforms.ToTensor()(image)
            # pass

        # image = image.permute(2, 0, 1)
        if label is not None:
            label = torch.from_numpy(label.astype(np.float32))

        # image = image.astype(np.float32)
        # label = label.astype(np.float32)
        sample = {'image': image,
                  'label': label,
                  'predict_head': predict_head,
                  'n_classes': n_classes}

        return sample


class NNUnetCreator:
    def __init__(self, patch_size=(224, 224), rotation_for_da=(-3.141592653589793, 3.141592653589793),
                 mirror_axes=(0, 1), foreground_labels=(1, 2, 3)):
        self.patch_size = patch_size
        self.rotation_for_da = rotation_for_da
        self.mirror_axes = mirror_axes

    @staticmethod
    def get_training_transforms(
            patch_size: Union[np.ndarray, Tuple[int]],
            rotation_for_DA: RandomScalar,
            deep_supervision_scales: Union[List, Tuple, None],
            mirror_axes: Tuple[int, ...],
            do_dummy_2d_data_aug: bool,
            use_mask_for_norm: List[bool] = None,
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        if do_dummy_2d_data_aug:
            ignore_axes = (0,)
            transforms.append(Convert3DTo2DTransform())
            patch_size_spatial = patch_size[1:]
        else:
            patch_size_spatial = patch_size
            ignore_axes = None
        transforms.append(
            SpatialTransform(
                patch_size_spatial, patch_center_dist_from_border=0, random_crop=False, p_elastic_deform=0,
                p_rotation=0.2,
                rotation=rotation_for_DA, p_scaling=0.2, scaling=(0.7, 1.4), p_synchronize_scaling_across_axes=1,
                bg_style_seg_sampling=False  # , mode_seg='nearest'
            )
        )

        if do_dummy_2d_data_aug:
            transforms.append(Convert2DTo3DTransform())

        transforms.append(RandomTransform(
            GaussianNoiseTransform(
                noise_variance=(0, 0.1),
                p_per_channel=1,
                synchronize_channels=True
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GaussianBlurTransform(
                blur_sigma=(0.5, 1.),
                synchronize_channels=False,
                synchronize_axes=False,
                p_per_channel=0.5, benchmark=True
            ), apply_probability=0.2
        ))
        transforms.append(RandomTransform(
            MultiplicativeBrightnessTransform(
                multiplier_range=BGContrast((0.75, 1.25)),
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            ContrastTransform(
                contrast_range=BGContrast((0.75, 1.25)),
                preserve_range=True,
                synchronize_channels=False,
                p_per_channel=1
            ), apply_probability=0.15
        ))
        transforms.append(RandomTransform(
            SimulateLowResolutionTransform(
                scale=(0.5, 1),
                synchronize_channels=False,
                synchronize_axes=True,
                ignore_axes=ignore_axes,
                allowed_channels=None,
                p_per_channel=0.5
            ), apply_probability=0.25
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=1,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.1
        ))
        transforms.append(RandomTransform(
            GammaTransform(
                gamma=BGContrast((0.7, 1.5)),
                p_invert_image=0,
                synchronize_channels=False,
                p_per_channel=1,
                p_retain_stats=1
            ), apply_probability=0.3
        ))
        if mirror_axes is not None and len(mirror_axes) > 0:
            transforms.append(
                MirrorTransform(
                    allowed_axes=mirror_axes
                )
            )

        if use_mask_for_norm is not None and any(use_mask_for_norm):
            transforms.append(MaskImageTransform(
                apply_to_channels=[i for i in range(len(use_mask_for_norm)) if use_mask_for_norm[i]],
                channel_idx_in_seg=0,
                set_outside_to=0,
            ))

        transforms.append(
            RemoveLabelTansform(-1, 0)
        )
        if is_cascaded:
            assert foreground_labels is not None, 'We need foreground_labels for cascade augmentations'
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )
            transforms.append(
                RandomTransform(
                    ApplyRandomBinaryOperatorTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        strel_size=(1, 8),
                        p_per_label=1
                    ), apply_probability=0.4
                )
            )
            transforms.append(
                RandomTransform(
                    RemoveRandomConnectedComponentFromOneHotEncodingTransform(
                        channel_idx=list(range(-len(foreground_labels), 0)),
                        fill_with_other_class_p=0,
                        dont_do_if_covers_more_than_x_percent=0.15,
                        p_per_label=1
                    ), apply_probability=0.2
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))

        return ComposeTransforms(transforms)

    @staticmethod
    def get_validation_transforms(
            deep_supervision_scales: Union[List, Tuple, None],
            is_cascaded: bool = False,
            foreground_labels: Union[Tuple[int, ...], List[int]] = None,
            regions: List[Union[List[int], Tuple[int, ...], int]] = None,
            ignore_label: int = None,
    ) -> BasicTransform:
        transforms = []
        transforms.append(
            RemoveLabelTansform(-1, 0)
        )

        if is_cascaded:
            transforms.append(
                MoveSegAsOneHotToDataTransform(
                    source_channel_idx=1,
                    all_labels=foreground_labels,
                    remove_channel_from_source=True
                )
            )

        if regions is not None:
            # the ignore label must also be converted
            transforms.append(
                ConvertSegmentationToRegionsTransform(
                    regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
                    channel_in_seg=0
                )
            )

        if deep_supervision_scales is not None:
            transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
        return ComposeTransforms(transforms)

    def get_transforms(self, foreground_labels=(1, 2, 3)):
        tr_transforms = self.get_training_transforms(
            self.patch_size, self.rotation_for_da, None, self.mirror_axes, False,
            use_mask_for_norm=None,
            is_cascaded=False,
            foreground_labels=foreground_labels,  # Does not matter cause is_cascaded is set to False!
            regions=None,
            ignore_label=None)

        # validation pipeline
        val_transforms = self.get_validation_transforms(None,
                                                        is_cascaded=False,
                                                        foreground_labels=foreground_labels,
                                                        regions=None,
                                                        ignore_label=None)
        return tr_transforms, val_transforms


class NNUNetCardiacDataset(Dataset):

    def __init__(self, csv_file_path, transform=None):
        self.transform = transform
        if isinstance(csv_file_path, list):
            self.dataframe = pd.DataFrame(csv_file_path, columns=["img_dir", "label_dir",
                                                                  "predict_head", "n_classes"])
        elif isinstance(csv_file_path, dict):
            self.dataframe = pd.DataFrame([csv_file_path], columns=["img_dir", "label_dir",
                                                                    "predict_head", "n_classes"])
        else:
            self.dataframe = pd.read_csv(csv_file_path)
        # self.mode = modes

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_dir = row['img_dir']
        label_dir = row.get('label_dir')
        predict_head = row['predict_head']
        n_classes = row['n_classes']

        if label_dir is not None:
            if label_dir.endswith(".npz"):
                label = np.load(label_dir)['arr_0'].astype(np.int32)
            elif label_dir.endswith(".jpg"):
                label = np.array(Image.open(label_dir))
                if label.max() == 0:
                    label = label.astype(np.int32)
                else:
                    label = (label / label.max() * (n_classes - 1)).astype(np.int32)
            else:
                raise ValueError(f"input {label_dir} is not valid")
        else:
            label = None

        sample = {
            'image': np.array(Image.open(img_dir)),
            'label': label,
            'predict_head': predict_head,
            'n_classes': n_classes,
            'case_name': split(img_dir)[-1].replace(".npz", "").replace(".jpg", "")
        }

        if self.transform:
            sample = self.transform(sample)

        sample = {k: v for k, v in sample.items() if v is not None}

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
        # modes='train'
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

    sample = {"img_dir": "/home/aicvi/projects/Swin-MAE-datasets/images/ct_coronary/12069336_0266.jpg",
              'n_classes': 3 + 1,
              'predict_head': 1,
              "label_dir": None
              }
    dataset = CardiacDataset(
        csv_file_path=sample,  # Assuming there is a csv file for training data
        transform=transforms.Compose(transforms_list),
        # modes='train'
    )

    # Print the dataset length
    print(f'Dataset length: {len(dataset)}')
    sample = dataset[0]
    image = sample['image']
    print(image.shape)
