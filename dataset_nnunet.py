import time

import torch
import random
from os.path import split
from typing import Tuple, Union, List

import numpy as np
import pandas as pd
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
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class NNUnetCreator:
    def __init__(self, patch_size=(224, 224), rotation_for_da=(-3.141592653589793, 3.141592653589793),
                 mirror_axes=(0, 1), foreground_labels=(1, 2, 3)):
        self.patch_size = patch_size
        self.rotation_for_da = rotation_for_da
        self.mirror_axes = mirror_axes
        self.foreground_labels = foreground_labels

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

    # @staticmethod
    # def get_validation_transforms(
    #         deep_supervision_scales: Union[List, Tuple, None],
    #         is_cascaded: bool = False,
    #         foreground_labels: Union[Tuple[int, ...], List[int]] = None,
    #         regions: List[Union[List[int], Tuple[int, ...], int]] = None,
    #         ignore_label: int = None,
    # ) -> BasicTransform:
    #     transforms = []
    #     transforms.append(
    #         RemoveLabelTansform(-1, 0)
    #     )

    # if is_cascaded:
    #     transforms.append(
    #         MoveSegAsOneHotToDataTransform(
    #             source_channel_idx=1,
    #             all_labels=foreground_labels,
    #             remove_channel_from_source=True
    #         )
    #     )

    # if regions is not None:
    #     the ignore label must also be converted
    # transforms.append(
    #     ConvertSegmentationToRegionsTransform(
    #         regions=list(regions) + [ignore_label] if ignore_label is not None else regions,
    #         channel_in_seg=0
    #     )
    # )
    #
    # if deep_supervision_scales is not None:
    #     transforms.append(DownsampleSegForDSTransform(ds_scales=deep_supervision_scales))
    # return ComposeTransforms(transforms)

    def get_transforms(self):
        tr_transforms = self.get_training_transforms(
            self.patch_size, self.rotation_for_da, None, self.mirror_axes, False,
            use_mask_for_norm=None,
            is_cascaded=False,
            foreground_labels=self.foreground_labels,  # Does not matter cause is_cascaded is set to False!
            regions=None,
            ignore_label=None)

        # validation pipeline
        val_transforms = self.get_training_transforms(self.patch_size, (0, 0), None, None, False,
                                                      use_mask_for_norm=None,
                                                      is_cascaded=False,
                                                      foreground_labels=self.foreground_labels,
                                                      # Does not matter cause is_cascaded is set to False!
                                                      regions=None,
                                                      ignore_label=None)
        return tr_transforms, val_transforms


class NNUNetCardiacDataset(Dataset):

    def __init__(self, csv_file_path, transform=None, verbose: bool = False):
        self.transform = transform
        if isinstance(csv_file_path, list):
            self.dataframe = pd.DataFrame(csv_file_path, columns=["data_dir", "predict_head", "n_classes"])
        elif isinstance(csv_file_path, dict):
            self.dataframe = pd.DataFrame([csv_file_path], columns=["data_dir", "predict_head", "n_classes"])
        else:
            self.dataframe = pd.read_csv(csv_file_path)
        self.verbose = verbose

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        tic = time.time()
        row = self.dataframe.iloc[idx]
        data_dir = row['data_dir']
        predict_head = row['predict_head']
        n_classes = row['n_classes']

        data = np.load(data_dir)
        img = data['data'].squeeze(0)
        seg = data['seg'].squeeze(0)
        if len(img.shape) == 3:
            r = random.randint(0, img.shape[0] - 1)
            img = img[r]
            seg = seg[r]

        if self.transform:
            img = torch.from_numpy(img).float()
            seg = torch.from_numpy(seg).to(torch.int16)
            tmp = self.transform(**{'image': img.unsqueeze(0), 'segmentation': seg.unsqueeze(0)})
            img = tmp['image']
            seg = tmp['segmentation'].squeeze(0)
        sample = {
            'image': img,
            'label': seg,
            'predict_head': predict_head,
            'n_classes': n_classes,
            'case_name': split(data_dir)[-1].replace(".npz", "").replace(".jpg", "").replace(".npy", "")
        }
        sample = {k: v for k, v in sample.items() if v is not None}
        if self.verbose:
            print(f"Loading {data_dir} took: {time.time() - tic}")
        return sample


if __name__ == '__main__':
    from torch.utils.data._utils.collate import default_collate


    def custom_collate_fn(batch):
        batch = [b for b in batch if b is not None]

        if len(batch) == 0:
            return None

        return default_collate(batch)


    img_size = 224

    tr_transform, val_transform = NNUnetCreator(patch_size=(img_size, img_size)).get_transforms()
    # max_iterations = args.max_iterations
    dataset_train = NNUNetCardiacDataset(
        verbose=True,
        csv_file_path="lists/datasets_train.csv",  # Assuming there is a csv file for training data
        transform=tr_transform,
    )

    dataset_val = NNUNetCardiacDataset(
        verbose=True,
        csv_file_path="lists/datasets_val.csv",  # Assuming there is a csv file for training data
        transform=val_transform,
    )
    trainloader = DataLoader(dataset_train,
                             batch_size=4,
                             shuffle=True,
                             num_workers=2,
                             prefetch_factor=1,
                             # pin_memory=True,
                             # collate_fn=custom_collate_fn,
                             )

    # valloader = DataLoader(dataset_val,
    #                        batch_size=8,
    #                        shuffle=False,
    #                        num_workers=8,
    #                        pin_memory=True,
    #                        # collate_fn=custom_collate_fn,
    #                        )
    tic = time.time()
    print("Starting to load data")
    for i in trainloader:
        print(i['image'].shape)
        break
    print("Overall: ", time.time() - tic)
