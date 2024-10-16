# M<sup>4</sup>oE

The codes for the work "M<sup>4</sup>oE: Foundation Model for Medical Multimodal Image Segmentation with Mixture of Experts"([Arxiv](https://arxiv.org/abs/2405.09446)). Our paper has been accepted by MICCAI 2024.

## 1. Pretrained Models: You can choose a pretrained model based on your preference.

### Opt 1.1 Download pre-trained swin transformer model (Swin-T)

* [Get pre-trained model in this link] (https://drive.google.com/drive/folders/1UC3XOoezeum0uck4KBVGa8osahs6rKUY?usp=sharing): Put pretrained Swin-T into folder "pretrained_ckpt/”

### Opt 1.2 Pretraining by using MAE methods on your datasets.

- References: (https://github.com/zian-xu/swin-mae)

## 2. Prepare data

- [AMOS 22](https://amos22.grand-challenge.org/Dataset/)
- [FLARE 22](https://flare22.grand-challenge.org/)
- [ATLAS](https://atlas.grand-challenge.org/)

## 3. Environment

- Please prepare an environment with python=3.8.10, and then use the command "pip install -r requirements.txt" for the dependencies.

## 4. Train/Test

- Run the train script on synapse dataset. The batch size we used is 36. If you do not have enough GPU memory, the bacth size can be reduced to 12 or 6 to save memory.

- Train

```bash
sh train.sh or python train.py --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --root_path your DATA_DIR --max_epochs 150 --output_dir your OUT_DIR  --img_size 224 --base_lr 0.05 --batch_size 24
```

- Test 

```bash
sh test.sh or python test.py --cfg configs/swin_tiny_patch4_window7_224_lite.yaml --is_saveni --volume_path your DATA_DIR --output_dir your OUT_DIR --max_epoch 150 --base_lr 0.05 --img_size 224 --batch_size 24
```

## References

* [SwinUnet](https://github.com/HuCaoFighting/Swin-Unet)
* [Swin-MAE](https://github.com/zian-xu/swin-mae)
* [MAE](https://github.com/facebookresearch/mae)

## Citation

```bibtex
@article{jiang2024m4oe,
  title   = {M$^4$oE: A Foundation Model for Medical Multimodal Image Segmentation with Mixture of Experts},
  author  = {Yufeng Jiang and Yiqing Shen},
  year    = {2024},
  journal = {arXiv preprint arXiv: 2405.09446}
}
```

# Trials
First try:
```commandline
# SERVER
python make_dataset_csv.py --name datasets_train --train --nnunet /scratch/isaac/nnunet_projects/nnunet_preprocessed
python make_dataset_csv.py --name datasets_val --nnunet /scratch/isaac/nnunet_projects/nnunet_preprocessed
# LOCAL
python make_dataset_csv.py --name datasets_train --train 
python make_dataset_csv.py --name datasets_val 

# Add --split to split the data!
python make_dataset_csv.py --name datasets_train --train --nnunet /scratch/isaac/nnunet_projects/nnunet_preprocessed --split
python make_dataset_csv.py --name datasets_val --nnunet /scratch/isaac/nnunet_projects/nnunet_preprocessed --split

CUDA_AVAIABLE_DEVICES=2 python train.py --cfg configs/dataset.yaml --max_epochs 150 --output_dir output  --img_size 224 --base_lr 0.05 --batch_size 256 --num_workers 64 --patience 25
```


# Inference:
```commandline
python inference_niftti.py --input /media/aicvi/11111bdb-a0c7-4342-9791-36af7eb70fc0/NNUNET_OUTPUT/nnunet_raw/Dataset001_mm/imagesTr/B0N3W8-2_0000.nii.gz --modality mri_mm --copy --save_seg
```