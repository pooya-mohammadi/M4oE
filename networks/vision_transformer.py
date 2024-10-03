# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging

import torch
import torch.nn as nn

from .swin_transformer_moe_decoder_encoder import SwinTransformerSys

logger = logging.getLogger(__name__)


# print(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
class SwinUnet(nn.Module):
    def __init__(self, config, img_size=224, num_classes=None, zero_head=False, vis=False):
        super(SwinUnet, self).__init__()
        self.num_classes = num_classes or []
        self.zero_head = zero_head
        self.config = config

        self.swin_unet = SwinTransformerSys(img_size=config.DATA.IMG_SIZE,
                                            patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                            in_chans=config.MODEL.SWIN.IN_CHANS,
                                            num_classes=self.num_classes,
                                            embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                            depths=config.MODEL.SWIN.DEPTHS,
                                            num_heads=config.MODEL.SWIN.NUM_HEADS,
                                            window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                            mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                            qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                            qk_scale=config.MODEL.SWIN.QK_SCALE,
                                            drop_rate=config.MODEL.DROP_RATE,
                                            drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                            ape=config.MODEL.SWIN.APE,
                                            patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                            use_checkpoint=config.TRAIN.USE_CHECKPOINT,
                                            )
        # print(self.swin_unet.flops()/1e9)

    def forward(self, x, dataset_id, predict_head):
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # make channels 3
        logits = self.swin_unet(x, dataset_id, predict_head)
        return logits

    def load_from(self, config):
        pretrained_path = config.MODEL.PRETRAIN_CKPT
        # logging.info("pretrained_path:{}".format(pretrained_path))
        if pretrained_path is not None:
            print("pretrained_path:{}".format(pretrained_path))
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            pretrained_dict = torch.load(pretrained_path, map_location=device)
            if "model" not in pretrained_dict:
                print("---start load pretrained model by splitting---")
                pretrained_dict = {k[17:]: v for k, v in pretrained_dict.items()}
                for k in list(pretrained_dict.keys()):
                    if "output" in k:
                        print("delete key:{}".format(k))
                        del pretrained_dict[k]
                msg = self.swin_unet.load_state_dict(pretrained_dict, strict=False)
                # print(msg)
                return
            pretrained_dict = pretrained_dict['model']
            print("---start load pretrained modle of swin encoder---")

            # filter gating keys and decoder keys
            filtered_dict = {
                k: v for k, v in pretrained_dict.items()
                if 'gating' not in k and 'layers_up' not in k
            }
            model_dict = self.swin_unet.state_dict()
            # full_dict = copy.deepcopy(pretrained_dict)
            # for k, v in pretrained_dict.items():
            #     pass
            #     if "layers." in k:
            #         current_layer_num = 3 - int(k[7:8])
            #         current_k = "layers_up." + str(current_layer_num) + k[8:]
            #         full_dict.update({current_k: v})
            # for k in list(full_dict.keys()):
            #     if k in model_dict:
            #         if full_dict[k].shape != model_dict[k].shape:
            #             print("delete:{};shape pretrain:{};shape model:{}".format(k, v.shape, model_dict[k].shape))
            #             del full_dict[k]
            full_dict = {}
            for k, v in filtered_dict.items():
                if k in model_dict and model_dict[k].size() == v.size():
                    full_dict[k] = v
                else:
                    print("Skipped loading parameter:", k)

            msg = self.swin_unet.load_state_dict(full_dict, strict=False)
            # print(msg)
        else:
            print("none pretrain")
