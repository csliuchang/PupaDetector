import torch
import torch.nn as nn
from .decode_head import BaseDecodeHead
from ...builder import HEADS, build_loss
import torch.nn.functional as F
from models.base.conv_module import ConvModule


@HEADS.register_module()
class OCRHead(BaseDecodeHead):
    def __init__(self, ocr_channels, scale=1, **kwargs):
        super(OCRHead, self).__init__()
        self.ocr_channels = ocr_channels
        self.scale = scale
        self.object_context_block = ObjectAttentionBlock(
            self.channels,
            self.ocr_channels,
            self.scale,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)
        self.spatial_gather_module = SpatialGatherModule(self.scale)

        self.bottleneck = ConvModule(
            self.in_channels,
            self.channels,
            3,
            padding=1,
            conv_cfg=self.conv_cfg,
            norm_cfg=self.norm_cfg,
            act_cfg=self.act_cfg)

    def forward(self, inputs, perv_output):
        x = self._transfrom_inputs(inputs)
        feats = self.bottleneck(x)
        context = self.spatial_gather_module(feats, perv_output)
        object_context = self.object_context_block(feats, context)
        output = self.cls_seg(object_context)

        return output
