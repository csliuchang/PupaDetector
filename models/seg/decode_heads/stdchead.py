import torch
import torch.nn as nn
from .decode_head import BaseDecodeHead
from ...builder import HEADS, build_loss
import torch.nn.functional as F

BatchNorm2d = nn.BatchNorm2d


@HEADS.register_module()
class STDCHead(BaseDecodeHead):
    def __init__(self, loss, bound_loss, use_boundary_2=True, use_boundary_4=True, use_boundary_8=True, **kwargs):
        super(STDCHead, self).__init__(**kwargs)
        self.arm16 = AttentionRefinementModule(512, 128)
        inplanes = 1024
        self.arm32 = AttentionRefinementModule(inplanes, 128)
        self.conv_head32 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_head16 = ConvBNReLU(128, 128, ks=3, stride=1, padding=1)
        self.conv_avg = ConvBNReLU(inplanes, 128, ks=1, stride=1, padding=0)

        self.use_boundary_2 = use_boundary_2
        self.use_boundary_4 = use_boundary_4
        self.use_boundary_8 = use_boundary_8

        conv_out_inplanes = 128
        sp2_inplanes = 32
        sp4_inplanes = 64
        sp8_inplanes = 256
        sp16_inplanes = 512
        inplane = sp8_inplanes + conv_out_inplanes

        self.ffm = FeatureFusionModule(inplane, 256)
        self.conv_out = BiSeNetOutput(256, 256, self.num_classes + 1)
        self.conv_out16 = BiSeNetOutput(conv_out_inplanes, 64, self.num_classes + 1)
        self.conv_out32 = BiSeNetOutput(conv_out_inplanes, 64, self.num_classes + 1)

        self.conv_out_sp16 = BiSeNetOutput(sp16_inplanes, 64, 1)

        self.conv_out_sp8 = BiSeNetOutput(sp8_inplanes, 64, 1)
        self.conv_out_sp4 = BiSeNetOutput(sp4_inplanes, 64, 1)
        self.conv_out_sp2 = BiSeNetOutput(sp2_inplanes, 64, 1)

        self.loss_p = build_loss(loss)
        self.loss_2 = build_loss(loss)
        self.loss_3 = build_loss(loss)
        self.loss_bound_1 = build_loss(bound_loss)
        self.loss_bound_2 = build_loss(bound_loss)
        self.loss_bound_3 = build_loss(bound_loss)
        self.init_weights()

    def init_weights(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def forward(self, inputs):
        feat2, feat4, feat8, feat16, feat32 = inputs
        H, W = feat2.shape[2] * 2, feat2.shape[3] * 2
        H8, W8 = feat8.size()[2:]
        H16, W16 = feat16.size()[2:]
        H32, W32 = feat32.size()[2:]
        avg = F.avg_pool2d(feat32, feat32.size()[2:])

        avg = self.conv_avg(avg)
        avg_up = F.interpolate(avg, (H32, W32), mode='nearest')

        feat32_arm = self.arm32(feat32)
        feat32_sum = feat32_arm + avg_up
        feat32_up = F.interpolate(feat32_sum, (H16, W16), mode='nearest')
        feat32_up = self.conv_head32(feat32_up)

        feat16_arm = self.arm16(feat16)
        feat16_sum = feat16_arm + feat32_up
        feat16_up = F.interpolate(feat16_sum, (H8, W8), mode='nearest')
        feat16_up = self.conv_head16(feat16_up)

        feat_out_sp2 = self.conv_out_sp2(feat2)

        feat_out_sp4 = self.conv_out_sp4(feat4)

        feat_out_sp8 = self.conv_out_sp8(feat8)

        feat_out_sp16 = self.conv_out_sp16(feat16)

        feat_fuse = self.ffm(feat8, feat16_up)

        feat_out = self.conv_out(feat_fuse)
        feat_out16 = self.conv_out16(feat16_up)
        feat_out32 = self.conv_out32(feat32_up)

        feat_out = F.interpolate(feat_out, (H, W), mode='bilinear', align_corners=True)
        feat_out16 = F.interpolate(feat_out16, (H, W), mode='bilinear', align_corners=True)
        feat_out32 = F.interpolate(feat_out32, (H, W), mode='bilinear', align_corners=True)

        if self.use_boundary_2 and self.use_boundary_4 and self.use_boundary_8 and self.training:
            return feat_out, feat_out16, feat_out32, feat_out_sp2, feat_out_sp4, feat_out_sp8  # x8, x16
        else:
            return feat_out

    def losses(self, seg_logit, seg_label):
        loss = dict()
        seg_label = seg_label['gt_masks']
        p1 = self.loss_p(seg_logit[0], seg_label)
        p2 = self.loss_2(seg_logit[1], seg_label)
        p3 = self.loss_3(seg_logit[2], seg_label)
        b1 = self.loss_bound_1(seg_logit[3], seg_label)
        b2 = self.loss_bound_2(seg_logit[4], seg_label)
        b3 = self.loss_bound_3(seg_logit[5], seg_label)
        loss['loss'] = p1 + p2 + p3 + b1 + b2 + b3
        return loss


class AttentionRefinementModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(AttentionRefinementModule, self).__init__()
        self.conv = ConvBNReLU(in_chan, out_chan, ks=3, stride=1, padding=1)
        self.conv_atten = nn.Conv2d(out_chan, out_chan, kernel_size=1, bias=False)
        self.bn_atten = BatchNorm2d(out_chan)
        # self.bn_atten = BatchNorm2d(out_chan, activation='none')

        self.sigmoid_atten = nn.Sigmoid()
        self.init_weight()

    def forward(self, x):
        feat = self.conv(x)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv_atten(atten)
        atten = self.bn_atten(atten)
        atten = self.sigmoid_atten(atten)
        out = torch.mul(feat, atten)
        return out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class ConvBNReLU(nn.Module):
    def __init__(self, in_chan, out_chan, ks=3, stride=1, padding=1, *args, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Conv2d(in_chan,
                              out_chan,
                              kernel_size=ks,
                              stride=stride,
                              padding=padding,
                              bias=False)
        self.bn = BatchNorm2d(out_chan)
        # self.bn = BatchNorm2d(out_chan, activation='none')
        self.relu = nn.ReLU()
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)


class FeatureFusionModule(nn.Module):
    def __init__(self, in_chan, out_chan, *args, **kwargs):
        super(FeatureFusionModule, self).__init__()
        self.convblk = ConvBNReLU(in_chan, out_chan, ks=1, stride=1, padding=0)
        self.conv1 = nn.Conv2d(out_chan,
                               out_chan // 4,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.conv2 = nn.Conv2d(out_chan // 4,
                               out_chan,
                               kernel_size=1,
                               stride=1,
                               padding=0,
                               bias=False)
        self.relu = nn.ReLU(inplace=True)
        self.sigmoid = nn.Sigmoid()
        self.init_weight()

    def forward(self, fsp, fcp):
        fcat = torch.cat([fsp, fcp], dim=1)
        feat = self.convblk(fcat)
        atten = F.avg_pool2d(feat, feat.size()[2:])
        atten = self.conv1(atten)
        atten = self.relu(atten)
        atten = self.conv2(atten)
        atten = self.sigmoid(atten)
        feat_atten = torch.mul(feat, atten)
        feat_out = feat_atten + feat
        return feat_out

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params


class BiSeNetOutput(nn.Module):
    def __init__(self, in_chan, mid_chan, n_classes, *args, **kwargs):
        super(BiSeNetOutput, self).__init__()
        self.conv = ConvBNReLU(in_chan, mid_chan, ks=3, stride=1, padding=1)
        self.conv_out = nn.Conv2d(mid_chan, n_classes, kernel_size=1, bias=False)
        self.init_weight()

    def forward(self, x):
        x = self.conv(x)
        x = self.conv_out(x)
        return x

    def init_weight(self):
        for ly in self.children():
            if isinstance(ly, nn.Conv2d):
                nn.init.kaiming_normal_(ly.weight, a=1)
                if not ly.bias is None: nn.init.constant_(ly.bias, 0)

    def get_params(self):
        wd_params, nowd_params = [], []
        for name, module in self.named_modules():
            if isinstance(module, (nn.Linear, nn.Conv2d)):
                wd_params.append(module.weight)
                if not module.bias is None:
                    nowd_params.append(module.bias)
            elif isinstance(module, BatchNorm2d):
                nowd_params += list(module.parameters())
        return wd_params, nowd_params
