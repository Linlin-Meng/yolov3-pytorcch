from collections import OrderedDict

import torch
from torch import nn

from nets.darknet import darknet53


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


# class SeparableConvBlock(nn.Module):
#     def __init__(self, in_channels, out_channels=None, norm=True, activation=False):
#         super(SeparableConvBlock, self).__init__()
#         if out_channels is None:
#             out_channels = in_channels
#
#         self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels, bias=False)
#         self.pointwise_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
#
#         self.norm = norm
#         if self.norm:
#             self.bn = nn.BatchNorm2d(num_features=out_channels, momentum=0.01, eps=1e-3)
#
#         self.activation = activation
#         if self.activation:
#             self.swish = Swish()
#
#     def forward(self, x):
#         x = self.depthwise_conv(x)
#         x = self.pointwise_conv(x)
#
#         if self.norm:
#             x = self.bn(x)
#
#         if self.activation:
#             x = self.swish(x)
#
#         return x


class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Upsample, self).__init__()

        self.upsample = nn.Sequential(
            conv2d(in_channels, out_channels, 1),
            nn.Upsample(scale_factor=2, mode='nearest')
        )

    def forward(self, x,):
        x = self.upsample(x)
        return x


def conv2d(filter_in, filter_out, kernel_size, stride=1):
    pad = (kernel_size - 1) // 2 if kernel_size else 0
    return nn.Sequential(OrderedDict([
        ("conv", nn.Conv2d(filter_in, filter_out, kernel_size=kernel_size, stride=stride, padding=pad, bias=False)),
        ("bn", nn.BatchNorm2d(filter_out)),
        ("relu", nn.LeakyReLU(0.1)),
    ]))


#------------------------------------------------------------------------#
#   make_last_layers里面一共有七个卷积，前五个用于提取特征。
#   后两个用于获得yolo网络的预测结果
#------------------------------------------------------------------------#
def make_last_layers(in_filters, out_filter):
    m = nn.Sequential(
        # SeparableConvBlock(in_filters, filters_list[0], 1),
        # SeparableConvBlock(filters_list[0], filters_list[1], 3),
        # SeparableConvBlock(filters_list[1], filters_list[0], 1),
        # SeparableConvBlock(filters_list[0], filters_list[1], 3),
        # SeparableConvBlock(filters_list[1], filters_list[0], 1),
        # conv2d(filters_list[0], filters_list[1], 3),
        # nn.Conv2d(filters_list[1], out_filter, kernel_size=1, stride=1, padding=0, bias=True)
        conv2d(in_filters, in_filters, 3),
        nn.Conv2d(in_filters, out_filter, kernel_size=1, stride=1, padding=0, bias=True)
    )
    return m



class YoloBody(nn.Module):
    def __init__(self, anchors_mask, num_classes, pretrained = False, epsilon=1e-4, attention=True):
        super(YoloBody, self).__init__()
        self.backbone = darknet53()
        if pretrained:
            self.backbone.load_state_dict(torch.load("model_data/darknet53_backbone_weights.pth"))

        self.num_channels = len(anchors_mask[0]) * (num_classes + 5)

        self.epsilon = epsilon
        self.swish = Swish()

        self.p3_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p3_w1_relu = nn.ReLU()
        self.p4_w1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p4_w1_relu = nn.ReLU()
        self.p4_w2 = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.p4_w2_relu = nn.ReLU()
        self.p5_w2 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.p5_w2_relu = nn.ReLU()

        self.p3_upsample = Upsample(512, 256)
        self.p4_upsample = Upsample(1024, 512)

        self.p4_downsample = conv2d(256, 512, 3, 2)
        self.p5_downsample = conv2d(512, 1024, 3, 2)

        self.conv4_up = conv2d(512, 512, 3)
        self.conv3_up = conv2d(256, 256, 3)

        self.conv4_down = conv2d(512, 512, 3)
        self.conv5_down = conv2d(1024, 1024, 3)

        # 计算yolo_head的输出通道数
        self.p3_out_0 = make_last_layers(256, self.num_channels)
        self.p4_out_0 = make_last_layers(512, self.num_channels)
        self.p5_out_0 = make_last_layers(1024, self.num_channels)

    def forward(self, x):
        p3_in, p4_in, p5_in = self.backbone(x)

        # 简单的注意力机制，用于确定更关注p5_in还是p4_in
        p4_w1 = self.p4_w1_relu(self.p4_w1)
        weight = p4_w1 / (torch.sum(p4_w1, dim=0) + self.epsilon)
        p4_td = self.conv4_up(self.swish(weight[0] * p4_in + weight[1] * self.p4_upsample(p5_in)))

        # 简单的注意力机制，用于确定更关注p3_in还是p4_td
        p3_w1 = self.p3_w1_relu(self.p3_w1)
        weight = p3_w1 / (torch.sum(p3_w1, dim=0) + self.epsilon)
        p3_out = self.conv3_up(self.swish(weight[0] * p3_in + weight[1] * self.p3_upsample(p4_td)))

        # 简单的注意力机制，用于确定更关注p4_in还是p4_td还是p3_out
        p4_w2 = self.p4_w2_relu(self.p4_w2)
        weight = p4_w2 / (torch.sum(p4_w2, dim=0) + self.epsilon)
        p4_out = self.conv4_down(
            self.swish(weight[0] * p4_in + weight[1] * p4_td + weight[2] * self.p4_downsample(p3_out)))

        # 简单的注意力机制，用于确定更关注p5_in还是p4_out
        p5_w2 = self.p5_w2_relu(self.p5_w2)
        weight = p5_w2 / (torch.sum(p5_w2, dim=0) + self.epsilon)
        p5_out = self.conv5_down(self.swish(weight[0] * p5_in + weight[1] * self.p5_downsample(p4_out)))

        p3_out = self.p3_out_0(p3_out)
        p4_out = self.p4_out_0(p4_out)
        p5_out = self.p5_out_0(p5_out)

        return p5_out, p4_out, p3_out
