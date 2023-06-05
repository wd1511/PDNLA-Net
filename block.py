import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from termcolor import cprint

from stylegan2.model import StyledConv, Blur, EqualLinear, EqualConv2d, ScaledLeakyReLU
from stylegan2.op import FusedLeakyReLU
from colorconvert import *


##########################################################################
## Channel Attention Block (CAB)
class CAB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, bias, act):
        super(CAB, self).__init__()
        modules_body = []
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1))
        modules_body.append(act)
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1))

        self.body = nn.Sequential(*modules_body)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # feature channel downscale and upscale --> channel weight
        self.conv_du = nn.Sequential(
            nn.Conv2d(n_feat, n_feat // reduction, 1, padding=0, bias=bias),
            nn.ReLU(inplace=True),
            nn.Conv2d(n_feat // reduction, n_feat, 1, padding=0, bias=bias),
            nn.Sigmoid()
        )

    def forward(self, x):
        res = self.body(x)
        y = self.avg_pool(res)
        y = self.conv_du(y)
        res = res * y
        res += x
        return res


##########################################################################
## Equal Conv Transpose (ECT)
class EqualConvTranspose2d(nn.Module):
    def __init__(
        self, in_channel, out_channel, kernel_size, stride=1, padding=0, bias=True
    ):
        super(EqualConvTranspose2d,self).__init__()

        self.weight = nn.Parameter(
            torch.randn(in_channel, out_channel, kernel_size, kernel_size)
        )
        self.scale = 1 / math.sqrt(in_channel * kernel_size ** 2)

        self.stride = stride
        self.padding = padding

        if bias:
            self.bias = nn.Parameter(torch.zeros(out_channel))

        else:
            self.bias = None

    def forward(self, input):
        out = F.conv_transpose2d(
            input,
            self.weight * self.scale,
            bias=self.bias,
            stride=self.stride,
            padding=self.padding,
        )

        return out

    def __repr__(self):
        return (
            f"{self.__class__.__name__}({self.weight.shape[0]}, {self.weight.shape[1]},"
            f" {self.weight.shape[2]}, stride={self.stride}, padding={self.padding})"
        )


##########################################################################
## pyramidAP (PAP)
class pyramidAP(nn.Module):
    def __init__(self):
        super(pyramidAP,self).__init__()
        self.AP1 = nn.AdaptiveAvgPool2d(1)
        self.AP3 = nn.AdaptiveAvgPool2d(3)
        self.AP6 = nn.AdaptiveAvgPool2d(6)
        self.AP8 = nn.AdaptiveAvgPool2d(8)

    def forward(self, input):
        b, c, h, w = input.size()
        x1 = self.AP1(input).view(b,c,-1)
        x3 = self.AP3(input).view(b,c,-1)
        x6 = self.AP6(input).view(b,c,-1)
        x8 = self.AP8(input).view(b,c,-1)
        output = torch.cat([x1,x3,x6,x8],2)
        return output


##########################################################################
## pyramidAP_inverse (PAP_inv)
class pyramidAP_inverse(nn.Module):
    def __init__(self):
        super(pyramidAP_inverse,self).__init__()
        self.AP8 = nn.AdaptiveAvgPool2d(8)

    def forward(self, input, feature_num, h, w):
        b, c, _ = input.size()
        input = input.view(b, c, feature_num, -1).mean(dim=2)
        x1 = input[:, :, 0].view(b, c, 1, 1)
        x2 = input[:, :, 1:10].view(b, c, 3, 3)
        x3 = input[:, :, 10:46].view(b, c, 6, 6)
        x4 = input[:, :, 46:110].view(b, c, 8, 8)
        output = self.AP8(x1) + self.AP8(x2) + self.AP8(x3) + self.AP8(x4)
        output = output / 4
        output = torch.nn.functional.interpolate(output,size=(h,w),mode='bicubic')
        return output


##########################################################################
## Channel Attention (CA)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1   = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2   = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


##########################################################################
## Spatial Attention (SA)
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


##########################################################################
## Up Sample
class UpSample(nn.Module):
    def __init__(self, in_channels,out_channels):
        super(UpSample, self).__init__()
        self.up = nn.Sequential(nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
                                nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False))

    def forward(self, x):
        x = self.up(x)
        return x


##########################################################################
## Conv Layer (CL)
class ConvLayer(nn.Sequential):
    def __init__(
        self,
        in_channel,
        out_channel,
        kernel_size,
        upsample=False,
        downsample=False,
        blur_kernel=(1, 3, 3, 1),
        bias=True,
        activate=True,
        padding="zero",
    ):
        layers = []

        self.padding = 0
        stride = 1

        if downsample:
            factor = 2
            p = (len(blur_kernel) - factor) + (kernel_size - 1)
            pad0 = (p + 1) // 2
            pad1 = p // 2

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

            stride = 2

        if upsample:
            layers.append(
                EqualConvTranspose2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=0,
                    stride=2,
                    bias=bias and not activate,
                )
            )

            factor = 2
            p = (len(blur_kernel) - factor) - (kernel_size - 1)
            pad0 = (p + 1) // 2 + factor - 1
            pad1 = p // 2 + 1

            layers.append(Blur(blur_kernel, pad=(pad0, pad1)))

        else:
            if not downsample:
                if padding == "zero":
                    self.padding = (kernel_size - 1) // 2

                elif padding == "reflect":
                    padding = (kernel_size - 1) // 2

                    if padding > 0:
                        layers.append(nn.ReflectionPad2d(padding))

                    self.padding = 0

                elif padding != "valid":
                    raise ValueError('Padding should be "zero", "reflect", or "valid"')

            layers.append(
                EqualConv2d(
                    in_channel,
                    out_channel,
                    kernel_size,
                    padding=self.padding,
                    stride=stride,
                    bias=bias and not activate,
                )
            )

        if activate:
            if bias:
                layers.append(FusedLeakyReLU(out_channel))

            else:
                layers.append(ScaledLeakyReLU(0.2))

        super(ConvLayer,self).__init__(*layers)


##########################################################################
## Residual Block (RB)
class ResBlock(nn.Module):
    def __init__(
        self,
        in_channel,
        out_channel,
        downsample,
        padding="zero",
        blur_kernel=(1, 3, 3, 1),
    ):
        super(ResBlock,self).__init__()

        self.conv1 = ConvLayer(in_channel, out_channel, 3, padding=padding)

        self.conv2 = ConvLayer(
            out_channel,
            out_channel,
            3,
            downsample=downsample,
            padding=padding,
            blur_kernel=blur_kernel,
        )

        if downsample or in_channel != out_channel:
            self.skip = ConvLayer(
                in_channel,
                out_channel,
                1,
                downsample=downsample,
                blur_kernel=blur_kernel,
                bias=False,
                activate=False,
            )

        else:
            self.skip = None

    def forward(self, input):
        out = self.conv1(input)
        out = self.conv2(out)

        if self.skip is not None:
            skip = self.skip(input)

        else:
            skip = input

        # print(out.shape)

        return (out + skip) / math.sqrt(2)


##########################################################################
## base non-local(BNL)
class BNL(nn.Module):
    """Spatial CGNL block with Gaussian RBF kernel for image classification.
    """
    def __init__(self, inplanes1, inplanes2, planes, use_scale=False, groups=None, order=2):
        self.use_scale = use_scale
        self.groups = groups
        self.order = order

        super(BNL, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes1, planes, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes2, planes, kernel_size=1, stride=1, bias=False)
        # conv g
        self.g = nn.Conv2d(inplanes2, planes, kernel_size=1, stride=1, bias=False)
        # conv z
        self.z = nn.Conv2d(planes, inplanes1, kernel_size=1, stride=1,
                                                  groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes1)

    def kernel(self, t, p, g, b1, c1, h1, w1, c2, h2, w2):

        t = t.view(b1, 1, c1 * h1 * w1)
        p = p.view(b1, 1, c2 * h2 * w2)
        g = g.view(b1, c2 * h2 * w2, 1)

        gamma = torch.Tensor(1).fill_(1e-4)
        beta = torch.exp(-2 * gamma)

        t_taylor = []
        p_taylor = []
        for order in range(self.order+1):
            alpha = torch.mul(
                    torch.div(
                    torch.pow(
                        (2 * gamma),
                        order),
                        math.factorial(order)),
                        beta)

            alpha = torch.sqrt(
                        alpha.cuda())

            _t = t.pow(order).mul(alpha)
            _p = p.pow(order).mul(alpha)

            t_taylor.append(_t)
            p_taylor.append(_p)

        t_taylor = torch.cat(t_taylor, dim=1)
        p_taylor = torch.cat(p_taylor, dim=1)

        att = torch.bmm(p_taylor, g)

        if self.use_scale:
            att = att.div((c2*h2*w2)**0.5)

        att = att.view(b1, 1, int(self.order+1))
        x = torch.bmm(att, t_taylor)
        x = x.view(b1, c1, h1, w1)

        return x

    def forward(self, input_high, input_low):
        residual = input_high

        t = self.t(input_high)
        p = self.p(input_low)
        g = self.g(input_low)

        b1, c1, h1, w1 = t.size()
        b2, c2, h2, w2 = p.size()

        if self.groups and self.groups > 1:
            _c1 = int(c1 / self.groups)
            _c2 = int(c2 / self.groups)

            ts = torch.split(t, split_size_or_sections=_c1, dim=1)
            ps = torch.split(p, split_size_or_sections=_c2, dim=1)
            gs = torch.split(g, split_size_or_sections=_c2, dim=1)

            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i],
                                 b1, _c1, h1, w1, _c2, h2, w2)
                _t_sequences.append(_x)

            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g,
                            b1, c1, h1, w1, c2, h2, w2)

        x = self.z(x)
        x = self.gn(x) + residual

        return x


def calc_mean_std(feat, eps=1e-5):
    # eps is a small value added to the variance to avoid divide-by-zero.
    size = feat.size()
    assert (len(size) == 4)
    N, C = size[:2]
    feat_var = feat.view(N, C, -1).var(dim=2) + eps
    feat_std = feat_var.sqrt().view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def mean_variance_norm(feat):
    size = feat.size()
    mean, std = calc_mean_std(feat)
    normalized_feat = (feat - mean.expand(size)) / std.expand(size)
    return normalized_feat


def get_feature_key(feature, last_id):
    results = []
    _, _, h, w = feature[last_id].shape
    for i in range(last_id):
        results.append(mean_variance_norm(nn.functional.interpolate(feature[i], (h, w))))
    results.append(mean_variance_norm(feature[last_id]))
    return torch.cat(results, dim=1)

##########################################################################
## multi-scale Cross non-loacl(MSNL)
class MSCNL(nn.Module):
    def __init__(self, channel_in, channel_min, feature_num, inter_channel, last_id):
        super(MSCNL, self).__init__()

        self.inter_channel = inter_channel
        self.feature_num = feature_num
        self.channel_in = channel_in
        self.last_id = last_id
        self.key_channel = 0

        self.conv_k1_list = nn.ModuleList()
        self.conv_v1_list = nn.ModuleList()
        self.conv_k2_list = nn.ModuleList()
        self.conv_v2_list = nn.ModuleList()
        self.conv_q4_list = nn.ModuleList()
        self.conv_k4_list = nn.ModuleList()

        in_ch = channel_min

        for i in range(self.feature_num):
            in_ch = in_ch * 2
            self.conv_k1_list.append(
                nn.Conv2d(in_channels=in_ch, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=1,
                          bias=False)
            )
            self.conv_v1_list.append(
                nn.Conv2d(in_channels=in_ch, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=1,
                          bias=False)
            )
            self.conv_k2_list.append(
                nn.Conv2d(in_channels=in_ch, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=1,
                          bias=False)
            )
            self.conv_v2_list.append(
                nn.Conv2d(in_channels=in_ch, out_channels=self.inter_channel, kernel_size=1, stride=1, padding=1,
                          bias=False)
            )

        for i in range(self.last_id + 1):
            self.key_channel = self.key_channel + channel_min * (2 ** (i + 1))

        self.conv_q1 = nn.Conv2d(in_channels=self.channel_in, out_channels=self.inter_channel, kernel_size=3, stride=1,
                                 padding=1, bias=False)
        self.conv_q2 = nn.Conv2d(in_channels=self.channel_in, out_channels=self.inter_channel, kernel_size=3, stride=1,
                                 padding=1, bias=False)

        self.PAP = pyramidAP()
        self.softmax = nn.Softmax(dim=1)

        self.conv_mask1 = nn.Conv2d(in_channels=self.inter_channel + self.channel_in, out_channels=self.channel_in, kernel_size=1,
                                    stride=1, padding=0, bias=False)
        self.conv_mask2 = nn.Conv2d(in_channels=self.inter_channel + self.channel_in, out_channels=self.channel_in, kernel_size=1,
                                    stride=1, padding=0, bias=False)

        self.conv_fuse11 = nn.Conv2d(in_channels=self.channel_in * self.feature_num, out_channels=self.channel_in,
                                    kernel_size=1,stride=1, padding=0, bias=False)
        self.conv_fuse21 = nn.Conv2d(in_channels=self.channel_in * self.feature_num, out_channels=self.channel_in,
                                    kernel_size=1, stride=1, padding=0, bias=False)
        #self.conv_fuse3 = nn.Conv2d(in_channels=self.channel_in * (self.feature_num + 1), out_channels=self.channel_in,
                                    #kernel_size=1, stride=1, padding=0, bias=False)

        self.BNLBlock = BNL(self.channel_in, self.channel_in, channel_min * 2, use_scale=False, groups=8)

        self.activate_f = torch.nn.Tanh()

        #self.ca = ChannelAttention(self.channel_in)
        #self.sa = SpatialAttention()

    def forward(self, input_fg, input_fc, input_g, input_c):
        b, c, h, w, = input_g.size()

        x_gray_list = []
        x_color_list = []
        x_q4_list = []
        x_k4_list = []

        # C' * N
        x_q1 = self.conv_q1(input_g).view(b, self.inter_channel, -1)
        x_q2 = self.conv_q2(input_c).view(b, self.inter_channel, -1)
        for i in range(self.feature_num):
            conv_k1 = self.conv_k1_list[i]
            conv_v1 = self.conv_v1_list[i]
            conv_k2 = self.conv_k2_list[i]
            conv_v2 = self.conv_v2_list[i]

            # S * C'
            x_k1 = self.PAP(conv_k1(input_fc[i])).permute(0, 2, 1).contiguous()
            x_k2 = self.PAP(conv_k2(input_fg[i])).permute(0, 2, 1).contiguous()
            # C' * S
            x_v1 = self.PAP(conv_v1(input_fc[i]))
            x_v2 = self.PAP(conv_v2(input_fg[i]))

            x_kq1 = self.softmax(torch.matmul(x_k1, x_q1).clip_(-64, 64))
            x_kq2 = self.softmax(torch.matmul(x_k2, x_q2).clip_(-64, 64))
            x_vkq1 = torch.matmul(x_v1, x_kq1).view(b, self.inter_channel, h, w)
            x_vkq2 = torch.matmul(x_v2, x_kq2).view(b, self.inter_channel, h, w)
            x_gray =  self.conv_mask1(torch.cat([input_g, x_vkq1], 1))
            x_color = self.conv_mask2(torch.cat([input_c, x_vkq2], 1))
            #x_gray = input_g + self.conv_mask1(torch.cat([input_g, x_vkq1], 1))
            #x_color = input_c + self.conv_mask2(torch.cat([input_c, x_vkq2], 1))

            x_gray_list.append(x_gray)
            x_color_list.append(x_color)

        x_gray = self.activate_f(self.conv_fuse11(torch.cat(x_gray_list, 1)))+ input_g
        x_color = self.activate_f(self.conv_fuse21(torch.cat(x_color_list, 1))) + input_c

        x_fuse = self.BNLBlock(x_gray, x_color) + input_g
        #x_gray_list.append(x_fuse)
        #x_fuse = self.conv_fuse3(torch.cat(x_gray_list, 1))
        #x_fuse = x_fuse + input_g
        #x_fuse = x_fuse * self.ca(input_g) * self.sa(input_g) + x_fuse
        return x_fuse



##########################################################################
## Original Colorization Block (OCB)
class OCB(nn.Module):
    def __init__(self, n_feat, kernel_size, reduction, act, bias, num_cab):
        super(OCB, self).__init__()
        modules_body = []
        modules_body = [CAB(n_feat, kernel_size, reduction, bias=bias, act=act) for _ in range(num_cab)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size, padding=(kernel_size // 2), bias=False, stride=1))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


