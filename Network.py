import math
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from colorconvert import *
from block import *
from canny.canny import Canny

def gram_matrix(y):
    """ Returns the gram matrix of y (used to compute style loss) """
    (b, c, h, w) = y.size()
    features = y.view(b, c, w * h)
    features_t = features.transpose(1, 2)   #C和w*h转置
    gram = features.bmm(features_t) / (c * h * w)   #bmm 将features与features_t相乘
    return gram

##########################################################################
## Encoder
class Encoder(nn.Module):
    def __init__(self, channel):
        super(Encoder,self).__init__()

        self.canny = Canny()
        self.conv1 = ConvLayer(4,channel,1)
        in_channel = channel

        ch1 = channel * (2 ** 1)
        self.res11 = ResBlock(in_channel, ch1, downsample=True, padding="reflect")
        #self.res12 = ResBlock(ch1, ch1, downsample=False, padding="reflect")
        in_channel = ch1

        ch2 = channel * (2 ** 2)
        self.res21 = ResBlock(in_channel, ch2, downsample=True, padding="reflect")
        #self.res22 = ResBlock(ch2, ch2, downsample=False, padding="reflect")
        in_channel = ch2

        ch3 = channel * (2 ** 3)
        self.res31 = ResBlock(in_channel, ch3, downsample=True, padding="reflect")
        #self.res32 = ResBlock(ch3, ch3, downsample=False, padding="reflect")
        in_channel = ch3

        ch4 = channel * (2 ** 4)
        self.res41 = ResBlock(in_channel, ch4, downsample=True, padding="reflect")
        #self.res42 = ResBlock(ch4, ch4, downsample=False, padding="reflect")
        in_channel = ch4

        self.conv2 = torch.nn.Conv2d(in_channel, channel, (1, 1))

        self.activate_f = torch.nn.Tanh()
        #self.bn = torch.nn.BatchNorm2d(channel)
        #self.conv2 = ConvLayer(in_channel, channel, 1)

    def forward(self, input):
        _, edge = self.canny(input)
        out0 = self.conv1(torch.cat([input, edge], dim=1))
        # out0 = self.conv1(input)
        # out1 = self.res12(self.activate_f(self.res11(out0)))
        # out2 = self.res22(self.activate_f(self.res21(out1)))
        # out3 = self.res32(self.activate_f(self.res31(out2)))
        # out4 = self.res42(self.activate_f(self.res41(out3)))
        out1 = self.activate_f(self.res11(out0))
        out2 = self.activate_f(self.res21(out1))
        out3 = self.activate_f(self.res31(out2))
        out4 = self.activate_f(self.res41(out3))
        outslide = []
        outslide.append(out1)
        outslide.append(out2)
        outslide.append(out3)
        outslide.append(out4)
        out_orthogonality = self.conv2(out4)
        out_orthogonality = (out_orthogonality - out_orthogonality.min()) / (
                    out_orthogonality.max() - out_orthogonality.min() + 1e-5)
        out_gram = gram_matrix(out_orthogonality)
        b, c, h, w, = out_orthogonality.size()
        out_orthogonality = out_orthogonality.view(b, -1)
        out_gram = out_gram.view(b, -1)
        return out4,outslide,out_orthogonality,out_gram


##########################################################################
## Generator
class Generator(nn.Module):
    def __init__(self, channel, blur_kernel=(1, 3, 3, 1)):
        super(Generator, self).__init__()
        self.feature_num = 4
        self.layers_conv1 = nn.ModuleList()
        self.layers_up = nn.ModuleList()
        self.NLB = nn.ModuleList()
        for i in range(self.feature_num):
            self.NLB.append(MSCNL(channel * (2 ** (self.feature_num - i)), channel, self.feature_num, channel * 2, self.feature_num -1 - i))
            self.layers_up.append(
                ConvLayer(
                    channel * (2 ** (self.feature_num - i)),
                    channel * (2 ** (self.feature_num - 1 - i)),
                    1,
                    upsample=True,
                    blur_kernel=blur_kernel,
                    bias=False,
                    activate=False,
                )
            )
        for i in range(self.feature_num - 1):
            self.layers_conv1.append(
                ConvLayer(
                    channel * (2 ** (self.feature_num - i)),
                    channel * (2 ** (self.feature_num - 1 - i)),
                    3,
                    upsample=False,
                    blur_kernel=blur_kernel,
                    bias=False,
                    activate=True,
                )
            )
        self.layers_conv1.append(ConvLayer(channel , channel, 3, activate=True))
        self.to_rgb = ConvLayer(channel , 3, 1, activate=False)

        self.activate_f = torch.nn.Tanh()

    def forward(self, input_gray, input_color):
        decoder_out = []
        feature = input_gray[3]
        for i in range(self.feature_num):
            NL = self.NLB[i]
            up = self.layers_up[i]
            conv1 = self.layers_conv1[i]
            feature = up(NL(input_gray, input_color, feature, input_color[3 - i]))
            if i < self.feature_num - 1:
                feature = torch.cat([feature,input_gray[2-i]],1)
            feature = conv1(self.activate_f(feature))
            decoder_out.append(feature)
        image_out = self.to_rgb(feature)
        return image_out,decoder_out


##########################################################################
## Original Colorization Network(OCNet)
class OCNet(nn.Module):
    def __init__(self, channel):
        super(OCNet, self).__init__()

        kernel_size = 3
        reduction = 4
        bias = False
        num_cab = 2
        act = nn.PReLU()

        self.conv0 = nn.Conv2d(6, channel, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv1 = CAB(channel, kernel_size, reduction, bias=bias, act=act)
        self.conv21 = nn.Conv2d(channel * 3, channel, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv31 = nn.Conv2d(channel * 3, channel, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv41 = nn.Conv2d(channel * 3, channel, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)
        self.conv51 = nn.Conv2d(channel * 3, channel, kernel_size, padding=(kernel_size // 2), bias=bias, stride=1)

        self.up1 = nn.Conv2d(channel * 2 , channel * 2, kernel_size=1, bias=bias)
        self.up2 = UpSample(channel * 4, channel * 2)
        self.up3 = nn.Sequential(UpSample(channel * 8, channel * 4), UpSample(channel * 4, channel * 2))
        self.up4 = nn.Sequential(UpSample(channel * 16, channel * 8), UpSample(channel * 8, channel * 4),
                                 UpSample(channel * 4, channel * 2))

        self.ocb1 = OCB(channel, kernel_size, reduction, act, bias, num_cab)
        self.ocb2 = OCB(channel, kernel_size, reduction, act, bias, num_cab)
        self.ocb3 = OCB(channel, kernel_size, reduction, act, bias, num_cab)
        self.ocb4 = OCB(channel, kernel_size, reduction, act, bias, num_cab)
        self.ocb5 = OCB(channel, kernel_size, reduction, act, bias, num_cab)

        self.to_rgb = ConvLayer(channel, 3, 1, activate=False)

    def forward(self, x1, x2, decoder_out):
        x = self.conv1(self.conv0(torch.cat([x1, x2], 1)))

        x = self.ocb1(x) + x
        d_feature1 = self.up1(decoder_out[3])
        x = torch.cat([x, d_feature1], 1)

        x = self.conv21(x)
        x = self.ocb2(x) + x
        d_feature2 = self.up2(decoder_out[2])
        x = torch.cat([x, d_feature2], 1)

        x = self.conv31(x)
        x = self.ocb3(x) + x
        d_feature3 = self.up3(decoder_out[1])
        x = torch.cat([x, d_feature3], 1)

        x = self.conv41(x)
        x = self.ocb4(x) + x
        d_feature4 = self.up4(decoder_out[0])
        x = torch.cat([x, d_feature4], 1)

        x = self.conv51(x)
        x = self.ocb5(x) + x
        x = self.to_rgb(x) + x1

        return x



##########################################################################
## Discriminator
class Discriminator(nn.Module):
    def __init__(self, size, channel_multiplier=1, blur_kernel=(1, 3, 3, 1)):
        super(Discriminator,self).__init__()

        channels = {
            4: 512,
            8: 512,
            16: 512,
            32: 512,
            64: 256 * channel_multiplier,
            128: 128 * channel_multiplier,
            256: 64 * channel_multiplier,
            512: 32 * channel_multiplier,
            1024: 16 * channel_multiplier,
        }

        convs = [ConvLayer(3, channels[size], 1)]

        log_size = int(math.log(size, 2))

        in_channel = channels[size]

        for i in range(log_size, 2, -1):
            out_channel = channels[2 ** (i - 1)]

            convs.append(ResBlock(in_channel, out_channel, downsample=True))

            in_channel = out_channel

        self.convs = nn.Sequential(*convs)

        self.final_conv = ConvLayer(in_channel, channels[4], 3)
        self.final_linear = nn.Sequential(
            EqualLinear(channels[4] * 4 * 4, channels[4], activation="fused_lrelu"),
            EqualLinear(channels[4], 1),
        )

    def forward(self, input):
        out = self.convs(input)
        out = self.final_conv(out)

        out = out.view(out.shape[0], -1)
        out = self.final_linear(out)

        return out


##########################################################################
## Cooccur Discriminator
class CooccurDiscriminator(nn.Module):
    def __init__(self, channel, size=256):
        super(CooccurDiscriminator,self).__init__()

        encoder = [ConvLayer(3, channel, 1)]

        ch_multiplier = (2, 4, 8, 12, 12, 24)
        downsample = (True, True, True, True, True, False)
        in_ch = channel
        for ch_mul, down in zip(ch_multiplier, downsample):
            encoder.append(ResBlock(in_ch, channel * ch_mul, down))
            in_ch = channel * ch_mul

        if size > 511:
            k_size = 3
            feat_size = 2 * 2

        else:
            k_size = 2
            feat_size = 1 * 1

        encoder.append(ConvLayer(in_ch, channel * 12, k_size, padding="valid"))

        self.encoder = nn.Sequential(*encoder)

        self.linear = nn.Sequential(
            EqualLinear(
                channel * 12 * 2 * feat_size, channel * 32, activation="fused_lrelu"
            ),
            EqualLinear(channel * 32, channel * 32, activation="fused_lrelu"),
            EqualLinear(channel * 32, channel * 16, activation="fused_lrelu"),
            EqualLinear(channel * 16, 1),
        )

    def forward(self, input, reference=None, ref_batch=None, ref_input=None):
        # print(input.shape)
        out_input = self.encoder(input)

        if ref_input is None:
            ref_input = self.encoder(reference)
            _, channel, height, width = ref_input.shape
            ref_input = ref_input.view(-1, ref_batch, channel, height, width)
            ref_input = ref_input.mean(1)

        out = torch.cat((out_input, ref_input), 1)
        out = torch.flatten(out, 1)
        out = self.linear(out)

        return out, ref_input


##########################################################################
## Multi Scale Discriminator
class MultiScaleDiscriminator(nn.Module):
    def __init__(self, real_crop_size=256, max_n_scales=9, scale_factor=2, base_channels=64, extra_conv_layers=0):
        super(MultiScaleDiscriminator, self).__init__()
        self.base_channels = base_channels
        self.scale_factor = scale_factor
        self.min_size = 16
        self.extra_conv_layers = extra_conv_layers

        # We want the max num of scales to fit the size of the real examples. further scaling would create networks that
        # only train on fake examples
        self.max_n_scales = np.min([np.int(np.ceil(np.log(np.min(real_crop_size) * 1.0 / self.min_size)
                                                   / np.log(self.scale_factor))), max_n_scales])

        # Prepare a list of all the networks for all the wanted scales
        self.nets = nn.ModuleList()

        # Create a network for each scale
        for _ in range(self.max_n_scales):
            self.nets.append(self.make_net())

    def make_net(self):
        base_channels = self.base_channels
        net = []

        # Entry block
        net += [nn.utils.spectral_norm(nn.Conv2d(3, base_channels, kernel_size=3, stride=1)),
                nn.BatchNorm2d(base_channels),
                nn.LeakyReLU(0.2, True)]

        # Downscaling blocks
        # A sequence of strided conv-blocks. Image dims shrink by 2, channels dim expands by 2 at each block
        net += [nn.utils.spectral_norm(nn.Conv2d(base_channels, base_channels * 2, kernel_size=3, stride=2)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Regular conv-block
        net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
                                                 out_channels=base_channels * 2,
                                                 kernel_size=3,
                                                 bias=True)),
                nn.BatchNorm2d(base_channels * 2),
                nn.LeakyReLU(0.2, True)]

        # Additional 1x1 conv-blocks
        for _ in range(self.extra_conv_layers):
            net += [nn.utils.spectral_norm(nn.Conv2d(in_channels=base_channels * 2,
                                                     out_channels=base_channels * 2,
                                                     kernel_size=3,
                                                     bias=True)),
                    nn.BatchNorm2d(base_channels * 2),
                    nn.LeakyReLU(0.2, True)]

        # Final conv-block
        # Ends with a Sigmoid to get a range of 0-1
        net += nn.Sequential(nn.utils.spectral_norm(nn.Conv2d(base_channels * 2, 1, kernel_size=1)),
                             nn.Sigmoid())

        # Make it a valid layers sequence and return
        return nn.Sequential(*net)

    def forward(self, input_tensor, scale_weights):
        aggregated_result_maps_from_all_scales = self.nets[0](input_tensor) * scale_weights[0]
        map_size = aggregated_result_maps_from_all_scales.shape[2:]

        # Run all nets over all scales and aggregate the interpolated results
        for net, scale_weight, i in zip(self.nets[1:], scale_weights[1:], range(1, len(scale_weights))):
            downscaled_image = F.interpolate(input_tensor, scale_factor=self.scale_factor**(-i), mode='bilinear')
            result_map_for_current_scale = net(downscaled_image)
            upscaled_result_map_for_current_scale = F.interpolate(result_map_for_current_scale,
                                                                  size=map_size,
                                                                  mode='bilinear')
            aggregated_result_maps_from_all_scales += upscaled_result_map_for_current_scale * scale_weight

        return aggregated_result_maps_from_all_scales

class PDNLAnet(nn.Module):
    def __init__(self, channel):
        super(PDNLAnet, self).__init__()
        self.enc1 = Encoder(channel*2)
        self.enc2 = Encoder(channel*2)
        self.gen = Generator(channel*2)
        self.ocn = OCNet(channel)

    def forward(self, input1, input2):
        content, content_slide, content_orth, content_gram = self.enc1(input1)
        style, style_slide, style_orth, style_gram = self.enc2(input2)
        fake_image1, decoder_out = self.gen(content_slide, style_slide)
        fake_image2 = self.ocn(input1, fake_image1, decoder_out)
        return fake_image2