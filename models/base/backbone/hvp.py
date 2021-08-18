import torch.nn as nn
import torch





class HVP(nn.Module):
    def __init__(self, P1=1, P2=1, P3=3, P4=5, reduction=4, pretrained=None):
        super(HVP, self).__init__()
        self.level1_0 = DownsamplerBlockConv(3, 16)
        self.level1 = nn.ModuleList()
        for i in range(0, P1):
            self.level1.append(ConvBlock(16, 16))
        self.level1.append(residual_att(16, reduction=reduction))
        self.branch1 = nn.Conv2d(16, 16, 1, stride=1, padding=0,bias=False)
        self.br1 = nn.Sequential(nn.BatchNorm2d(16), nn.PReLU(16))

        self.level2_0 = DownsamplerBlockDepthwiseConv(16, 32)
        self.level2 = nn.ModuleList()
        for i in range(0, P2):
            self.level2.append(nn.Dropout2d(0.1, True))
            self.level2.append(gycblock(32, 32))
            self.level2.append(residual_att(32, reduction=reduction))
        self.branch2 = nn.Conv2d(32, 32, 1, stride=1, padding=0,bias=False)
        self.br2 = nn.Sequential(nn.BatchNorm2d(32), nn.PReLU(32))

        self.level3_0 = DownsamplerBlockDepthwiseConv(32, 64)
        self.level3 = nn.ModuleList()
        for i in range(0, P3):
            self.level3.append(nn.Dropout2d(0.1, True))
            self.level3.append(gycblock(64, 64))
            self.level3.append(residual_att(64, reduction=reduction))
        self.branch3 = nn.Conv2d(64, 64, 1, stride=1, padding=0,bias=False)
        self.br3 = nn.Sequential(nn.BatchNorm2d(64), nn.PReLU(64))

        self.level4_0 = DownsamplerBlockDepthwiseConv(64, 128)
        self.level4 = nn.ModuleList()
        for i in range(0, P4):
            self.level4.append(nn.Dropout2d(0.1, True))
            self.level4.append(gycblock(128, 128))
            self.level4.append(residual_att(128, reduction=reduction))
        self.branch4 = nn.Conv2d(128, 128, 1, stride=1, padding=0,bias=False)
        self.br4 = nn.Sequential(nn.BatchNorm2d(128), nn.PReLU(128))

        if pretrained is not None:
            self.load_state_dict(torch.load(pretrained))
            print('Pretrained Model Loaded!')

    def forward(self, input):
        output1_0 = self.level1_0(input)

        output1 = output1_0
        for layer in self.level1:
            output1 = layer(output1)
        output1 = self.br1(self.branch1(output1_0) + output1)

        output2_0 = self.level2_0(output1)
        output2 = output2_0
        for layer in self.level2:
            output2 = layer(output2)
        output2 = self.br2(self.branch2(output2_0) + output2)

        output3_0 = self.level3_0(output2)
        output3 = output3_0
        for layer in self.level3:
            output3 = layer(output3)
        output3 = self.br3(self.branch3(output3_0) + output3)

        output4_0 = self.level4_0(output3)
        output4 = output4_0
        for layer in self.level4:
            output4 = layer(output4)
        output4 = self.br4(self.branch4(output4_0) + output4)

        return output1, output2, output3, output4


class DownsamplerBlockConv(nn.Module):
    def __init__(self, nIn, nOut):
        super(DownsamplerBlockConv, self).__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            self.conv = nn.Conv2d(nIn, nOut-nIn, 3, stride=2, padding=1, bias=False)
            #self.pool = nn.MaxPool2d(2, stride=2)
            self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        else:
            self.conv = nn.Conv2d(nIn, nOut, 3, stride=2, padding=1, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        if self.nIn < self.nOut:
            output = torch.cat([self.conv(input), self.pool(input)], 1)
        else:
            output = self.conv(input)

        output = self.bn(output)
        output = self.act(output)

        return output


class ConvBlock(nn.Module):
    def __init__(self, nIn, nOut, add=True):
        super(ConvBlock, self).__init__()
        self.conv = nn.Conv2d(nIn, nOut, 3, stride=1, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)
        self.add = add

    def forward(self, input):
        output = self.conv(input)
        if self.add:
            output = input + output

        output = self.bn(output)
        output = self.act(output)

        return output


class gycblock(nn.Module):
    def __init__(self,channels_in,channels_out):
        super(gycblock, self).__init__()
        self.recp7 = nn.Sequential(
            BasicConv(channels_in*1, channels_in, kernel_size=7, dilation=1, padding=3, groups=channels_in, bias=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False),

            BasicConv(channels_in, channels_in, kernel_size=3, dilation=7, padding=7, groups=channels_in, bias=False,
                      relu=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False)
        )

        self.recp5 = nn.Sequential(
            BasicConv(channels_in*2, channels_in, kernel_size=5, dilation=1, padding=2, groups=channels_in, bias=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False),

            BasicConv(channels_in, channels_in, kernel_size=3, dilation=5, padding=5, groups=channels_in, bias=False,
                      relu=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False)
        )

        self.recp3 = nn.Sequential(
            BasicConv(channels_in*3, channels_in, kernel_size=3, dilation=1, padding=1, groups=channels_in, bias=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False),

            BasicConv(channels_in, channels_in, kernel_size=3, dilation=3, padding=3, groups=channels_in, bias=False,
                      relu=False),
            BasicConv(channels_in, channels_in, kernel_size=1, dilation=1, bias=False)
        )

        self.recp1 = nn.Sequential(
            BasicConv(channels_in*4, channels_out, kernel_size=1, dilation=1, bias=False,relu=True)
        )

    def forward(self, x):
        x0 = self.recp7(x)
        x1 = self.recp5(torch.cat([x,x0],dim=1))
        x2 = self.recp3(torch.cat([x,x0,x1],dim=1))
        out = self.recp1(torch.cat([x,x0,x1,x2],dim=1))

        return out


class BasicConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True, bias=False):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.bn = nn.BatchNorm2d(out_planes,eps=1e-5, momentum=0.01, affine=True) if bn else None
        self.relu = nn.PReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)

        return x


class residual_att(nn.Module):
    def __init__(self,channels_in, reduction=4):
        super(residual_att, self).__init__()
        self.channel_att=Channelatt(channels_in, reduction=reduction)
        self.spatialatt=Spatialatt(channels_in)

    def forward(self, x):
        return x + self.spatialatt(self.channel_att(x))


class Spatialatt(nn.Module):
    def __init__(self,channels_in):
        super(Spatialatt, self).__init__()
        kernel_size = 3
        self.spatial = BasicConv(channels_in, 1, kernel_size, stride=1, padding=(kernel_size-1) // 2, relu=False)
    def forward(self, x):
        x_out = self.spatial(x)
        scale = torch.sigmoid(x_out) # broadcasting

        return x * scale


class Channelatt(nn.Module):
    def __init__(self, channel, reduction=4):
        super(Channelatt, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)

        return x * y.expand_as(x)


class DownsamplerBlockDepthwiseConv(nn.Module):
    def __init__(self, nIn, nOut):
        super(DownsamplerBlockDepthwiseConv, self).__init__()
        self.nIn = nIn
        self.nOut = nOut

        if self.nIn < self.nOut:
            self.conv0 = nn.Conv2d(nIn, nOut-nIn, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
            self.conv1 = nn.Conv2d(nOut-nIn, nOut-nIn, 5, stride=2, padding=2, dilation=1, groups=nOut-nIn, bias=False)
            #self.pool = nn.MaxPool2d(2, stride=2)
            self.pool = nn.MaxPool2d(2, stride=2, ceil_mode=True)
        else:
            self.conv0 = nn.Conv2d(nIn, nOut, 1, stride=1, padding=0, dilation=1, groups=1, bias=False)
            self.conv1 = nn.Conv2d(nOut, nOut, 5, stride=2, padding=2, dilation=1, groups=nOut, bias=False)

        self.bn = nn.BatchNorm2d(nOut)
        self.act = nn.PReLU(nOut)

    def forward(self, input):
        if self.nIn < self.nOut:
            output = torch.cat([self.conv1(self.conv0(input)), self.pool(input)], 1)
        else:
            output = self.conv1(self.conv0(input))

        output = self.bn(output)
        output = self.act(output)

        return output