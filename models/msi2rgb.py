import torch
import torch.nn as nn
import torch.nn.functional as F

# 定义基于UNet的网络结构
class MSI2RGBNet(nn.Module):
    def __init__(self, msi2rgb_weight = False):
        super(MSI2RGBNet, self).__init__()
        # 下采样部分
        self.encoder1 = self.conv_block(28, 64)
        self.encoder2 = self.conv_block(64, 128)
        self.encoder3 = self.conv_block(128, 256)
        self.encoder4 = self.conv_block(256, 512)

        # 最底层的部分
        self.bottleneck = self.conv_block(512, 1024)

        # 上采样部分
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.decoder4 = self.conv_block(1024, 512)
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.decoder3 = self.conv_block(512, 256)
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.decoder2 = self.conv_block(256, 128)
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.decoder1 = self.conv_block(128, 64)

        # 输出层，将64通道转换为3通道（RGB）
        self.output_conv = nn.Conv2d(64, 3, kernel_size=1)

        if msi2rgb_weight:
            print(f'Loading pretrained weights from {msi2rgb_weight}...')
            weights = torch.load(msi2rgb_weight)
            self.load_state_dict(weights)
            print('Pretrained weights loaded successfully.')
            for param in self.parameters():
                param.requires_grad = False

    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        # 编码器部分
        e1 = self.encoder1(x)
        e2 = self.encoder2(F.max_pool2d(e1, 2))
        e3 = self.encoder3(F.max_pool2d(e2, 2))
        e4 = self.encoder4(F.max_pool2d(e3, 2))

        # 瓶颈部分
        b = self.bottleneck(F.max_pool2d(e4, 2))

        # 解码器部分
        d4 = self.upconv4(b)
        d4 = torch.cat((e4, d4), dim=1)
        d4 = self.decoder4(d4)
        d3 = self.upconv3(d4)
        d3 = torch.cat((e3, d3), dim=1)
        d3 = self.decoder3(d3)
        d2 = self.upconv2(d3)
        d2 = torch.cat((e2, d2), dim=1)
        d2 = self.decoder2(d2)
        d1 = self.upconv1(d2)
        d1 = torch.cat((e1, d1), dim=1)
        d1 = self.decoder1(d1)

        # 输出层
        out = self.output_conv(d1)
        return out