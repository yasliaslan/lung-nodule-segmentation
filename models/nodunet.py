import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import mobilenet_v2, MobileNet_V2_Weights

class LightASPP(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 1)
        self.conv2 = nn.Conv2d(in_ch, out_ch, 3, padding=2, dilation=2)
        self.conv3 = nn.Conv2d(in_ch, out_ch, 3, padding=4, dilation=4)
        self.out = nn.Conv2d(out_ch * 3, out_ch, 1)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        return self.relu(self.out(torch.cat([x1, x2, x3], dim=1)))

class ResidualDilatedBlock(nn.Module):
    def __init__(self, in_ch, out_ch, dilation=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=dilation, dilation=dilation)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace=True)
        self.res_conv = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        identity = self.res_conv(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + identity)

class SpatioChannelAttention(nn.Module):
    def __init__(self, in_ch):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.channel_att = nn.Sequential(
            nn.Conv2d(in_ch, max(in_ch // 8, 4), 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(max(in_ch // 8, 4), in_ch, 1),
            nn.Sigmoid()
        )
        self.spatial_att = nn.Sequential(
            nn.Conv2d(2, 1, 7, padding=3),
            nn.Sigmoid()
        )

    def forward(self, x):
        ca = self.channel_att(self.avg_pool(x))
        x = x * ca
        max_pool = torch.max(x, dim=1, keepdim=True)[0]
        mean_pool = torch.mean(x, dim=1, keepdim=True)
        sa = self.spatial_att(torch.cat([max_pool, mean_pool], dim=1))
        return x * sa

class AttentionGate(nn.Module):
    def __init__(self, F_g, F_l, F_int):
        super().__init__()
        self.W_g = nn.Sequential(nn.Conv2d(F_g, F_int, 1), nn.BatchNorm2d(F_int))
        self.W_x = nn.Sequential(nn.Conv2d(F_l, F_int, 1), nn.BatchNorm2d(F_int))
        self.psi = nn.Sequential(nn.ReLU(inplace=True), nn.Conv2d(F_int, 1, 1), nn.Sigmoid())

    def forward(self, g, x):
        g1 = self.W_g(g)
        x1 = self.W_x(x)
        psi = self.psi(g1 + x1)
        return x * psi

class MobileNoduNet95(nn.Module):
    def __init__(self):
        super().__init__()
        mobilenet = mobilenet_v2(weights=MobileNet_V2_Weights.DEFAULT)
        mobilenet.features[0][0] = nn.Conv2d(1, 32, 3, stride=2, padding=1, bias=False)

        self.enc1 = nn.Sequential(*mobilenet.features[0:2])
        self.enc2 = nn.Sequential(*mobilenet.features[2:4])
        self.enc3 = nn.Sequential(*mobilenet.features[4:7])
        self.enc4 = nn.Sequential(*mobilenet.features[7:14])
        self.bottleneck = nn.Sequential(*mobilenet.features[14:17])

        self.aspp = nn.Sequential(
            ResidualDilatedBlock(160, 256),
            SpatioChannelAttention(256),
            LightASPP(256, 256)
        )

        self.up3 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.att3 = AttentionGate(128, 96, 64)
        self.norm_skip3 = nn.BatchNorm2d(128)
        self.dec3 = ResidualDilatedBlock(224, 128)

        self.up2 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.att2 = AttentionGate(64, 32, 32)
        self.norm_skip2 = nn.BatchNorm2d(64)
        self.dec2 = ResidualDilatedBlock(96, 64)

        self.up1 = nn.ConvTranspose2d(64, 32, 2, stride=2)
        self.att1 = AttentionGate(32, 24, 16)
        self.norm_skip1 = nn.BatchNorm2d(32)
        self.dec1 = ResidualDilatedBlock(56, 32)

        self.up0 = nn.ConvTranspose2d(32, 16, 2, stride=2)
        self.att0 = AttentionGate(16, 16, 8)
        self.norm_skip0 = nn.BatchNorm2d(16)
        self.dec0 = ResidualDilatedBlock(32, 16)

        self.up_out = nn.ConvTranspose2d(16, 8, 2, stride=2)
        self.final = nn.Sequential(
            nn.Conv2d(8, 8, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(8, 1, 1)
        )

        self.aux_out2 = nn.Conv2d(64, 1, 1)
        self.aux_out3 = nn.Conv2d(128, 1, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(e1)
        e3 = self.enc3(e2)
        e4 = self.enc4(e3)
        b = self.bottleneck(e4)
        b = self.aspp(b)

        d3 = self.up3(b)
        d3 = self.dec3(torch.cat([self.norm_skip3(d3), self.att3(d3, e4)], dim=1))

        d2 = self.up2(d3)
        d2 = self.dec2(torch.cat([self.norm_skip2(d2), self.att2(d2, e3)], dim=1))

        d1 = self.up1(d2)
        d1 = self.dec1(torch.cat([self.norm_skip1(d1), self.att1(d1, e2)], dim=1))

        d0 = self.up0(d1)
        d0 = self.dec0(torch.cat([self.norm_skip0(d0), self.att0(d0, e1)], dim=1))

        out = self.up_out(d0)
        out = self.final(out)

        aux2 = F.interpolate(self.aux_out2(d2), size=x.shape[2:], mode="bilinear", align_corners=False) * 0.5
        aux3 = F.interpolate(self.aux_out3(d3), size=x.shape[2:], mode="bilinear", align_corners=False) * 0.3

        return out, aux2, aux3