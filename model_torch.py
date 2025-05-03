# ----------------- model_torch.py -------------------
import torch.nn as nn, torch


class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1):
        super().__init__()
        self.c1 = ConvBlock(in_ch, 64)
        self.p1 = nn.MaxPool2d(2)
        self.c2 = ConvBlock(64, 128)
        self.p2 = nn.MaxPool2d(2)
        self.c3 = ConvBlock(128, 256)
        self.p3 = nn.MaxPool2d(2)
        self.c4 = ConvBlock(256, 512)
        self.dropout4 = nn.Dropout(0.5)
        self.p4 = nn.MaxPool2d(2)
        self.c5 = ConvBlock(512, 1024)
        self.dropout5 = nn.Dropout(0.5)
        # Expansive path
        self.up6 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.c6 = ConvBlock(1024, 512)
        self.up7 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.c7 = ConvBlock(512, 256)
        self.up8 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.c8 = ConvBlock(256, 128)
        self.up9 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.c9 = ConvBlock(128, 64)
        self.c9_out = nn.Conv2d(64, 2, kernel_size=3, padding=1)
        self.out_conv = nn.Conv2d(2, out_ch, kernel_size=1)

    def forward(self, x):
        c1 = self.c1(x)
        p1 = self.p1(c1)
        c2 = self.c2(p1)
        p2 = self.p2(c2)
        c3 = self.c3(p2)
        p3 = self.p3(c3)
        c4 = self.c4(p3)
        c4d = self.dropout4(c4)
        p4 = self.p4(c4d)
        c5 = self.dropout5(self.c5(p4))
        up6 = torch.relu(self.up6(c5))
        merge6 = torch.cat([c4d, up6], dim=1)
        c6 = self.c6(merge6)
        up7 = torch.relu(self.up7(c6))
        merge7 = torch.cat([c3, up7], dim=1)
        c7 = self.c7(merge7)
        up8 = torch.relu(self.up8(c7))
        merge8 = torch.cat([c2, up8], dim=1)
        c8 = self.c8(merge8)
        up9 = torch.relu(self.up9(c8))
        merge9 = torch.cat([c1, up9], dim=1)
        c9 = self.c9(merge9)
        c9_out = self.c9_out(c9)
        return self.out_conv(c9_out)
