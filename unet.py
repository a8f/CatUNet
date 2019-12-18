from torch import nn


class UNet(nn.Module):
    """
    UNet modified from the original paper to work on 180*176 images
    """
    def __init__(self):
        super(Q2Net, self).__init__()
        self.down1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.down2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.down3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.up1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.up2 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.up3 = nn.Sequential(
            nn.Conv2d(32, 3, 3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Upsample(scale_factor=2)
        )
        self.final = nn.Conv2d(3, 2, 3, padding=1)

    def forward(self, x):
        out = self.down1(x)
        out = self.down2(out)
        out = self.down3(out)
        out = self.middle(out)
        out = self.up1(out)
        out = self.up2(out)
        out = self.up3(out)
        return self.final(out)
