import torch
import torch.nn as nn

# custom residual block class
class ResidualBlock(nn.Module):
    # initialize block, downsampling will occur explicitly externally
    def __init__(self, in_channels, out_channels, downsample=False):
        # inheritance
        super(ResidualBlock, self).__init__()
        # set stride
        stride = None
        if downsample:
            stride = 2
        else:
            stride = 1
        # apply 2 convolutional layers per level: convolution 3 x 3 + BatchNorm + ReLU
        # convolution: kernel size 3 + padding 1 preserves spatial dimensions, bias is false bc proceeding BatchNorm
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False)
        # normalize convolution output
        self.bn1 = nn.BatchNorm2d(out_channels)
        # activation (non-linear)
        self.relu = nn.ReLU(inplace=True)
        # 2nd convolution applied to output from first layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, stride=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Downsampling layer for identity path, if needed
        if downsample or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.downsample = nn.Identity()
    # forward pass
    def forward(self, x):
        # identity path
        identity = self.downsample(x)
        # main path
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # skip connection (addition:  main + identity)
        out += identity
        out = self.relu(out)
        # final tensor
        return out
# ResNet-based NN architecture class 
class ResNetLite(nn.Module):
    # only one class needed for sigmoid activation
    def __init__(self, num_classes=1):
        super(ResNetLite, self).__init__()
        # 3 input channels (RGB) converted to 64 feature maps
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        # NN layers, increase feature maps when downsampling, otherwise don't
        self.layer1 = ResidualBlock(64, 64, downsample=False)
        self.layer2 = ResidualBlock(64, 128, downsample=True)
        # reduce feature maps to single val and map to fully connected layer
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        # encoder layer 1 (returns 64 feature maps)
        x = self.stem(x)
        # encoder layer 2 (returns 64 feature maps)
        x = self.layer1(x)
        # encoder layer 3 (returns 128 feature maps)
        x = self.layer2(x)
        # reduce and flatten feature maps to 128 dim. tensor and transform into single raw logit
        x = self.global_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        # raw logits
        return x 

def get_model(num_classes=1):
    return ResNetLite(num_classes=num_classes)