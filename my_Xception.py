import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_normalization(x):
    batch_norm = nn.BatchNorm2d(num_features=x.shape[1])
    batch_norm = batch_norm.cuda()
    x = batch_norm(x)
    return x


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels=1, out_channels=1, kernel_size=3, stride=1, padding=1, groups=1):
        super(SeparableConv2d, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups, bias=False)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x
4
class EntryFlow(nn.Module):
    def __init__(self):
        super(EntryFlow, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 2, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1, 1)
        self.sepconv1 = SeparableConv2d(64, 128)
        self.sepconv2 = SeparableConv2d(128, 128)
        self.sepconv3 = SeparableConv2d(128, 256)
        self.sepconv4 = SeparableConv2d(256, 256)
        self.sepconv5 = SeparableConv2d(256, 728)
        self.sepconv6 = SeparableConv2d(728, 728)
        self.skip_conv1 = nn.Conv2d(128, 128, 1, 2, 0)
        self.skip_conv2 = nn.Conv2d(256, 256, 1, 2, 0)
        self.skip_conv3 = nn.Conv2d(728, 1024, 1, 2, 0)
        self.pool = nn.MaxPool2d(2, 2)


    def forward(self, x):
        x = F.relu(self.conv1(x))                   
        x = F.relu(self.conv2(x))
        x = batch_normalization(x)
        x_skip = F.relu(self.sepconv1(x))           #First block - input: 64
        x = F.relu(self.sepconv2(x_skip))
        x = self.pool(x)
        skip = F.relu(self.skip_conv1(x_skip))
        x = x + skip                                #Second block - input: 128
        x = batch_normalization(x)
        x_skip = F.relu(self.sepconv3(x))           
        x = F.relu(self.sepconv4(x_skip))
        x = self.pool(x)
        skip = F.relu(self.skip_conv2(x_skip))
        x = x + skip                                #Third block - input: 256
        x = batch_normalization(x)
        x_skip = F.relu(self.sepconv5(x))
        x = F.relu(self.sepconv6(x_skip))
        x = self.pool(x)
        skip = F.relu(self.skip_conv3(x_skip))      #x dimention = 728, skip dimention = 1024

        return x, skip


class MiddleFlow(nn.Module):
    def __init__(self, num_blocks):
        super(MiddleFlow, self).__init__()
        self.blocks = nn.ModuleList([self._make_block() for _ in range(num_blocks)])

    def _make_block(self):
        return nn.Sequential(
            SeparableConv2d(728, 728, 3),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv2d(728, 728, 3),
            nn.BatchNorm2d(728),
            nn.ReLU(),
            SeparableConv2d(728, 728, 3),
            nn.BatchNorm2d(728),
            nn.ReLU()
        )

    def forward(self, x):
        for block in self.blocks:
            x = block(x) + x
        return x

class ExitFlow(nn.Module):
    def __init__(self):
        super(ExitFlow, self).__init__()
        self.sepconv1 = SeparableConv2d(728, 728, 3)
        self.sepconv2 = SeparableConv2d(728, 1024, 3)
        self.sepconv3 = SeparableConv2d(1024, 1536, 3)
        self.sepconv4 = SeparableConv2d(1536, 2048, 3)        
        self.pool = nn.MaxPool2d(3, 2)
        self.batch_norm = nn.BatchNorm2d(1024)
        self.conv = nn.Conv2d(512, 2, 1)
        self.fc = nn.Linear(2048,1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, skip):
        x = F.relu(self.sepconv1(x))
        x = F.relu(self.sepconv2(x))
        x = x + skip
        x = self.pool(x)

        x = F.relu(self.batch_norm(x))
        x = F.relu(self.sepconv3(x))
        x = F.relu(self.sepconv4(x))
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = torch.squeeze(x)
        return x

class Xception(nn.Module):
    def __init__(self):
        super(Xception, self).__init__()
        self.entry_flow = EntryFlow()
        self.middle_flow = MiddleFlow(num_blocks=8)  # Number of middle blocks
        self.exit_flow = ExitFlow()

    def forward(self, x):
        x, skip = self.entry_flow(x)
        x = self.middle_flow(x)
        x = self.exit_flow(x, skip)
        return x


if __name__ == "__main__":
    # Instantiate and print the model
    model = Xception()
    print(model)

    # Example input
    x = torch.randn(1, 1, 512, 512)  # Batch size of 1, 1 channel, 512x512 image
    output = model(x)
    print(output.shape)  # Should be (1, 2) for 2-class classification
    print(output)
