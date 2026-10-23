import torch.nn as nn

# Adapted from the COMP3710 MNIST CNN practical (BasicBlock / ConvNetwork).


class BasicBlock(nn.Module):
    def __init__(self, in_dim, dim, kernel_size, stride=1, act_layer=nn.ReLU,
                 norm_layer=nn.BatchNorm2d, dropout_p=0.3):
        super().__init__()
        self.features = nn.Conv2d(in_dim, dim, kernel_size=kernel_size, stride=stride, bias=True)
        self.norm1 = norm_layer(dim)
        self.act_layer = act_layer()
        self.dropout = nn.Dropout2d(p=dropout_p)

    def forward(self, x):
        out = self.norm1(self.features(x))
        out = self.act_layer(out)
        return self.dropout(out)


class ConvNetwork(nn.Module):
    """Baseline CNN: two stride-2 conv blocks, global average pool, linear head."""

    def __init__(self, dim=64, in_channels=1, num_classes=2, dropout_p=0.3):
        super().__init__()
        self.block1 = BasicBlock(in_channels, dim, 3, stride=2, dropout_p=dropout_p)
        self.block2 = BasicBlock(dim, dim * 2, 3, stride=2, dropout_p=dropout_p)
        self.ave_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(dim * 2, num_classes)

    def forward(self, x):
        x = self.block2(self.block1(x))
        out = self.ave_pool(x).view(x.size(0), -1)
        return self.classifier(out)   # raw logits, no softmax
