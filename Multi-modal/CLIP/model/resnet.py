import torch
import torch.nn as nn
import torch.nn.functional as F
from .attnpool import AttentionPool2d
from .utils import LayerNorm, QuickGELU
from collections import OrderedDict

class Bottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm2d(out_channels * self.expansion)
        )

        # Downsampling applied when stride > 1 or input/output channels do not match
        self.downsample = nn.Sequential()
        if stride != 1 or in_channels != out_channels * self.expansion:
            self.downsample = nn.Sequential(
                nn.AvgPool2d(stride),
                nn.Conv2d(in_channels, out_channels * self.expansion, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels * self.expansion)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.conv3(out)
        
        identity = self.downsample(identity)

        out += identity
        out = self.relu(out)
        return out
    
    
class ModifiedResnet(nn.Module):
    def __init__(self, num_blocks, output_dim, heads, input_resolution=224, width=64):
        super(ModifiedResnet, self).__init__()
        self.output_dim = output_dim
        self.input_resolution = input_resolution

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=width // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(num_features=width // 2),
            nn.ReLU(),
            nn.Conv2d(width // 2, width // 2, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width // 2),
            nn.ReLU(),
            nn.Conv2d(width // 2, width, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(width),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        )

        # ADD: Modify the layers with Bottleneck blocks and strides
        self._inplanes = width
        self.conv2 = self._make_layer(width, num_blocks[0], stride=1)
        self.conv3 = self._make_layer(width, num_blocks[1], stride=2)
        self.conv4 = self._make_layer(width * 2, num_blocks[2], stride=2)
        self.conv5 = self._make_layer(width * 4, num_blocks[3], stride=2)
        embed_dim = width * 32  # the ResNet feature dimension

        # ADD: Attention-based final pooling layer (instead of average pool)
        self.attnpool = AttentionPool2d(input_resolution // 32,embed_dim, heads, output_dim)

        # Initialize the layers
        self._init_layer()

    def _make_layer(self, planes, blocks, stride=1):
        num_blocks = [Bottleneck(self._inplanes, planes, stride)]

        self._inplanes = planes * Bottleneck.expansion
        for _ in range(1, blocks):
            num_blocks.append(Bottleneck(self._inplanes, planes))

        return nn.Sequential(*num_blocks)

    def _init_layer(self):
        # ADD: Weight initialization (same as in the original ResNet code)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.attnpool(x)
        
        return x
    
class ResidualAttentionBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int, attn_mask: torch.Tensor = None):
        super().__init__()

        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ln_1 = LayerNorm(d_model)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, d_model * 4)),
            ("gelu", QuickGELU()),
            ("c_proj", nn.Linear(d_model * 4, d_model))
        ]))
        self.ln_2 = LayerNorm(d_model)
        self.attn_mask = attn_mask

    def attention(self, x: torch.Tensor):
        self.attn_mask = self.attn_mask.to(dtype=x.dtype, device=x.device) if self.attn_mask is not None else None
        return self.attn(x, x, x, need_weights=False, attn_mask=self.attn_mask)[0]

    def forward(self, x: torch.Tensor):
        x = x + self.attention(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x