import torch
from torch import nn
from models.ETNet.args import ARGS
from torchvision.models.resnet import Bottleneck

class EdgeGuidanceModule(nn.Module):
    """Edge Guidance Module
    """
    def __init__(self):
        super().__init__()
        in_1_channels = ARGS['encoder'][0] * Bottleneck.expansion
        in_2_channels = ARGS['encoder'][1] * Bottleneck.expansion
        out_edge_channels = ARGS['egm'][0]
        out_channels = ARGS['egm'][1] * Bottleneck.expansion
        
        self.input_conv_1 = nn.Sequential(
            nn.Conv2d(in_1_channels, out_edge_channels, kernel_size=1, bias=False), # 1x1
            nn.BatchNorm2d(out_edge_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_edge_channels, out_edge_channels, kernel_size=3, padding=1, bias=False), # 3x3
            nn.BatchNorm2d(out_edge_channels),
            nn.ReLU(inplace=True),
        )

        self.input_conv_2 = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear'),
            nn.Conv2d(in_2_channels, out_edge_channels, kernel_size=1, bias=False), # 1x1
            nn.BatchNorm2d(out_edge_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_edge_channels, out_edge_channels, kernel_size=3, padding=1, bias=False), # 3x3
            nn.BatchNorm2d(out_edge_channels),
            nn.ReLU(inplace=True),
        )

        self.output_edge_conv = nn.Sequential(
            nn.Conv2d(2*out_edge_channels, out_edge_channels, kernel_size=1, bias=False), # 1x1
            # nn.BatchNorm2d(out_edge_channels),
            # nn.ReLU(inplace=True),
            #  # No ReLU because of incoming softmax
            
        )

        self.output_conv = nn.Sequential(
            nn.Conv2d(2*out_edge_channels, out_channels, kernel_size=1, bias=False), # 1x1
            # nn.BatchNorm2d(out_channels),
            # nn.ReLU(inplace=True),
        )

    def forward(self, x_1, x_2):
        input_conv_1 = self.input_conv_1(x_1)
        input_conv_2 = self.input_conv_2(x_2)
        input_conv = torch.cat((input_conv_1, input_conv_2), dim=1)
        return self.output_edge_conv(input_conv), self.output_conv(input_conv)
    
