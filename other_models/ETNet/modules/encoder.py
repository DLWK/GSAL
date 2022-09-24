# from turtle import shape
# from torchvision.models.resnet import ResNet, Bottleneck
from torchvision import models
from torch import nn

# class Encoder(ResNet):
#     def __init__(self):
#         super(Encoder, self).__init__(block=Bottleneck, layers=[3, 4, 6, 3])
#         # self.conv1.stride = 1 # 2 -> 1: No size/2

#     def _forward_impl(self, x):
          
#         x = self.conv1(x) 
#         x = self.bn1(x)
#         x = self.relu(x)
        
#         # x = self.maxpool(x) # No maxpool: No size/2

#         output_1 = self.layer1(x)
#         output_2 = self.layer2(output_1)
#         output_3 = self.layer3(output_2)
#         output_4 = self.layer4(output_3)

#         return output_1, output_2, output_3, output_4


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        ################################vgg16#######################################
        # feats = list(models.vgg16_bn(pretrained=True).features.children())
        # # print(nn.Sequential(*feats[:]))
        # feats[0] = nn.Conv2d(n_channels, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))   #适应任务
        # self.conv1 = nn.Sequential(*feats[:6])
        # # print(self.conv1)
        # self.conv2 = nn.Sequential(*feats[6:13])
        # # print(self.conv2)
        # self.conv3 = nn.Sequential(*feats[13:23])
        # self.conv4 = nn.Sequential(*feats[23:33])
        # self.conv5 = nn.Sequential(*feats[33:43])   #####增强细节
        # print(self.conv5)
       
        ################################Gate#######################################
        resnet = models.resnet50(pretrained=True)
        # print(resnet)
        resnet.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3))  
        self.firstconv = resnet.conv1
        
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4
        # print(self.encoder4)

    def forward(self, x):
      
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        x1 = self.firstmaxpool(x)
      
        x2 = self.encoder1(x1)
        x3 = self.encoder2(x2)
        x4 = self.encoder3(x3)
        x5 = self.encoder4(x4)

        return x2,x3,x4,x5