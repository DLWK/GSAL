import torch
from torch import nn
from models.ETNet.modules.encoder import Encoder

from models.ETNet.modules.decoder import Decoder
from models.ETNet.modules.edge_guidance_module import EdgeGuidanceModule
from models.ETNet.modules.weighted_aggregation_module import WeightedAggregationModule
from models.ETNet.args import ARGS
from torch.nn import functional as F

class ET_Net(nn.Module):
    """ET-Net: A Generic Edge-aTtention Guidance Network for Medical Image Segmentation
    """
    def __init__(self):
        super().__init__()
        self.encoder = Encoder() 
        self.decoder = Decoder()
        self.egm = EdgeGuidanceModule()
        self.wam = WeightedAggregationModule()

    def forward(self, x):
        _, _, h, w = x.size()
        enc_1, enc_2, enc_3, enc_4 = self.encoder(x)
        # print(enc_1.shape)
        # print(enc_2.shape)
        # print(enc_3.shape)
        # print(enc_4.shape)
      
        dec_1, dec_2, dec_3 = self.decoder(enc_1, enc_2, enc_3, enc_4)
        # print(dec_1.shape)
        # print(dec_2.shape)
        # print(dec_3.shape)
        
        edge_pred, egm = self.egm(enc_1, enc_2)
        pred = self.wam(dec_1, dec_2, dec_3, egm)
        edge_pred=F.interpolate(edge_pred, size=(h, w), mode='bilinear', align_corners=True)
        pred =F.interpolate(pred , size=(h, w), mode='bilinear', align_corners=True)

        return edge_pred, pred
    
    # def load_encoder_weight(self):
    #     # One could get the pretrained weights via PyTorch official.
    #     self.encoder.load_state_dict(torch.load(ARGS['encoder_weight']))

if __name__ == "__main__":


    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    net = ET_Net()
    # net.load_encoder_weight()
    
    net = net.cuda()
    net.train()
    img = torch.randn((2, 3, 512, 512)).cuda()
    out_edge, out = net(img)
    print(out.shape)
    
    # print(net)
    