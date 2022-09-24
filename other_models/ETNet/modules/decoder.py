from torch import nn
from models.ETNet.blocks.decoding_block import DecodingBlock
from torchvision.models.resnet import Bottleneck
from models.ETNet.args import ARGS

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.d_block_1 = DecodingBlock(
            ARGS['encoder'][2] * Bottleneck.expansion,
            ARGS['encoder'][3] * Bottleneck.expansion,
            ARGS['decoder'][0] * Bottleneck.expansion
        )
        self.d_block_2 = DecodingBlock(
            ARGS['encoder'][1] * Bottleneck.expansion,
            ARGS['decoder'][0] * Bottleneck.expansion,
            ARGS['decoder'][1] * Bottleneck.expansion
        )
        self.d_block_3 = DecodingBlock(
            ARGS['encoder'][0] * Bottleneck.expansion,
            ARGS['decoder'][1] * Bottleneck.expansion,
            ARGS['decoder'][2] * Bottleneck.expansion
        )

    def forward(self, x_1, x_2, x_3, x_4):
        output_1 = self.d_block_1(x_3, x_4)
        output_2 = self.d_block_2(x_2, output_1)
        output_3 = self.d_block_3(x_1, output_2)

        # print(x_1.size())
        # print(x_2.size())
        # print(x_3.size())
        # print(x_4.size())

        # print(output_1.size())
        # print(output_2.size())
        # print(output_3.size())

        return output_1, output_2, output_3