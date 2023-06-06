import numpy as np
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, z_dim=10, im_chan=1, hidden_dim=64):
        super(Generator, self).__init__()
        
        self.z_dim = z_dim
        
        self.block_1 = self.gen_block(z_dim, hidden_dim * 4, kernel_size=3, stride=2)
        self.block_2 = self.gen_block(hidden_dim * 4, hidden_dim * 2, kernel_size=4)
        self.block_3 = self.gen_block(hidden_dim * 2, hidden_dim, kernel_size=3, stride=2)
        self.final_block = nn.Sequential(
            nn.ConvTranspose2d(hidden_dim, im_chan, kernel_size=4, stride=2, padding=0),
            nn.Tanh()
        )
        
    def gen_block(self, input_channel, output_channel, kernel_size, stride=1):
        return nn.Sequential(
            nn.ConvTranspose2d(input_channel, output_channel, kernel_size, stride, padding=0),
            nn.BatchNorm2d(output_channel),
            nn.ReLU(inplace=True),
        )
    
    
    def forward(self, noise):
        x = noise.view(len(noise), self.z_dim, 1, 1)
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.final_block(x)
        return x
    
    