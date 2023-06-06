import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, im_chan=1, hidden_dim=16):
        super(Discriminator, self).__init__()
        
        self.conv1 = self.conv_block(im_chan, hidden_dim * 4)
            
        self.conv2 = self.conv_block(hidden_dim * 4, hidden_dim * 8)
        
        self.conv3 = nn.Conv2d(hidden_dim * 8, 1, kernel_size=4, stride=2, padding=0)

        
    def conv_block(self, input_channel, output_channel):
        return nn.Sequential(
                nn.Conv2d(input_channel, output_channel, kernel_size=4, stride=2, padding=0),
                nn.BatchNorm2d(output_channel),
                nn.LeakyReLU(0.2, inplace=True)
        )
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x
