import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

class Generator(nn.Module):
      def __init__(self,z_dim, channels_img , features_g) :
            super(Generator,self).__init__()
            self.net = nn.Sequential(
                  self._block(z_dim, features_g*16, 4,1,0),
                  self._block(features_g*16, features_g*8, 4,2,1),
                  self._block(features_g*8, features_g*4, 4,2,1),
                  self._block(features_g*4, features_g*2, 4,2,1),
                  nn.ConvTranspose2d(
                        features_g*2,channels_img,4,2,1
                  ),
                  nn.Tanh(),
            )
      
      def _block(self , in_channels , out_channels, kernel_size , stride , padding):
            return nn.Sequential(
                  nn.ConvTranspose2d(in_channels,out_channels,kernel_size,stride,padding),
                  nn.BatchNorm2d(out_channels),
                  nn.ReLU(),
            )
      def forward(self,x):
            return self.net(x)

