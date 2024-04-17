import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
class Discriminator(nn.Module):
      def __init__(self, in_channels, features):
            super(Discriminator, self).__init__()
            self.disc = nn.Sequential(
                  nn.Conv2d(
                        in_channels,
                        features,
                        kernel_size=4,
                        stride=2,
                        padding=1,
                  ),
                  nn.BatchNorm2d(features),
                  nn.LeakyReLU(0.2),
                  self._block(features,features*2,4,2,1),
                  self._block(features*2,features*4,4,2,1),
                  self._block(features*4,features*8,4,2,1),
                  nn.Conv2d(
                        features*8,
                        1,
                        kernel_size=4,
                        stride=2,
                        padding=0,
                  ),
                  nn.Sigmoid(),

            )



      def _block(self, in_channels , features, kernel_size , stride , padding):
            return nn.Sequential(
                  nn.Conv2d(in_channels,
                            features,
                            kernel_size,
                            stride,
                            padding,
                            bias=False
                            ),
                  nn.BatchNorm2d(features),
                  nn.LeakyReLU(0.2)
            )
      def forward(self,x):
            return self.disc(x)
