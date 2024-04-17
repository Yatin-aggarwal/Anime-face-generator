import torch 
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.utils
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
import torch.utils.data
import torch.utils.data.dataset
from Discriminator import Discriminator
from Generator import Generator
from weights import intialize_weights
import torchvision
from torchvision import transforms 
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import os 
from PIL import Image
from torchvision.utils import save_image

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
lr= 2e-4
batch_size = 128
image_size = 64
channels_img = 3
z_dim = 100
num_epoch = 2000
feature_Disc = 64
feature_Gen = 64


path = Path('Dataset')
transforms = transforms.Compose(
      [
            transforms.CenterCrop(128),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))
      ]
)
class dataset(Dataset):
      def __init__(self ,path ,transforms= None ):
            self.path = path
            self.transform = transforms
      
      def __len__(self):
            return len(os.listdir(self.path))
      def __getitem__(self, index) :
            dir = os.listdir(self.path)
            img = path/f'{dir[index]}'
            image = Image.open(img)
            if self.transform:
                  image = self.transform(image)
            return image


real_label = 1
data_set = dataset(path , transforms)    
data = DataLoader(data_set , batch_size , shuffle=True, num_workers=3)
gen = Generator(z_dim,channels_img,feature_Gen).to(device)
disc = Discriminator(channels_img,feature_Disc).to(device)
intialize_weights(gen)
intialize_weights(disc)
optimizer_gen = optim.Adam(gen.parameters(),lr=lr, betas=(0.5,0.999) )
optimizer_disc = optim.Adam(disc.parameters(),lr = lr , betas = (0.5,0.999))
criterion = nn.BCELoss()
fixed_noise = torch.randn(32, z_dim, 1,1 ).to(device)
writer_real = SummaryWriter(f"runs/Anime/real")
writer_fake = SummaryWriter(f"runs/Anime/fake")
gen.train()
disc.train()

step = 0 
for epoch in range(num_epoch):
      for i , real in enumerate(data):
           disc.zero_grad()
           real = real.to(device)
           disc_real = disc(real).view(-1)
           batch = real.size(0)
           noise = torch.randn(batch, z_dim,1,1).to(device)
           label = torch.full((batch,), real_label).to(device)
           loss_disc_real = criterion(disc_real.float(), label.float()).to(device)
           loss_disc_real.backward()
           label.fill_(0)
           fake = gen(noise)
           disc_fake = disc(fake.detach()).view(-1)
           loss_disc_fake = criterion(disc_fake.float(),label.float()).to(device)
           loss_disc_fake.backward()
           loss_disc = (loss_disc_real+loss_disc_fake)
           optimizer_disc.step()
           gen.zero_grad()
           label.fill_(real_label)
           output = disc(fake).view(-1)
           loss_gen = criterion(output.float(),label.float()).to(device)
           loss_gen.backward()
           optimizer_gen.step()
           if i% 5 == 0 :
                 print(f"Epoch [{epoch}/{num_epoch}] Batch {i}/{len(data)} Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}")
                 with torch.no_grad():
                       fake = gen(fixed_noise)
                       img_grid_real = torchvision.utils.make_grid(real[:32],normalize=True)
                       img_grid_fake = torchvision.utils.make_grid(fake[:32],normalize=True)
                       writer_real.add_image("Real",img_grid_real,global_step=step)
                       writer_fake.add_image("Fake",img_grid_fake,global_step=step)
                       step += 1
            
