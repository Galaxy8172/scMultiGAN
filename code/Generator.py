import torch
import torch.nn as nn
import torch.nn.functional as F

def add_mask_transformer(self, temperature=.66, hard_sigmoid=(-.1, 1.1)):
    self.temperature = temperature
    self.hard_sigmoid = hard_sigmoid

    if hard_sigmoid is False:
        self.transform = lambda x: torch.sigmoid(x / temperature)
    elif hard_sigmoid is True:
        self.transform = lambda x: F.hardtanh(
            x / temperature, 0, 1)
    else:
        a, b = hard_sigmoid
        self.transform = lambda x: F.hardtanh(
            torch.sigmoid(x / temperature) * (b - a) + a, 0, 1)

class ConvGenerator(nn.Module):
    def __init__(self,img_size,latent_dim,ncls,channels):
        super().__init__()
        self.img_size = img_size
        self.latent_dim = latent_dim
        self.ncls = ncls
        self.channels = channels
        self.cn1 = 32
        self.init_size = self.img_size // 4
        self.l1p = nn.Sequential(nn.Linear(self.latent_dim, self.cn1 * (self.img_size ** 2)))
        self.conv_blocks_01p = nn.Sequential(
            nn.BatchNorm2d(self.cn1),
            #            nn.Upsample(scale_factor=2),
            nn.Conv2d(self.cn1, self.cn1, 3, stride=1, padding=1),  # 32->32
            nn.BatchNorm2d(self.cn1, 0.8),
            nn.ReLU(),
        )
        self.conv_blocks_02p = nn.Sequential(
            #            nn.BatchNorm2d(9),
            nn.Upsample(scale_factor=self.img_size,mode="bilinear"),  # torch.Size([bs, 128, 16, 16]) #upsample
            nn.Conv2d(self.ncls, self.cn1 // 4, 3, stride=1, padding=1),  # torch.Size([bs, 128, 16, 16])
            nn.BatchNorm2d(self.cn1 // 4),
            nn.ReLU(),
        )
        self.conv_blocks_1 = nn.Sequential(
            nn.BatchNorm2d(40, 0.8),
            nn.Conv2d(40, self.cn1*16, 3, stride=1, padding=1),  # torch.Size([bs, 1, 32, 32])
            nn.BatchNorm2d(self.cn1*16),
            nn.ReLU(),
            nn.Conv2d(self.cn1*16,self.cn1*8,3,1,1),
            nn.BatchNorm2d(self.cn1*8),
            nn.ReLU(),
            nn.Conv2d(self.cn1*8,self.cn1*4,3,1,1),
            nn.BatchNorm2d(self.cn1*4),
            nn.ReLU(),
            nn.Conv2d(self.cn1*4,self.cn1*2,3,1,1),
            nn.BatchNorm2d(self.cn1*2),
            nn.ReLU(),
            nn.Conv2d(self.cn1*2,self.cn1,3,1,1),
            nn.BatchNorm2d(self.cn1),
            nn.ReLU(),
            nn.Conv2d(self.cn1, self.channels, 3, stride=1, padding=1),  # torch.Size([bs, 1, 32, 32])
        )
    def forward(self, noise,label_hot):
        out = self.l1p(noise)
        out = out.view(out.shape[0], self.cn1, self.img_size, self.img_size)
        out01 = self.conv_blocks_01p(out)

        label_hot = label_hot.unsqueeze(2)
        label_hot = label_hot.unsqueeze(3)
        out02 = self.conv_blocks_02p(label_hot)
        out1 = torch.cat((out01, out02), 1)
        out1 = self.conv_blocks_1(out1)
        return self.transform(out1)

class ConvDataGenerator(ConvGenerator):
    def __init__(self,img_size,latent_dim,ncls,channels):
        super().__init__(img_size=img_size,latent_dim=latent_dim,ncls=ncls,channels=channels)
        self.transform = lambda x: torch.sigmoid(x)

class ConvMaskGenerator(ConvGenerator):
    def __init__(self,img_size,latent_dim,ncls,channels,temperature=.66,
                 hard_sigmoid=(-.1, 1.1)):
        super().__init__(img_size=img_size,latent_dim=latent_dim,ncls=ncls,channels=channels)
        add_mask_transformer(self, temperature, hard_sigmoid)