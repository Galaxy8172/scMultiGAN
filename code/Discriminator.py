import torch
import torch.nn as nn

def conv_ln_lrelu(in_dim, out_dim):
    return nn.Sequential(
        nn.Conv2d(in_dim, out_dim, 3, 1, 1),
        nn.MaxPool2d(2,2),
        nn.InstanceNorm2d(out_dim, affine=True),
        nn.LeakyReLU(0.2))
class ConvCritic(nn.Module):
    def __init__(self,img_size,channels,ncls):
        super().__init__()
        dim = 16
        self.img_size = img_size
        self.channels = channels
        self.ncls = ncls
        self.input_size = 64
        self.l1 = nn.Sequential(nn.Linear(self.channels*self.img_size**2,self.input_size**2))
        self.firstconv = nn.Sequential(nn.Conv2d(self.channels, 16, kernel_size=3, padding=1, stride=1),nn.MaxPool2d(2,2),nn.BatchNorm2d(16),
                                       nn.LeakyReLU(0.2))
        self.upsample = nn.Sequential(nn.Conv2d(self.ncls, 8, kernel_size=3, padding=1, stride=1), nn.BatchNorm2d(8),nn.LeakyReLU(0.2),
                                      nn.Upsample(scale_factor=self.input_size//2, mode="bilinear"))
        self.ls = nn.Sequential(
            nn.Conv2d(24, dim, 3, 1, 1),nn.BatchNorm2d(dim), nn.LeakyReLU(0.2),
            conv_ln_lrelu(dim, dim // 2),
            conv_ln_lrelu(dim // 2, dim // 4),
            conv_ln_lrelu(dim // 4, dim // 8),
            conv_ln_lrelu(dim // 8, dim // 8),
            nn.Conv2d(dim // 8, 1, 2,1,0),
            nn.Sigmoid())

    def forward(self, input,label_hot):
        out1 = self.l1(input.view(input.shape[0],-1)).view(input.shape[0],self.channels,self.input_size,self.input_size)
        out1 = self.firstconv(out1)
        label_hot = label_hot.unsqueeze(2)
        label_hot = label_hot.unsqueeze(3)
        out2 = self.upsample(label_hot)

        out = torch.cat((out1,out2),1)
        net = self.ls(out)
        #net = torch.mean(net,-1)
        #net = torch.mean(net,-1)
        return net.view(-1)
#channels = 1;img_size=10;ncls=3
#c = ConvCritic(img_size=10,ncls=3,channels=1)
#input = torch.arange(1,401,dtype=torch.float32).view(4,1,10,10)
#label_hot = torch.arange(1,13,dtype=torch.float32).view(4,3)
#out = c(input,label_hot)
#print(out.shape)
#from MisGAN import CriticUpdater
#import torch.optim as optim
#lrate = 1e-4
#eps = torch.FloatTensor(4, 1, 1, 1)
#ones = torch.ones(4)
#data_critic_optimizer = optim.Adam(
       # c.parameters(), lr=lrate, betas=(.5, .9))
#update = CriticUpdater(c,data_critic_optimizer,eps=eps,ones=ones)
#update(input,input,label_hot=label_hot)