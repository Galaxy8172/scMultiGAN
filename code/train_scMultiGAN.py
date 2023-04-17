from __future__ import print_function, division
import argparse
from pathlib import Path
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch
from scMultiGAN import scMultiGAN
import Discriminator
import Generator
from utiles import MyDataset,ToTensor
parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=1000)
parser.add_argument('--batch-size', type=int, default=18)
parser.add_argument('--save-interval', type=int, default=10)
parser.add_argument('--tau', type=float, default=0)
parser.add_argument('--alpha', type=float, default=.2)
parser.add_argument('--lambd', type=float, default=10)
parser.add_argument('--n-critic', type=int, default=5)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--d_file', type=str, default='../breast_cancer/scMultiGAN.csv', help='path of data file')
parser.add_argument('--c_file', type=str, default='../breast_cancer/kmeans_label.txt', help='path of cls file')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--img_size', type=int, default=192, help='size of each image dimension')
parser.add_argument('--ncls', type=int, default=20, help='number of clusters')
parser.add_argument('--output_dir',type=str,default='../breast_cancer',help='path of the project')
parser.add_argument('--checkpoint',type=str,default=None,help='path of the output_checkpoint')
parser.add_argument('--lr',type=int,default=1e-4,help='Learning rate')
opt = parser.parse_args()
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
print("\n Using CUDA for training: {}".format(cuda))
print("********************\n")
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
print("reading expression matrix and mask matrix form disk\n"+"****************")
print("Expression matrix path is {}".format(opt.d_file))
print("Mask matrix path is {}".format(opt.c_file))
print("****************")
transformed_dataset = MyDataset(opt=opt,transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=opt.batch_size,shuffle=True, num_workers=42,drop_last=True)
print("Loading data: sucess!")
print("*********************\n")
data_gene = Generator.ConvDataGenerator(img_size=opt.img_size,ncls=opt.ncls,latent_dim=opt.latent_dim,channels=opt.channels).to(device)
mask_gene = Generator.ConvMaskGenerator(img_size=opt.img_size,ncls=opt.ncls,latent_dim=opt.latent_dim,channels=opt.channels).to(device)
data_critic = Discriminator.ConvCritic(img_size=opt.img_size,ncls=opt.ncls,channels=opt.channels).to(device)
mask_critic = Discriminator.ConvCritic(img_size=opt.img_size,ncls=opt.ncls,channels=opt.channels).to(device)
#output_dir = Path("../breast_cancer")
train = True
if(train):
    scMultiGAN(opt, data_gene, mask_gene, data_critic, mask_critic, data_loader=dataloader)
