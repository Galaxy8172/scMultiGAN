from __future__ import print_function, division
import argparse
from pathlib import Path
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data.sampler import BatchSampler
import Generator
import Discriminator
from scMultiGAN_impute import scMultiGAN_impute
import imputer
from utiles import MyDataset,ToTensor,one_hot,mask_data

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=300)
parser.add_argument('--batch-size', type=int, default=12)
parser.add_argument('--save-interval', type=int, default=10)
parser.add_argument('--tau', type=float, default=0.5)
parser.add_argument('--alpha', type=float, default=.1)
parser.add_argument('--gamma', type=float, default=0)
parser.add_argument('--beta', type=float, default=.1)
parser.add_argument('--pretrain', default=Path('../breast_cancer/model/0369.pth'))
parser.add_argument('--gp-lambda', type=float, default=10)
parser.add_argument('--n-critic', type=int, default=5)
parser.add_argument('--latent_dim', type=int, default=128)
parser.add_argument('--d_file', type=str, default='../breast_cancer/scMultiGAN.csv', help='path of data file')
parser.add_argument('--c_file', type=str, default='../breast_cancer/kmeans_label.txt', help='path of cls file')
parser.add_argument('--channels', type=int, default=1, help='number of image channels')
parser.add_argument('--img_size', type=int, default=192, help='size of each image dimension')
parser.add_argument('--imputeronly', action='store_true')
parser.add_argument('--ncls', type=int, default=20, help='number of clusters')
opt = parser.parse_args()
cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor

transformed_dataset = MyDataset(opt=opt,transform=transforms.Compose([ToTensor()]))
dataloader = DataLoader(transformed_dataset, batch_size=opt.batch_size,shuffle=True, num_workers=42,drop_last=True)

#dataloader = DataLoader(transformed_dataset, batch_size=opt.batch_size,shuffle=True, num_workers=24,drop_last=True)

data_gene = Generator.ConvDataGenerator(img_size=opt.img_size,ncls=opt.ncls,latent_dim=opt.latent_dim,channels=opt.channels).to(device)

imputer = imputer.UNetImputer(channels=opt.channels,img_size=opt.img_size,ncls=opt.ncls).to(device)
impu_critic = Discriminator.ConvCritic(img_size=opt.img_size,ncls=opt.ncls,channels=opt.channels).to(device)

output_dir = Path('../breast_cancer')
train = False
if(train):
    scMultiGAN_impute(opt, data_gene, imputer, impu_critic,
                  dataloader, output_dir, checkpoint=None)


for param in imputer.parameters():
    param.requires_grad = False
checkpoint = torch.load(Path("../breast_cancer/model_impute/0039.pth"))
#impu_noise = torch.FloatTensor(opt.batch_size,1,opt.img_size,opt.img_size).to(device)
imputer.load_state_dict(checkpoint['imputer'])
print(opt.d_file)
print(opt.c_file)

transformed_dataset = MyDataset(opt=opt,transform=transforms.Compose([ToTensor()]))
batch_sampler = BatchSampler(torch.utils.data.SequentialSampler(transformed_dataset),opt.batch_size,drop_last=False)
dataloader = DataLoader(transformed_dataset, batch_sampler=batch_sampler, shuffle=False, num_workers=42)

result = pd.DataFrame()
torch.manual_seed(123)
for j,real_sample in enumerate(dataloader):
    real_data = real_sample['real_data'].to(device)
    real_mask = real_sample['real_mask'].to(device)
    label = real_sample['label'].to(device)
    label_hot = one_hot((label).type(torch.LongTensor), opt.ncls).type(Tensor).to(device)
    impu_noise = torch.FloatTensor(real_data.shape[0], 1, opt.img_size, opt.img_size).to(device)
    impu_noise.normal_()
    impute_data = imputer(real_data,real_mask,impu_noise,label_hot)
    masked_imputed_data = mask_data(real_data, real_mask, impute_data)
    #real_data = real_data.reshape((real_data.shape[0],opt.img_size*opt.img_size)).T
    masked_imputed_data = masked_imputed_data.reshape((real_data.shape[0],opt.img_size**2)).T
    masked_imputed_data = masked_imputed_data.detach().cpu().numpy()
    result = pd.concat([result,pd.DataFrame(masked_imputed_data)],ignore_index=True,axis=1)

output_file = "../breast_cancer/imputed_data/scMultiGAN.csv"
result.to_csv(output_file, index=False)