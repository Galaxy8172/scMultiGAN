from torch.autograd import grad
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
import pandas as pd

class CriticUpdater:
    def __init__(self, critic, critic_optimizer, eps, ones, gp_lambda=10):
        self.critic = critic
        self.critic_optimizer = critic_optimizer
        self.eps = eps
        self.ones = ones
        self.gp_lambda = gp_lambda

    def __call__(self, real, fake,label_hot):
        real = real.detach()
        fake = fake.detach()
        label_hot = label_hot.detach()
        self.critic.zero_grad()
        self.eps.uniform_(0, 1)
        interp = (self.eps * real + (1 - self.eps) * fake).requires_grad_()
        grad_d = grad(self.critic(interp,label_hot), interp, grad_outputs=self.ones,
                      create_graph=True)[0]
        grad_d = grad_d.view(real.shape[0], -1)
        grad_penalty = ((grad_d.norm(dim=1) - 1)**2).mean() * self.gp_lambda
        w_dist = self.critic(fake,label_hot).mean() - self.critic(real,label_hot).mean()
        loss = w_dist + grad_penalty
        loss.backward()
        self.critic_optimizer.step()
        self.loss_value = loss.item()

def mask_norm(diff, mask):
    """Mask normalization"""
    dim = 1, 2, 3
    # Assume mask.sum(1) is non-zero throughout
    return ((diff * mask).sum(dim) / mask.sum(dim)).mean()

def mask_data(data, mask, tau):
    return mask * data + (1 - mask) * tau

def mkdir(path):
    path.mkdir(parents=True, exist_ok=True)
    return

def one_hot(batch,depth):
    ones = torch.eye(depth)
    return ones.index_select(0,batch)
class ToTensor(object):
    def __call__(self, sample):
        data,label,mask = sample['real_data'], sample['label'], sample['real_mask']
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        data = data.transpose((2, 0, 1))
        mask = mask.transpose((2, 0, 1))
        return {'real_data': torch.from_numpy(data).to(torch.float32),
                'label': torch.from_numpy(label),
                'real_mask': torch.from_numpy(mask).to(torch.float32)
                }
class MyDataset(Dataset):
    def __init__(self, opt,transform=None):
        self.data = pd.read_csv(opt.d_file, header=0, index_col=0,sep=",")
        d = pd.read_csv(opt.c_file, header=0, index_col=False)  #
        self.data_cls = pd.Categorical(d.iloc[:, 0]).codes  # ndarray
        self.transform = transform
        self.fig_h = opt.img_size  ##

    def __len__(self):
        return len(self.data_cls)

    def __getitem__(self, idx):
        # use astype('double/float') to sovle the runtime error caused by data mismatch.
        data = self.data.iloc[:, idx].values[0:(self.fig_h * self.fig_h), ].reshape(self.fig_h, self.fig_h, 1).astype(
            'float')  #
        label = np.array(self.data_cls[idx]).astype('int32')  #
        mask = np.where(data>0,1,0).astype('float')
        sample = {'real_data': data, 'label': label, 'real_mask':mask}
        if self.transform:
            sample = self.transform(sample)
        return sample

class GeneratorLoss(nn.Module):
    def __init__(self):
        super(GeneratorLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
    def forward(self,data_loss,out_images,real_images):
        adversarial_loss = data_loss
        image_loss = self.mse_loss(out_images, real_images)
        return image_loss + 0.1 * adversarial_loss