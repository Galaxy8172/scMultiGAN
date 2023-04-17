import torch
import torch.optim as optim
import time
import sys
import seaborn as sns
from utiles import CriticUpdater, mask_norm, mkdir, mask_data,one_hot,GeneratorLoss
from pathlib import Path
use_cuda = torch.cuda.is_available()
device = torch.device('cuda' if use_cuda else 'cpu')
Tensor = torch.cuda.FloatTensor if use_cuda else torch.Tensor

def scMultiGAN_impute(args, data_gen, imputer,
                   impu_critic,
                  data_loader, output_dir, checkpoint=None):
    impute_dir = output_dir
    mkdir(impute_dir/'model_impute')
    model_dir = Path('../breast_cancer/model_impute')
    n_critic = args.n_critic
    gp_lambda = args.gp_lambda
    batch_size = args.batch_size
    nz = args.latent_dim
    epochs = args.epoch
    save_model_interval = args.save_interval
    alpha = args.alpha
    beta = args.beta
    gamma = args.gamma
    tau = args.tau
    img_size = args.img_size
    update_all_networks = not args.imputeronly

    n_batch = len(data_loader)
    data_noise = torch.FloatTensor(batch_size, nz).to(device)
    impu_noise = torch.FloatTensor(batch_size, args.channels,img_size,img_size).to(device)
    eps = torch.FloatTensor(batch_size, 1, 1, 1).to(device)
    ones = torch.ones(batch_size).to(device)
    lrate = 1e-4
    imputer_lrate = 1e-5

    generator_loss = GeneratorLoss()
    imputer_optimizer = optim.Adam(
        imputer.parameters(), lr=imputer_lrate, betas=(.5, .9))
    impu_critic_optimizer = optim.Adam(
        impu_critic.parameters(), lr=imputer_lrate, betas=(.5, .9))
    update_impu_critic = CriticUpdater(
        impu_critic, impu_critic_optimizer, eps, ones, gp_lambda)

    start_epoch = 0
    critic_updates = 0

    pretrain = torch.load(args.pretrain, map_location='cpu')
    data_gen.load_state_dict(pretrain['data_gen'])


    def save_model(path, epoch, critic_updates=0):
        torch.save({
            'imputer': imputer.state_dict(),
            'impu_critic': impu_critic.state_dict(),
            'epoch': epoch + 1,
            'critic_updates': critic_updates,
            'args': args,
        }, str(path))

    sns.set()
    start = time.time()
    epoch_start = start


    for epoch in range(start_epoch, epochs):
        print('start train')
        sum_data_loss, sum_mask_loss, sum_impu_loss = 0, 0, 0
        for i,real_sample in enumerate(data_loader):
            real_data = real_sample['real_data'].to(device)
            real_mask = real_sample['real_mask'].to(device)
            label = real_sample['label'].to(device)
            label_hot = one_hot((label).type(torch.LongTensor),args.ncls).type(Tensor).to(device)
            data_noise.normal_()
            fake_data = data_gen(data_noise,label_hot)

            impu_noise.uniform_()
            imputed_data = imputer(real_data, real_mask, impu_noise,label_hot)
            masked_imputed_data = mask_data(real_data, real_mask, imputed_data)

            update_impu_critic(fake_data, masked_imputed_data,label_hot)
            sum_impu_loss += update_impu_critic.loss_value
            critic_updates += 1
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D_impute loss: %f]" % (
                epoch + 1, args.epoch, i + 1, len(data_loader),
                sum_impu_loss))
            sys.stdout.flush()
            if critic_updates == n_critic:
                critic_updates = 0
                for p in impu_critic.parameters():
                    p.requires_grad_(False)
                impu_noise.uniform_()
                imputed_data = imputer(real_data, real_mask, impu_noise,label_hot)
                masked_imputed_data = mask_data(real_data, real_mask,
                                                imputed_data)
                impu_loss = 1-impu_critic(masked_imputed_data,label_hot).mean()
                impu_loss = generator_loss(impu_loss,fake_data,masked_imputed_data)
                imputer.zero_grad()
                if gamma > 0:
                    imputer_mismatch_loss = mask_norm(
                        (imputed_data - real_data) ** 2, real_mask)
                    (impu_loss + imputer_mismatch_loss * gamma).backward()
                else:
                    impu_loss.backward()
                imputer_optimizer.step()
                for p in impu_critic.parameters():
                    p.requires_grad_(True)
                print("\r[Epoch %d/%d] [Batch %d/%d] [G_impute loss: %f]" % (
                    epoch + 1, args.epoch, i + 1, len(data_loader),
                    impu_loss.detach().cpu().numpy().item()))
        mean_impu_loss = sum_impu_loss / n_batch
        print("epoch {} data_loss is {}".format(epoch + 1, mean_impu_loss))

        if save_model_interval > 0 and (epoch + 1) % save_model_interval == 0:
            save_model(model_dir / f'{epoch:04d}.pth', epoch, critic_updates)
        epoch_end = time.time()
        time_elapsed = epoch_end - start
        epoch_time = epoch_end - epoch_start
        epoch_start = epoch_end