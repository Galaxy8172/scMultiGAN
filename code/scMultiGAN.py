import torch
import torch.optim as optim
import time
from utiles import CriticUpdater, mask_data, mkdir,GeneratorLoss,one_hot
import sys
from pathlib import Path
import os

cuda = torch.cuda.is_available()
device = torch.device('cuda' if cuda else 'cpu')
Tensor = torch.cuda.FloatTensor if cuda else torch.Tensor

def scMultiGAN(args, data_gen, mask_gen, data_critic, mask_critic, data_loader):
    n_critic = args.n_critic
    lambd = args.lambd
    batch_size = args.batch_size
    nz = args.latent_dim
    epochs = args.epoch
    save_interval = args.save_interval
    alpha = args.alpha
    tau = args.tau
    output_dir = args.output_dir
    checkpoint = args.checkpoint
    model_dir = mkdir(Path(output_dir) / 'model')
    model_dir = os.path.join(output_dir,"model")

    n_batch = len(data_loader)

    data_noise = torch.FloatTensor(batch_size, nz).to(device)
    mask_noise = torch.FloatTensor(batch_size, nz).to(device)


    eps = torch.FloatTensor(batch_size, 1, 1, 1).to(device)


    ones = torch.ones(batch_size).to(device)

    lrate = args.lr
    generator_loss = GeneratorLoss()
    data_gen_optimizer = optim.Adam(
        data_gen.parameters(), lr=lrate, betas=(.5, .9))
    mask_gen_optimizer = optim.Adam(
        mask_gen.parameters(), lr=lrate, betas=(.5, .9))

    data_critic_optimizer = optim.Adam(
        data_critic.parameters(), lr=lrate, betas=(.5, .9))
    mask_critic_optimizer = optim.Adam(
        mask_critic.parameters(), lr=lrate, betas=(.5, .9))

    update_data_critic = CriticUpdater(
        data_critic, data_critic_optimizer, eps, ones, lambd)
    update_mask_critic = CriticUpdater(
        mask_critic, mask_critic_optimizer, eps, ones, lambd)

    start_epoch = 0
    critic_updates = 0

    if checkpoint:
        print("Using pretained model {}".format(checkpoint))
        checkpoint = torch.load(Path(checkpoint))
        data_gen.load_state_dict(checkpoint['data_gen'])
        mask_gen.load_state_dict(checkpoint['mask_gen'])
        data_critic.load_state_dict(checkpoint['data_critic'])
        mask_critic.load_state_dict(checkpoint['mask_critic'])
        start_epoch = checkpoint['epoch'] - 1
        critic_updates = checkpoint['critic_updates']

    def save_model(path, epoch, critic_updates=0):
        torch.save({
            'data_gen': data_gen.state_dict(),
            'mask_gen': mask_gen.state_dict(),
            'data_critic': data_critic.state_dict(),
            'mask_critic': mask_critic.state_dict(),
            'epoch': epoch + 1,
            'critic_updates': critic_updates,
            'args': args,
        }, str(path))


    start = time.time()
    epoch_start = start

    for epoch in range(start_epoch, epochs):
        print("trainning is start")
        sum_data_loss, sum_mask_loss = 0, 0
        for i,real_sample in enumerate(data_loader):
            real_data = real_sample['real_data'].to(device)
            real_mask = real_sample['real_mask'].to(device)
            label = real_sample['label'].to(device)
            label_hot = one_hot((label).type(torch.LongTensor),args.ncls).type(Tensor).to(device)


            masked_real_data = mask_data(real_data, real_mask, tau)
            data_noise.normal_()
            mask_noise.normal_()
            #print(data_noise.dtype)
            fake_data = data_gen(data_noise,label_hot)
            fake_mask = mask_gen(mask_noise,label_hot)
            #print(fake_mask.dtype,fake_data.dtype)
            masked_fake_data = mask_data(fake_data, fake_mask, tau)
            #print(masked_fake_data.dtype)
            #interp = (eps * masked_real_data + (1 - eps) * masked_fake_data).requires_grad_()
            #real_out = data_critic(interp,label_hot)
            update_data_critic(masked_real_data, masked_fake_data,label_hot)
            update_mask_critic(real_mask, fake_mask,label_hot)

            sum_data_loss += update_data_critic.loss_value
            sum_mask_loss += update_mask_critic.loss_value

            critic_updates += 1
            sys.stdout.write("\r[Epoch %d/%d] [Batch %d/%d] [D_data loss: %f] [D_mask loss: %f]" % (
                epoch + 1, args.epoch, i + 1, len(data_loader),
                sum_data_loss, sum_mask_loss))
            sys.stdout.flush()
            if critic_updates == n_critic:
                critic_updates = 0

                # Update generators' parameters
                for p in data_critic.parameters():
                    p.requires_grad_(False)
                for p in mask_critic.parameters():
                    p.requires_grad_(False)

                data_noise.normal_()
                mask_noise.normal_()

                fake_data = data_gen(data_noise,label_hot)
                fake_mask = mask_gen(mask_noise,label_hot)
                masked_fake_data = mask_data(fake_data, fake_mask, tau)

                data_loss = 1-data_critic(masked_fake_data,label_hot).mean()
                data_loss = generator_loss(data_loss,masked_real_data,masked_fake_data)
                data_gen.zero_grad()
                data_loss.backward()
                data_gen_optimizer.step()

                data_noise.normal_()
                mask_noise.normal_()

                fake_data = data_gen(data_noise,label_hot)
                fake_mask = mask_gen(mask_noise,label_hot)
                masked_fake_data = mask_data(fake_data, fake_mask, tau)

                data_loss = 1-data_critic(masked_fake_data,label_hot).mean()
                data_loss = generator_loss(data_loss,masked_real_data,masked_fake_data)
                mask_loss = 1-mask_critic(fake_mask,label_hot).mean()
                mask_loss = generator_loss(mask_loss,real_mask,fake_mask)
                mask_gen.zero_grad()
                (mask_loss + data_loss * alpha).backward()
                mask_gen_optimizer.step()


                for p in data_critic.parameters():
                    p.requires_grad_(True)
                for p in mask_critic.parameters():
                    p.requires_grad_(True)
                print("\r[Epoch %d/%d] [Batch %d/%d] [G_data loss: %f] [G_mask loss: %f]" % (
                    epoch + 1, args.epoch, i + 1, len(data_loader),
                    data_loss.detach().cpu().numpy().item(),(mask_loss+data_loss*alpha).detach().cpu().numpy().item()))
        mean_data_loss = sum_data_loss / n_batch
        mean_mask_loss = sum_mask_loss / n_batch
        print("epoch {} data_loss is {}; mask_loss is {}".format(epoch+1,mean_data_loss,mean_mask_loss))


        if save_interval > 0 and (epoch + 1) % save_interval == 0:
            save_model(model_dir / f'{epoch:04d}.pth', epoch, critic_updates)

        epoch_end = time.time()
        time_elapsed = epoch_end - start
        epoch_time = epoch_end - epoch_start
        epoch_start = epoch_end
