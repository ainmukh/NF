import torch

import wandb
from tqdm import tqdm

from src.loss import calc_loss


def train_epoch(dataloader, config, model, optimizer, epoch, path):
    n_bins = 2.0 ** config.n_bits
    device = config.device

    step = 0
    for image in tqdm(dataloader, total=len(dataloader)):
        real_image = image
        image = image.to(device)
        image = image * 255

        if config.n_bits < 8:
            image = torch.floor(image / 2 ** (8 - config.n_bits))

        image = image / n_bins - 0.5

        optimizer.zero_grad()
        log_p, log_det, z_outs = model(image + torch.rand_like(image) / n_bins)
        log_det = log_det.mean()
        loss, log_p, log_det = calc_loss(log_p, log_det, config.img_size, n_bins)
        loss.backward()
        optimizer.step()

        wandb.log({'loss': loss.item(), 'log_p': log_p.item(), 'log_det': log_det.item()})

        if step % 100 == 0:
            with torch.no_grad():
                real_image = real_image.numpy()
                fake_image = model.reverse(z_outs, reconstruct=True).cpu().numpy()
                sampled_image = model.sample(config.n_sample).cpu().numpy()
                total_steps = step + epoch * len(dataloader)
                wandb.log({'step': step,
                           'real images': [
                               wandb.Image(real_image[i].transpose(1, 2, 0),
                                           caption=f"real, step: {total_steps}")
                               for i in range(config.batch)],
                           'fake images': [
                               wandb.Image(fake_image[i].transpose(1, 2, 0),
                                           caption=f"fake, step: {total_steps}")
                               for i in range(config.batch)],
                           'sampled images': [
                               wandb.Image(sampled_image[i].transpose(1, 2, 0),
                                           caption=f"sampled, step: {total_steps}")
                               for i in range(config.n_sample)]})

        if step % (len(dataloader) // 2) == 0:
            torch.save(model.state_dict(), path + f'glow{epoch}.pth')
        step += 1


def train(dataloader, model, optimizer, config, num_epochs: int = 10):
    wandb.init(project='GAN_HW3_NF')

    for i in range(num_epochs):
        path = ''
        train_epoch(dataloader, config, model, optimizer, i, path)
