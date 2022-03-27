import requests
url = 'https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/spring-2022/utils/datasets/celeba.py'
open('celeba.py', 'wb').write(requests.get(url).content)
url = 'https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/spring-2022/data/celeba/list_attr_celeba.txt'
open('list_attr_celeba.txt', 'wb').write(requests.get(url).content)

import wandb
from src.dataset import CelebaCustomDataset
from torchvision import transforms
import torch
from src.model import VAE
from src.config import Config
from tqdm import tqdm
import torch.nn as nn

config = Config

# GET DATA
t_normalize = lambda x: x * 2 - 1
t_invnormalize = lambda x: (x + 1) / 2
transform = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    t_normalize,
])

dataset = CelebaCustomDataset(
    transform=transform,
    attr_file_path='list_attr_celeba.txt',
    crop=False
)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, drop_last=True)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = VAE(config)
model.to(device)

ckpt = torch.load('vae_ckpt.pth', map_location=device)
model.load_state_dict(ckpt)

optimizer = torch.optim.Adam(model.parameters(), lr=3e-4)


def train_epoch(dataloader, config, model, optimizer, epoch, path):
    device = config.device

    step = 0
    for image in tqdm(dataloader, total=len(dataloader)):
        real_image = image.to(device)
        image = image.to(device)

        optimizer.zero_grad()
        fake_image, mean, log_var = model(image)

        rec_loss = nn.MSELoss(reduction='sum')(real_image, fake_image)
        KL = -0.5 * torch.sum(1 + log_var - (mean ** 2) - torch.exp(log_var))
        loss = rec_loss + KL

        loss.backward()
        optimizer.step()

        wandb.log({'loss': loss.item(), 'rec_loss': rec_loss.item(), 'KL': KL.item()})

        if step % 100 == 0:
            with torch.no_grad():
                real_image = real_image.cpu().numpy()
                fake_image = fake_image.detach().cpu().numpy()
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
            torch.save(model.state_dict(), path + f'vae{epoch}.pth')
        step += 1


for i in range(10):
    wandb.init(project='GAN_HW3_VAE')
    path = ''
    train_epoch(dataloader, config, model, optimizer, i, path)
