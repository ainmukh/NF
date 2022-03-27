import requests
url = 'https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/spring-2022/utils/datasets/celeba.py'
open('celeba.py', 'wb').write(requests.get(url).content)
url = 'https://raw.githubusercontent.com/vpozdnyakov/DeepGenerativeModels/spring-2022/data/celeba/list_attr_celeba.txt'
open('list_attr_celeba.txt', 'wb').write(requests.get(url).content)

from src.train import train
from src.dataset import CelebaCustomDataset
from torchvision import transforms
import torch
from src.model import Glow
from src.config import Config

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
dataloader = torch.utils.data.DataLoader(dataset, batch_size=16, drop_last=True)


device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
model = Glow(config)
model.to(device)

n_bins = 2.0 ** config.n_bits
with torch.no_grad():
    image = next(iter(dataloader))
    model.init_with_data(image)
    image = image.to(device)
    log_p, log_det, _ = model(image + torch.rand_like(image) / n_bins)

optimizer = torch.optim.Adam(model.parameters(), lr=config.lr)

train(dataloader, model, optimizer, config, num_epochs=10)
