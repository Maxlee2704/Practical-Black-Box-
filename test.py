import torch

PATH = 'pretrain/mnist-b07bb66b.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = torch.load(PATH)

