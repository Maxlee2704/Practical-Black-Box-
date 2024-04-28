import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
import torch.optim as optim
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Substitute(nn.Module):
    def __init__(self,num_cls):
        super().__init__()
        self.num_cls = num_cls
        backbone = models.resnet50(pretrained=False)
        self.model_backbone = nn.Sequential(*list(backbone.children())[:-1])
        self.clf = nn.Sequential(
            nn.Linear(2048,512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.5),
            nn.Linear(512,self.num_cls)
            )
    def forward(self,input):
        output = self.model_backbone(input)
        output = output.view(output.size(0), -1)
        output = self.clf(output)
        return output

def JBDA(x, lamda, data_grad):
    sign_data_grad = data_grad.sign()
    x_hat = x + lamda * sign_data_grad
    return x_hat

data = datasets.MNIST(root='./data')
dataloader = torch.utils.data.DataLoader(data, batch_size=32, shuffle=True)



lamda = 0.1
num_epochs = 30
lr = 0.0002
beta1 = 0.5
beta2 = 0.999
num_cls = 10

sub_model = Substitute().to(device)
BCE = nn.BCELoss()
opt = optim.Adam(sub_model.parameters(), lr=lr, betas=(beta1,beta2))


for sub_epch in range(4):
    for i,batch in enumerate(dataloader):
        batch = batch.to(device)
        batch.requires_grad = True
        output = sub_model(batch)

        loss = BCE(output, y)
        sub_model.zero_grad()
        loss.backward()

        data_grad = data.grad.data

        #Jacobian-based Dataset Augmentation
        temp = JBDA(batch,lamda,data_grad)

