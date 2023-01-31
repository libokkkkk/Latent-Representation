import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.utils.data as Data
import torchvision.datasets as datasets
import torchvision.transforms as transforms
import datetime
import numpy as np
from torchsummary import summary

class ResBlock(nn.Module):
    def __init__(self, n_chans):
        super(ResBlock, self).__init__()
        self.conv = nn.Conv2d(n_chans, n_chans,
                              kernel_size=3, padding=1,
                              bias=False)
        self.batch_norm = nn.BatchNorm2d(num_features=n_chans)  
        torch.nn.init.kaiming_normal_(self.conv.weight,
                                      nonlinearity='relu')  
        torch.nn.init.constant_(self.batch_norm.weight, 0.5)  
        torch.nn.init.zeros_(self.batch_norm.bias)

    def forward(self, x):
        out = self.conv(x)
        out = self.batch_norm(out)
        out = torch.relu(out)
        return out + x

class NetResDeep(nn.Module):
    def __init__(self, n_chans1=32, n_blocks=5, inchannel=3, fc1_dim = 149600):
        super(NetResDeep, self).__init__()
        self.n_chans1 = n_chans1
        self.conv1 = nn.Conv2d(inchannel, n_chans1, kernel_size=3, padding=1)
        self.resblocks = nn.Sequential(
            *(n_blocks * [ResBlock(n_chans=n_chans1)])
        )  
        self.fc1 = nn.Linear(fc1_dim, 32)
        self.fc2 = nn.Linear(32+7, 2)

    def forward(self, x,x2):
        in_size = x.size(0)  
        out = F.max_pool2d(torch.tanh(self.conv1(x)), 2)
        out = self.resblocks(out)
        out = F.max_pool2d(out, 2)
        
        out = out.view(in_size, -1)
        out = torch.relu(self.fc1(out))
        out = torch.cat((out, x2), dim=1)
        out = self.fc2(out)
        return out

print(NetResDeep())

device = (torch.device('cuda') if torch.cuda.is_available()
          else torch.device('cpu'))
print(f"Training on device {device}.")


def training_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(1, n_epochs + 1):
        loss_train = 0.0
        for imgs, x2, labels in train_loader:
            imgs = imgs.to(device=device)
            labels = labels.to(device=device)

            outputs = model(imgs,x2)
            loss = loss_fn(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_train += loss.item()

        if epoch == 1 or epoch % 10 == 0:
            print("{} Epoch {}, Training loss {:.6f}".format(
                datetime.datetime.now(), epoch,
                loss_train / len(train_loader)
            ))
    return model



