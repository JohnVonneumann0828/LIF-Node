import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, surrogate, layer, encoding
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
import numpy as np
from torch.cuda import amp
import torch.utils.data.dataloader as Dataloader
from torch.utils.data import Dataset
import torch
import pandas as pd
from torchvision.io import read_image
from PIL import Image
from tqdm import tqdm
import time
from sklearn.preprocessing import normalize
import torchvision


class MyData(torch.utils.data.Dataset):
    def __init__(self, root_dir, label_dir, label_oneht):
        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        self.img_list = os.listdir(self.path)
        self.label_oneht = label_oneht

    def __getitem__(self, index):
        img_name = self.img_list[index]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        data=np.array(Image.open(img_item_path).convert('L'), 'f')
        data=normalize(data,axis=0,norm='max')
        #print(data)
        img = torch.tensor(data)
        #print(img)
        img.resize_(1, 181, 156)
        # print(img)
        label = self.label_oneht
        return img, label

    def __len__(self):
        return len(self.img_list)


class PythonNet(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T
        rate = 0.9
        self.static_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(4),
            # nn.Dropout(rate),
        )

        self.conv = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            # torch.nn.Dropout(rate),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(4, 4),  # 39 * 45
            # nn.Conv2d(4, 4, kernel_size=3, padding=1, bias=False),
            # nn.Dropout(rate),
            # nn.BatchNorm2d(4),
            # neuron.IFNode(surrogate_function=surrogate.ATan()),
            # nn.BatchNorm2d(8),
            # nn.Conv2d(8, 8, kernel_size=3, padding=1, bias=False),
            # nn.BatchNorm2d(8),
            # neuron.IFNode(surrogate_function=surrogate.ATan()),
            # nn.BatchNorm2d(8),
            nn.MaxPool2d(2, 2),  # 9 * 11

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 19 * 22, 4 * 4 * 4, bias=False),
            # torch.nn.Dropout(rate),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            # nn.Linear(4 * 10 * 10, 4 * 4 * 4, bias=False),
            # neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(4 * 4 * 4, 4, bias=False),
            # torch.nn.Dropout(rate),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            # nn.Linear(128 * 4 * 4, 4, bias=False),
            # neuron.IFNode(surrogate_function=surrogate.ATan()),
        )

    def forward(self, x):
        x = self.static_conv(x)

        out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter / self.T

class newnet1(nn.Module):
    def __init__(self,T,tau):
        super().__init__()
        self.T = T
        self.tau = tau
        self.out = nn.Sequential(
            nn.Flatten(),
            #nn.MaxPool2d(4,4),
            nn.Linear(16*181* 156, 16*4*3, bias=False),
            #neuron.IFNode(surrogate_function=surrogate.ATan()),
            neuron.LIFNode(tau=tau),
#            nn.BatchNorm1d(12),
            nn.Linear(16*4*3,4,bias=False),
            neuron.LIFNode(tau=tau),
            #neuron.IFNode(surrogate_function=surrogate.ATan()),
            #neuron.LIFNode(tau=tau),
            #nn.Linear(4*4,4,bias=False),
            #neuron.LIFNode(tau=tau)
        )
    def forward(self,x):
        out_spikes_counter = self.out(x)
       # print("DEEz")
        for t in range(1, self.T):
            #print(out_spikes_counter)
            out_spikes_counter += self.out(x)
        return out_spikes_counter/self.T



def main():
    tau = 2.0
    T = 16
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    #net = PythonNet(T)
    #net.to(device)
    lr = 0.0001
    train_list = []
    test_list = []
    root_dir = r"D:\sheep_classification\SheepFaceImages"
    label_dir = "Marino"
    train_list.append(MyData(root_dir, label_dir, 0))  # [1,0,0,0]
    label_dir = "Poll_Dorset"
    train_list.append(MyData(root_dir, label_dir, 1))
    label_dir = "Suffolk"
    train_list.append(MyData(root_dir, label_dir, 2))
    label_dir = "White_Suffolk"
    train_list.append(MyData(root_dir, label_dir, 3))

    root_dir = r"D:\sheep_classification\test_dataset"
    label_dir = "Marino"
    test_list.append(MyData(root_dir, label_dir, 0))
    label_dir = "Poll_Dorset"
    test_list.append(MyData(root_dir, label_dir, 1))
    label_dir = "Suffolk"
    test_list.append(MyData(root_dir, label_dir, 2))
    label_dir = "White_Suffolk"
    test_list.append(MyData(root_dir, label_dir, 3))

    out_dir = r"A:\Results_Sheep"
    start_epoch = 0
    T_max = 64

    max_test_acc = 0
    net=newnet1(T,tau)
    net = net.to(device)
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    encoder = encoding.PoissonEncoder()
    writer = SummaryWriter(os.path.join(out_dir, 'fmnist_logs'), purge_step=start_epoch)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)
    #checkpoint = torch.load(r"A:\Results_Sheep\checkpoint_max.pth", map_location='cuda')
    #net.load_state_dict(checkpoint['net'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #start_epoch = checkpoint['epoch'] + 1
    #max_test_acc = checkpoint['max_test_acc']

    lamda = 0.000001
    for i in range(len(train_list)):
        train_list[i] = torch.utils.data.DataLoader(dataset=train_list[i], batch_size=16,
                                                    shuffle=True,
                                                    drop_last=True,
                                                    num_workers=0
                                                    )
        test_list[i] = torch.utils.data.DataLoader(dataset=test_list[i], batch_size=2,
                                                   shuffle=True,
                                                   drop_last=True,
                                                   num_workers=0
                                                   )
    # [1, 10, 181, 156]
    # [10,1,181,156]
    for epoch in range(256):
        start_time = time.time()
        net.train()
        train_samples = 0
        train_acc = 0
        train_loss = 0
        regularization_loss = 0
        test_acc = 0
        for i in range(len(train_list)):
            for img, label in (tqdm(train_list[i])):
                # img.reshape(10, 1, 181, 156)
                img = img.to(device)
                label = label.to(device)
                # print(label)
                #print(img)
                label_onehot = F.one_hot(label, 4).float()  # [1,0,0,0]
                optimizer.zero_grad()
                #label_onehot.unsqueeze(-1)
                # label_onehot.resize(4)
                #print(encoder(img).float())
                print(list(img.size()))
                #print(encoder(img).float())
                out_spikes_counter_frequency=net(encoder(img).float())
                #print(out_spikes_counter_frequency)
                #print(out_spikes_counter_frequency)
                #print(out_spikes_counter_frequency)
                #print(label_onehot)
                # out_spikes_counter_frequency
                #print(torch.Size(label_onehot))
        #        print(out_spikes_counter_frequency[0])
        #        print(label_onehot[0])
                # out_fr=net(img)
                # for param in net.parameters():
                #    regularization_loss+=torch.sum(abs(param))
                # print(label_onehot)
                # print(label_onehot)
                # print(out_spikes_counter_frequency)
                loss = F.mse_loss(out_spikes_counter_frequency, label_onehot)
                # loss.data=loss.data+lamda*regularization_loss
                loss.backward()
                optimizer.step()

                train_samples += label.numel()
                train_loss += loss.item() * label.numel()
                train_acc += (out_spikes_counter_frequency.max(1)[1] == label).float().sum().item()
                functional.reset_net(net)
        train_acc /= train_samples
        train_loss /= train_samples
        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for i in range(len(test_list)):
                for frame, label in test_list[i]:
                    img = frame.to(device)
                    label = label.to(device)
                    # print(label)
                    label_onehot = F.one_hot(label, 4).float()
                    label_onehot.unsqueeze(-1)
                    # print(img)
#                    for t in range(T):
#                        if t == 0:
#                            out_spikes_counter = net(encoder(img).float())
#                        else:
#                            out_spikes_counter += net(encoder(img).float())
                    out_spikes_counter_frequency =net(encoder(img).float())
                    # out_fr = net(frame)
                    # print(out_fr)
                    loss = F.mse_loss(out_spikes_counter_frequency, label_onehot)
                    # print(out_fr)
                    test_samples += label.numel()
                    test_loss += loss.item() * label.numel()
                    test_acc += (out_spikes_counter_frequency.max(1)[1] == label).float().sum().item()
                    # test_acc += (out_fr.argmax(1) == label).float().sum().item()
                    functional.reset_net(net)
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }
        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(out_dir)
        print(
            f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')

    return 0


if __name__ == '__main__':
    main()