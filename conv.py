import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
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
import torchvision
from matplotlib import pyplot as plt
from spikingjelly import visualizing
class MyData(torch.utils.data.Dataset):
    def __init__(self,root_dir,label_dir,label_oneht):
        self.root_dir=root_dir
        self.label_dir=label_dir
        self.path=os.path.join(self.root_dir,self.label_dir)
        self.img_list=os.listdir(self.path)
        self.label_oneht=label_oneht
    def __getitem__(self, index):
        img_name=self.img_list[index]
        img_item_path=os.path.join(self.root_dir,self.label_dir,img_name)
        img=torch.LongTensor(np.array(Image.open(img_item_path).convert('L'),'f'))
        img.resize_(1,181,156)
        #print(img)
        label=self.label_oneht
        return img,label
    def __len__ (self):
        return len(self.img_list)

class PythonNet(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

        self.static_conv = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=4, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(4),
        )

        self.conv = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.BatchNorm2d(4),
            nn.MaxPool2d(4,4),  # 39 * 45
            nn.Conv2d(4, 4, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm2d(4),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            #nn.MaxPool2d(2, 2)  # 9 * 11

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(4 * 39 * 45, 4*10*10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(4 * 10 * 10, 4 * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(4 * 4 * 4, 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            #nn.Linear(128 * 4 * 4, 4, bias=False),
            #neuron.IFNode(surrogate_function=surrogate.ATan()),
        )


    def forward(self, x):
        x = self.static_conv(x)

        out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter / self.T
def main():
    print("Deez")
    T=4
    device=("cpu" if torch.cuda.is_available() else "cpu")
    net=PythonNet(T)
    net.to(device)
    lr=0.00001
    optimizer=torch.optim.Adam(net.parameters(),lr)

    b=2
    train_list=[]
    test_list=[]

    root_dir = r"D:\sheep_classification\SheepFaceImages"
    label_dir = "Marino"
    train_list.append(MyData(root_dir, label_dir,0))#[1,0,0,0]
    label_dir = "Poll_Dorset"
    train_list.append(MyData(root_dir, label_dir,1))
    label_dir = "Suffolk"
    train_list.append(MyData(root_dir, label_dir,2))
    label_dir = "White_Suffolk"
    train_list.append(MyData(root_dir, label_dir,3))

    root_dir = r"D:\sheep_classification\test_dataset"
    label_dir = "Marino"
    test_list.append(MyData(root_dir, label_dir,0))
    label_dir = "Poll_Dorset"
    test_list.append(MyData(root_dir, label_dir,1))
    label_dir = "Suffolk"
    test_list.append(MyData(root_dir, label_dir,2))
    label_dir = "White_Suffolk"
    test_list.append(MyData(root_dir, label_dir,3))

    out_dir=r"A:\Results_Sheep"
    start_epoch=0
    T_max=64

    writer = SummaryWriter(os.path.join(out_dir, 'fmnist_logs'), purge_step=start_epoch)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    max_test_acc = 0

    #checkpoint = torch.load("checkpoint_max.pth", map_location='cpu')
    #net.load_state_dict(checkpoint['net'])
    #optimizer.load_state_dict(checkpoint['optimizer'])
    #lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
    #start_epoch = checkpoint['epoch'] + 1
    #max_test_acc = checkpoint['max_test_acc']
    encoder=nn.Sequential(
        net.static_conv,
        net.conv[0]
    )
    encoder.eval()
    lr=0.0001
    for i in range(len(train_list)):
        test_list[i]=torch.utils.data.DataLoader(dataset=test_list[i],batch_size=1,
        shuffle=True,
        drop_last=True,
        num_workers=0
                                                 )
    #[1, 10, 181, 156]
    #[10,1,181,156]
    for epoch in range(1):
        start_time = time.time()
        train_acc=0
        train_loss=0
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for i in range(len(test_list)):
                for img, label in test_list[i]:
                    img=img.float().to(device)
                    label = label.to(device)
                    fig = plt.figure(dpi=200)
                    out_fr = net(img)
                    out_spikes = 0
                    num=out_fr.argmax(1)
                    label=label[0]
                    num=num[0]
                    label_name=""
                    actual_name = ""
                    print(label)
                    print(num)
                    if label==0:
                        label_name="Marino"
                    elif label==1:
                        label_name ="Poll Dorset"
                    elif label==2:
                        label_name="Suffolk"
                    elif label==3:
                        label_name = "White Suffolk"
                    if num==0:
                        actual_name = "Marino"
                    elif num==1:
                        actual_name = "Poll Dorset"
                    elif num==2:
                        actual_name = "Suffolk"
                    elif num == 3:
                        actual_name = "White Suffolk"
                    print(label_name)
                    print(actual_name)
                    plt.text(0,0,"Actual Sheep: "+label_name,ha='center',va='bottom')
                    plt.text(0, 10,"Predicted Sheep: "+ actual_name, ha='center', va='bottom')
                    plt.imshow(img.squeeze().numpy(), cmap='gray')
                    # 注意输入到网络的图片尺寸是 ``[1, 1, 28, 28]``，第0个维度是 ``batch``，第1个维度是 ``channel``
                    # 因此在调用 ``imshow`` 时，先使用 ``squeeze()`` 将尺寸变成 ``[28, 28]``
                    plt.title('Input image', fontsize=20)
                    plt.xticks([])
                    plt.yticks([])
                    plt.show()
                    for t in range(net.T):
                        out_spikes += encoder(img).squeeze()
                        # encoder(img)的尺寸是 ``[1, 128, 28, 28]``，同样使用 ``squeeze()`` 变换尺寸为 ``[128, 28, 28]``
                        if t == 0 or t == net.T - 1:
                            out_spikes_c = out_spikes.clone()
                            for i in range(out_spikes_c.shape[0]):
                                if out_spikes_c[i].max().item() > out_spikes_c[i].min().item():
                                    # 对每个feature map做归一化，使显示更清晰
                                    out_spikes_c[i] = (out_spikes_c[i] - out_spikes_c[i].min()) / (
                                                out_spikes_c[i].max() - out_spikes_c[i].min())
                            visualizing.plot_2d_spiking_feature_map(out_spikes_c, nrows=45, ncols=39, space=1, title=None)
                            plt.title('$\\sum_{t} S_{t}$ at $t = ' + str(t) + '$'+label_name+' '+actual_name, fontsize=20)
                            plt.show()
        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)
        print(out_dir)
        print(f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')

    return 0
if __name__ == '__main__':
    main()