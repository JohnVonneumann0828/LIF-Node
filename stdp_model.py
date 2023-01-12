from bindsnet.network.nodes import Input, LIFNodes
from bindsnet.network.topology import Connection
from bindsnet.learning import PostPre, MSTDP
from bindsnet.network import Network
import os
import torch
from PIL import Image
import numpy as np
from bindsnet.network.monitors import Monitor
from spikingjelly.clock_driven import neuron, functional, surrogate, layer, encoding
from tqdm import tqdm
from sklearn.preprocessing import normalize

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
        img=np.array(Image.open(img_item_path).convert('L'),'f')
        data = normalize(img, axis=0, norm='max')
        img=torch.tensor(data)
        img.resize_(181,156)
        #print(img)
        label=self.label_oneht
        return img,label
    def __len__ (self):
        return len(self.img_list)

network=Network(batch_size= 8)
time=500
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dt=0.005
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

source_layer=Input(n=181*156,traces=True)
target_layer=LIFNodes(n=1,traces=True)

connect=Connection(
    source=source_layer,target=target_layer,update_rule=MSTDP,nu=(1e-4,1e-2)
    )

network.add_layer(
    layer=source_layer,name="A"
    )

network.add_layer(
    layer=target_layer,name="B"
    )

network.add_connection(
    connection=connect,source="A",target="B"
    )
# Voltage recording for excitatory and inhibitory layers.
exc_voltage_monitor = Monitor(
    network.layers["A"], ["v"], time=int(time / dt),
)
inh_voltage_monitor = Monitor(
    network.layers["B"], ["v"], time=int(time / dt),
)
network.add_monitor(exc_voltage_monitor, name="exc_voltage")
network.add_monitor(inh_voltage_monitor, name="inh_voltage")

encoder = encoding.PoissonEncoder()

for i in range(len(train_list)):
    train_list[i] = torch.utils.data.DataLoader(dataset=train_list[i], batch_size=8,
                                                shuffle=True,
                                                drop_last=True,
                                                num_workers=0
                                                )
    test_list[i] = torch.utils.data.DataLoader(dataset=test_list[i], batch_size=8,
                                               shuffle=True,
                                               drop_last=True,
                                               num_workers=0
                                               )
for epoch in range(100):
    for i in train_list:
        for img1,label in tqdm(i):
            img1=encoder(img1).float()
            img1.to(device)
            label.to(device)
            img1={"A":img1}
            network.run(inputs=img1,time=500,input_time_dim=1)
            exc_voltages = exc_voltage_monitor.get("v")
            inh_voltages = inh_voltage_monitor.get("v")
            network.reset_state_variables()  # Reset state variables.
print(exc_voltages)
print(inh_voltages)

