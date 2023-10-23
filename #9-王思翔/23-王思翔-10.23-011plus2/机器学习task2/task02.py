
import torch.nn.functional as F
import torch
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets,transforms
from torchvision.datasets import MNIST
from torchvision.transforms import Compose,ToTensor,Normalize
import torch.nn as nn
from torch.optim import Adam
from torch.utils.data import random_split
#引入需要的库和方法
#引入数据并将其标准化，张量化
transform_fn=Compose([
    ToTensor(),#将图像化作张量
    Normalize(mean=(0.1307,), std=(0.3081,))#将图像进行标准化处理，减去均值再除以方差，这些值是根据 MNIST 数据集的统计信息计算得到的。（我也不知道它们咋来的）
]
)

mni = torchvision.datasets.MNIST(root="D:\mnist01",train=True,download=False,transform=transform_fn)
#mni[0][0].show()，可以用啦桀桀桀
train_size=int(0.8*len(mni))
test_size=len(mni)-train_size

train_set,test_set=random_split(mni,[train_size,test_size])

train_l=DataLoader(train_set,batch_size=10,shuffle=True)
test_l=DataLoader(test_set,batch_size=10,shuffle=False)



class MNISTNN(nn.Module):
    def __init__(self):
        super(MNISTNN,self).__init__()#是用于调用父类构造函数的语句
        self.hidden1=nn.Linear(1*28*28,392)#灰色图不像rgb一样有三个channel，只有一个channel
        self.hidden2=nn.Linear(392,196)
        self.hidden3=nn.Linear(196,98)
        self.hidden4=nn.Linear(98,10)
        self.relu=nn.ReLU()
        self.softmax=nn.Softmax()

    def forward(self,input):
        x=input.view([input.size(0),1*28*28])#转化成二维张量然后狠狠地用linear
        x=self.hidden1(x)
        x=self.relu(x)
        x=self.hidden2(x)
        x=self.relu(x)
        x=self.hidden3(x)
        x=self.relu(x)
        x=self.hidden4(x)
        out=F.log_softmax(x)
        return out

losss=torch.nn.CrossEntropyLoss()
model=MNISTNN()
def train(epoch):
    epoch=10
    optimizer=Adam(model.parameters(),lr=0.01)
    for idx,(input,target) in enumerate(train_l):
        output=model(input)
        loss=losss(output,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if idx%100==0:
            print(epoch,idx,loss.item())



def test():
    loss_list=[]
    acc_list=[]
    for idx,(input,target) in enumerate(test_l):#即enumerate，进行枚举（其索引及其对象）
        with torch.no_grad():#不进行梯度计算（加快计算速度）
            output=model(input)
            tloss=losss(output,target)
            loss_list.append(tloss)
            pred=output.max(dim=-1)[1]#沿着最后一个维度找最大值，取得最大值对应的索引
            cur_acc=pred.eq(target).float().mean()
            acc_list.append(cur_acc)
    print("平均准确率，平均损失：",np.mean(acc_list),np.mean(loss_list))

for i in range(1):
    train(i)
test()