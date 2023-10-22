import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader,Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler


train_file=r"D:\task01plus\data\train_data .csv"
test_file=r"D:\task01plus\data\test_data.csv"
train_data = pd.read_csv(train_file,na_values=' ')
test_data = pd.read_csv(test_file,na_values=' ')
#用pandas读取文件，并将空白值认定为缺失值
train_data.fillna(0)#尝试过将空白值填充为均值和中位数，非常离谱的发生了，准确率都在30到40.不如填充0呜呜呜
test_data.fillna(0)


scaler=StandardScaler()

train_data = scaler.fit_transform(train_data)
test_data = scaler.fit_transform(test_data)
#标准化数据,作用是计算数据集的均值（mean）和标准差（standard deviation）对每个特征的数据进行标准化处理，即将每个特征的值减去均值，然后再除以标准差，从而使数据的均值为0，标准差为1。
# 计算数据集的均值（mean）和标准差（standard deviation）。
# 对每个特征的数据进行标准化处理，即将每个特征的值减去均值，然后再除以标准差，从而使数据的均值为0，标准差为1。




train_data_feature = torch.FloatTensor(train_data[:,1:-1])#保留所有的行，从第二列切到倒数第二列作为特征值
train_data_label = torch.LongTensor(train_data[:,-1])#切最后一行作为标签值
test_data_feature = torch.FloatTensor(test_data[:,1:-1])
test_data_label = torch.LongTensor(test_data[:,-1])

train_data_label = train_data_label.reshape(-1)#转化为一维向量，标签必须是一维的，变成N*1
test_data_label = test_data_label.reshape(-1)

class CustomDataset(Dataset):#用于加载特征和标签数据，并提供用于访问数据的方法
    def __init__(self,feature,label):
        self.feature = torch.FloatTensor(feature)
        self.label = torch.LongTensor(label)

    def __len__(self):
        return len(self.feature)

    def __getitem__(self,idx):
        feature = self.feature[idx]
        label = self.label[idx]
        return feature,label

train_data=CustomDataset(train_data_feature,train_data_label)#创建 CustomDataset 类并将特征和标签数据封装成数据集对象的作用是为了更方便地处理和使用数据。
test_data=CustomDataset(test_data_feature,test_data_label)
# 数据集封装：将特征和标签数据结合到一个数据集对象中，使得数据的组织更加清晰，方便管理和维护。
# 索引访问：通过实现 __getitem__ 方法，可以通过索引来访问数据集中的具体样本，方便在训练和评估过程中按需加载和处理数据。
# 数据集长度获取：通过实现 __len__ 方法，可以获取数据集的长度，即特征和标签的样本数

train_data=DataLoader(train_data,batch_size=32,shuffle=True)#batch_size为每个批次样本数量，样本数量过多可能让运行速率大大下降，shuffle=True即为打乱样本
test_data=DataLoader(test_data,batch_size=32,shuffle=False)


class SIRI(nn.Module):
    def __init__(self):
        super(SIRI, self).__init__()
        self.hidden1=nn.Linear(107,50)
        self.hidden2=nn.Linear(50,20)
        self.out=nn.Linear(20,6)
        self.relu=nn.ReLU()


    def forward(self, x):
        x=self.hidden1(x)
        x=self.relu(x)
        x=self.hidden2(x)
        x=self.relu(x)
        x=self.out(x)
        x=F.log_softmax(x,dim=1)

        return x

siri=SIRI()

losss=torch.nn.CrossEntropyLoss()#多分类问题时需要使用的交叉熵函数，二分类时需要的是另一个，已在学习笔记里面打出
optimizer=torch.optim.Adam(siri.parameters(),lr=0.01)#优化器用Adam(求求你不要拷打我Adam优化器的具体优化方法了我看不明白呜呜呜呜呜呜)

xlcs=10
for epoch in range(xlcs):
        cost=0
        for idx,(input,target) in enumerate(train_data):
            output=siri(input)
            loss=losss(output,target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            cost=cost+loss.item()
            if idx%300==0:
                print(epoch,idx,loss.item())

correct = 0
total = 0
with torch.no_grad():
    for (inputs, labels) in test_data:
        outputs = siri(inputs)
        _, predicted = torch.max(outputs, 1)#第一个值返回最大值，我们不需要所以狠狠舍去，第二个值是索引值，取索引值和正确值对比
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
print('accuracy:%d %%[%d/%d]'%(100*correct/total,correct,total) )


#下方是一开始尝试的失败的方法
# def test():
#     loss_list=[]
#     acc_list=[]
#     for idx,(input,target) in enumerate(test_data):#即enumerate，进行枚举（其索引及其对象）
#         with torch.no_grad():#不进行梯度计算（加快计算速度）
#             output=siri(input)
#             tloss=losss(output,target)
#             loss_list.append(tloss)
#             pred=output.max(dim=-1)[1]
#             cur_acc=pred.eq(target).float().mean()
#             acc_list.append(cur_acc)
#     print("平均准确率，平均损失：",np.mean(acc_list),np.mean(loss_list))
#
#
# if __name__=='__main__':
#     for i in range(1):
#         train(i)
#     test()
#这个便是在学习笔记里面提到的失败使用方法，这样的方法显示超出索引范围，





