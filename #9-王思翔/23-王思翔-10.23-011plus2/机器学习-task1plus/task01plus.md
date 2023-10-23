# task01plus

呜呜呜呜呜呜最后打出来准确率只有66%左右，尝试过无数方法，都失败了（猛男哭泣），最离谱的事是尝试过无数次其它方法填充空缺值，什么平均数啊，中位数啊，最后发现准确率不如填0。。。。

```
accuracy:66 %[662/1000]

Process finished with exit code 0

```



## 1.数据的提取与标准化与封装

1.一开始并没有以下的代码，想当然认为空格会被自动认作空缺值，最后导致代码完全跑不起来，最后才发现应该使用na_values=‘ ’来标志空白值等于缺失值

```
train_data = pd.read_csv(train_file,na_values=' ')
test_data = pd.read_csv(test_file,na_values=' ')
```

2.在标准化的过程中，尝试过使用mnist的处理方式，发现自己纯纯脑袋有问题，经过不断的试错，（如Min-Max Normalization，Z-Score），最终选择了standardScaler(我也不知道为什么不要拷打我呜呜呜呜呜呜)

3.使用customdataset,用于加载特征和标签数据，并提供用于访问数据的方法

数据集封装：将特征和标签数据结合到一个数据集对象中，使得数据的组织更加清晰，方便管理和维护。

```
class CustomDataset(Dataset):
    def __init__(self,feature,label):
        self.feature = torch.FloatTensor(feature)
        self.label = torch.LongTensor(label)

    def __len__(self):#数据集长度获取：通过实现 __len__ 方法，可以获取数据集的长度，即特征和标签的样本数
        return len(self.feature)

    def __getitem__(self,idx):#索引访问：通过实现 __getitem__ 方法，可以通过索引来访问数据集中的具体样本，方便在训练和评估过程中按需加载和处理数据。
        feature = self.feature[idx]
        label = self.label[idx]
        return feature,label
```



## 2.数据的测试集尝试



大致结构与iris别无二致，除了使用切片把特征值和标签值切割开来了（最前面有sample_id，不切掉无法进行），之后尝试使用了task02的test检验方法，但是显示了超出索引范围（含有了-1这个超出索引范围的值)

```
#def test():
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

```

尝试过用筛子把不符合范围的值筛去，但如果这么做的化label和prediction就无法再一一对应上，如果同时删去了不符合范围的prediction所对应的label的话，太多数据会被删去，结果不再具有普适性（呜呜呜）

于是再次采用iris里面的方法，发现意外的可行（桀桀桀），但是准确率依旧只有悲剧的66%

我意识到是空缺值的填补问题导致了这种情况，但非常明显的是各种数据之间并没有明显的函数关系，搞得我焦头烂额，最终只好用0来填补空缺（呜呜呜）



## 3.关于交叉熵函数

交叉熵函数主要分为两种，二分类交叉熵函数，多分类交叉熵函数

二分类交叉熵函数代码：loss=nn.BCEWithLogitsLoss()

多分类交叉熵函数代码：loss=torch.nn.CrossEntropyLoss()
