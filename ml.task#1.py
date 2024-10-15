import torch #导入库
import torch.nn as nn #神经网络模块
import pandas as pd #读取数据

import matplotlib.pyplot as plt #绘图
#torch.autograd.set_detect_anomaly(True) #检测nan

data=pd.read_csv('housing6.csv')
#print(data.head())
#print(data.shape)

target=data.iloc[:, 13] #读取目标值
features=data.drop(data.columns[13],axis=1) #读取特征值
#print(features.head())
#print(target.head())

#标准化处理
from sklearn import preprocessing
input= preprocessing.StandardScaler().fit_transform(features)
#print(input)
#张量
input_tensor= torch.tensor(input, dtype=torch.float)
target_tensor=torch.tensor(target, dtype=torch.float)
#print(input_tensor)
#print(target_tensor)
#改变形状
target_tensor = target_tensor.unsqueeze(-1)


# 构建
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.linear = nn.Linear(13, 1)#13个特征值，输出一个值
    def forward(self,x):#前向传播
        x=self.linear(x)
        return x

# 实例化
ne = Net()
#print(ne)


criterion=nn.MSELoss()#MSE损失函数
optimizer=torch.optim.SGD(ne.parameters(), lr=0.01) #SGD优化器

# 训练
losses=[]
for epoch in range(500):
    optimizer.zero_grad() #清除数据
    output = ne(input_tensor) #前向传播
    loss = criterion(output, target_tensor) #损失
    loss.backward() #反向传播
    optimizer.step() #更新参数

    losses.append(loss.item())  

# 绘图
plt.plot(range(500), losses)
plt.xlabel('epoch')
plt.ylabel('losses')
plt.show()

