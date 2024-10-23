# 机器学习task1实验报告

## 【1】崎岖的实验过程

* 茫然地在b站和csdn上看了一些资料，弄了弄ananconda和pytorch，然后我开始写task1......

前面的时间，我写一点运行一点，然后报了各式各样的bug🤣



![946E27D584B58E767EA48C94BF368938](https://raw.githubusercontent.com/nnihawk404/images/main/ml2.png)

![8D34FDD34476E7A96CE8C7BC4AC678D2](https://raw.githubusercontent.com/nnihawk404/images/main/ml1.png)

* 修了很久bug，然后又出了bug说是输入了一个[506.505]的矩阵，和层的[13.1]不匹配，我当时把神经元数量从13改成了505，然后就运行成功了😲



<img src="https://raw.githubusercontent.com/nnihawk404/images/main/ml3.jpg" alt="6ACEA6A3BCBFD7A34F944BD46C683231" style="zoom: 25%;" />

* 我已经准备开始弄task2了，突然意识到了不能这么改,是输入数据不太对，输出了数据的特征维度，发现是[505.1] （数据没有分列），我手动用wps把数据分了个列
* 分列之后终于运行成功了，然后得到了一个诡异的空白图像🫤

<img src="https://raw.githubusercontent.com/nnihawk404/images/main/ml4.jpg" alt="327D0B6701B292578D33972585FC20A7" style="zoom: 80%;" />





<img src="https://raw.githubusercontent.com/nnihawk404/images/main/ml111.jpg" alt="b2e5cdffd54fe9e38a4a5ab3a3d225d0" style="zoom: 33%;" />





尝试把学习率调到了0.000000000000000000000000000000000000000000000001也没有用，

用torch.autograd.set_detect_anomaly(True)检查了一下，发现有些行的数据里出现了很多nan（应该是我手动分列造成的）

由于不知道太好的解决办法，只能又手动删了大概50个样本🙃

然后就运行成功了



## 【2】模型结构和训练过程

* 模型结构

  > \# 构建
  >
  > class Net(nn.Module):
  >
  >   def __init__(self):
  >
  > ​    super(Net, self).__init__()
  >
  > ​    self.linear = nn.Linear(13, 1)#13个特征值，输出一个值
  >
  >   def forward(self,x):#前向传播
  >
  > ​    x=self.linear(x)
  >
  > ​    return x

我只建了一个层，有13个神经元，输入13个特征值，输出1个预测值

* 训练过程

  > criterion=nn.MSELoss()#MSE损失函数
  >
  > optimizer=torch.optim.SGD(ne.parameters(), lr=0.01) #SGD优化器
  >
  > 
  >
  > \# 训练
  >
  > losses=[]
  >
  > for epoch in range(500):
  >
  >   optimizer.zero_grad() #清除数据
  >
  >   output = ne(input_tensor) #前向传播
  >
  >   loss = criterion(output, target_tensor) #损失
  >
  >   loss.backward() #反向传播
  >
  >   optimizer.step() #更新参数
  >
  > 
  >
  >   losses.append(loss.item())  
  >
  > 
  >
  > \# 绘图
  >
  > plt.plot(range(500), losses)
  >
  > plt.xlabel('epoch')
  >
  > plt.ylabel('losses')
  >
  > plt.show()

  

我用了MSE损失函数和SGD优化器

学习率0.01，500次

## 【3】结果



![屏幕截图 2024-10-15 195210](https://raw.githubusercontent.com/nnihawk404/images/main/ml5.png)

随训练次数增加，损失值不断变小，模型性能有所提升



## 附

*完整代码：*

> import torch #导入库
>
>import torch.nn as nn #神经网络模块
>
>import pandas as pd #读取数据
>
>
>
>import matplotlib.pyplot as plt #绘图
>
>\#torch.autograd.set_detect_anomaly(True) #检测nan
>
>
>
>data=pd.read_csv('housing6.csv')
>
>\#print(data.head())
>
>\#print(data.shape)
>
>
>
>target=data.iloc[:, 13] #读取目标值
>
>features=data.drop(data.columns[13],axis=1) #读取特征值
>
>\#print(features.head())
>
>\#print(target.head())
>
>
>
>\#标准化处理
>
>from sklearn import preprocessing
>
>input= preprocessing.StandardScaler().fit_transform(features)
>
>\#print(input)
>
>\#张量
>
>input_tensor= torch.tensor(input, dtype=torch.float)
>
>target_tensor=torch.tensor(target, dtype=torch.float)
>
>\#print(input_tensor)
>
>\#print(target_tensor)
>
>\#改变形状
>
>target_tensor = target_tensor.unsqueeze(-1)
>
>
>
>
>
>\# 构建
>
>class Net(nn.Module):
>
>  def __init__(self):
>
>​    super(Net, self).__init__()
>
>​    self.linear = nn.Linear(13, 1)#13个特征值，输出一个值
>
>  def forward(self,x):#前向传播
>
>​    x=self.linear(x)
>
>​    return x
>
>
>
>\# 实例化
>
>ne = Net()
>
>\#print(ne)
>
>
>
>
>
>criterion=nn.MSELoss()#MSE损失函数
>
>optimizer=torch.optim.SGD(ne.parameters(), lr=0.01) #SGD优化器
>
>
>
>\# 训练
>
>losses=[]
>
>for epoch in range(500):
>
>  optimizer.zero_grad() #清除数据
>
>  output = ne(input_tensor) #前向传播
>
>  loss = criterion(output, target_tensor) #损失
>
>  loss.backward() #反向传播
>
>  optimizer.step() #更新参数
>
>
>
>  losses.append(loss.item())  
>
>
>
>\# 绘图
>
>plt.plot(range(500), losses)
>
>plt.xlabel('epoch')
>
>plt.ylabel('losses')
>
>plt.show()



# END