import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

#torch.autograd.set_detect_anomaly(True)
#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 5000#每一批的样本数量
learning_rate = 0.01#学习率
num_epochs = 200#训练次数

# 分为训练集，测试集
trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True), #transform=transform)
trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True)#, transform=transform)
testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

#构建神经网络
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        self.linear1 = nn.Linear(3072, 128)#图像为3*32*32
        self.relu=nn.ReLU()#非线性层
        self.linear2 = nn.Linear(128,32)
        self.linear3 = nn.Linear(32,10)#最终分为十类
        
#前向传播
    def forward(self, x):
        x=x.view(x.size(0),-1)#展平数据
        x=self.linear1(x)
        x=self.relu(x)
        x=self.linear2(x)
        x=self.linear3(x)
        return x
        

#实例化
model = Network()

criterion =torch.nn.CrossEntropyLoss ()#分类交叉熵损失函数
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate) #随机梯度下降

#创建空列表，储存数据
losses=[]
accuracies=[]

#训练
def train():
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0 
        total = 0
        for i, data in enumerate(trainloader, 0):
            inputs, labels = data
            
            inputs_tensor=torch.tensor(inputs,dtype=torch.float)
            labels_tensor=torch.tensor(labels,dtype=torch.long)
            #inputs_tensor, labels = inputs_tensor.to(device), labels.to(device)
            #print(inputs_tensor.shape)
            #print(labels_tensor.shape)

            optimizer.zero_grad()#清零

            outputs = model(inputs_tensor)
            loss = criterion(outputs, labels_tensor)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        accuracy = 100 * correct / total
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(trainloader):.4f}, Accuracy: {accuracy:.2f}%')
        accuracy_tensor=torch.tensor(accuracy,dtype=torch.float)
        losses.append(loss.item())
        accuracies.append(accuracy_tensor.item())
        
#测试
def test():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in testloader:
            images, labels = data
            images_tensor=torch.tensor(images,dtype=torch.float)
            labels_tensor=torch.tensor(labels,dtype=torch.long)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f'Accuracy of the model on the 10000 test images: {accuracy:.2f}%')
   
    
if __name__ == "__main__":
    train()
    test()

# 绘图
import matplotlib.pyplot as plt
plt.plot(range(200), accuracies)
plt.xlabel('epoch')
plt.ylabel('accuracies')
plt.show()

plt.plot(range(200), losses)
plt.xlabel('epoch')
plt.ylabel('losses')
plt.show()
