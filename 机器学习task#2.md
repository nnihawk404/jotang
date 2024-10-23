# 机器学习task#2





<img src="https://raw.githubusercontent.com/nnihawk404/images/main/ml200.jpg" alt="23020D54A8B099189F56685D139AEFD1" style="zoom:150%;" />



## 模型

1.  参数设置

   > batch_size = 5000#每一次使用的的样本数量
   >
   > learning_rate = 0.01#学习率
   >
   > num_epochs = 200#训练次数

2.  模型结构

   > self.linear1 = nn.Linear(3072, 128)#图像为3×32×32
   >
   > self.relu=nn.ReLU()#非线性层
   >
   >  self.linear2 = nn.Linear(128,32)
   >
   >  self.linear3 = nn.Linear(32,10)#最终分为十类

    输入层的维度为3072，对应图像的3×32×32

​        中间增加了一个非线性的ReLU层

   3.  损失函数和优化器

      我原本用了在task#1中用过的MSE函数😶‍🌫️，后来发现分类问题不能用这个，所以改成了分类交叉熵

      优化器使用了随机梯度下降

        4. 绘图

        绘制了两张图，y轴分别为准确率和损失值

        将200次训练中的准确率和损失值放入列表

        然后分别在两张表中呈现出来

        > losses=[]
        >
        > accuracies=[]

        > plt.plot(range(200), accuracies)
        >
        > plt.xlabel('epoch')
        >
        > plt.ylabel('accuracies')
        >
        > plt.show()
        >
        > 
        >
        > plt.plot(range(200), losses)
        >
        > plt.xlabel('epoch')
        >
        > plt.ylabel('losses')

        ## 遇到的问题

        1. 张量转换

           > transform = transforms.Compose([
           >
           >   transforms.ToTensor()
           >
           > ])

           不知道为什么这个张量的转换没有生效😑，只能手动写了几个转换

           如何又改了几次数据类型

        2.  形状转换

           在数据输入后无法运行，用了

           > x=x.view(x.size(0),-1)#展平数据

           使数据的（？，3，32，32）变成了（？，3072）
          
        3. 运算

           我直接运行之后报了不过，说什么数据有的在gpu，有的在cpu，
      
           最后是在kaggle上运行出来的
      
        ## 结果
      
        

![屏幕截图 2024-10-20 122046](https://raw.githubusercontent.com/nnihawk404/images/main/202410211508214.png)![屏幕截图 2024-10-20 122037](https://raw.githubusercontent.com/nnihawk404/images/main/屏幕截图 2024-10-20 122037.png)

Epoch [1/200], Loss: 2.3051, Accuracy: 10.43%

Epoch [2/200], Loss: 2.2948, Accuracy: 11.23%

Epoch [3/200], Loss: 2.2872, Accuracy: 12.17%

Epoch [4/200], Loss: 2.2807, Accuracy: 15.00%

Epoch [5/200], Loss: 2.2748, Accuracy: 18.37%

Epoch [6/200], Loss: 2.2690, Accuracy: 19.38%

Epoch [7/200], Loss: 2.2632, Accuracy: 20.24%

Epoch [8/200], Loss: 2.2573, Accuracy: 21.00%

Epoch [9/200], Loss: 2.2515, Accuracy: 21.46%

Epoch [10/200], Loss: 2.2454, Accuracy: 22.14%

Epoch [11/200], Loss: 2.2393, Accuracy: 22.65%

Epoch [12/200], Loss: 2.2332, Accuracy: 23.11%

Epoch [13/200], Loss: 2.2270, Accuracy: 23.47%

Epoch [14/200], Loss: 2.2205, Accuracy: 23.80%

Epoch [15/200], Loss: 2.2139, Accuracy: 24.07%

Epoch [16/200], Loss: 2.2071, Accuracy: 24.27%

Epoch [17/200], Loss: 2.2002, Accuracy: 24.55%

Epoch [18/200], Loss: 2.1931, Accuracy: 24.75%

Epoch [19/200], Loss: 2.1859, Accuracy: 24.98%

Epoch [20/200], Loss: 2.1785, Accuracy: 25.05%

Epoch [21/200], Loss: 2.1710, Accuracy: 25.24%

Epoch [22/200], Loss: 2.1634, Accuracy: 25.36%

Epoch [23/200], Loss: 2.1558, Accuracy: 25.50%

Epoch [24/200], Loss: 2.1482, Accuracy: 25.58%

Epoch [25/200], Loss: 2.1407, Accuracy: 25.80%

Epoch [26/200], Loss: 2.1332, Accuracy: 25.93%

Epoch [27/200], Loss: 2.1258, Accuracy: 26.15%

Epoch [28/200], Loss: 2.1187, Accuracy: 26.38%

Epoch [29/200], Loss: 2.1116, Accuracy: 26.49%

Epoch [30/200], Loss: 2.1046, Accuracy: 26.69%

Epoch [31/200], Loss: 2.0979, Accuracy: 26.78%

Epoch [32/200], Loss: 2.0914, Accuracy: 26.89%

Epoch [33/200], Loss: 2.0850, Accuracy: 27.13%

Epoch [34/200], Loss: 2.0789, Accuracy: 27.23%

Epoch [35/200], Loss: 2.0729, Accuracy: 27.41%

Epoch [36/200], Loss: 2.0671, Accuracy: 27.52%

Epoch [37/200], Loss: 2.0615, Accuracy: 27.68%

Epoch [38/200], Loss: 2.0562, Accuracy: 27.83%

Epoch [39/200], Loss: 2.0510, Accuracy: 28.01%

Epoch [40/200], Loss: 2.0459, Accuracy: 27.97%

Epoch [41/200], Loss: 2.0410, Accuracy: 28.28%

Epoch [42/200], Loss: 2.0363, Accuracy: 28.33%

Epoch [43/200], Loss: 2.0318, Accuracy: 28.47%

Epoch [44/200], Loss: 2.0274, Accuracy: 28.61%

Epoch [45/200], Loss: 2.0231, Accuracy: 28.76%

Epoch [46/200], Loss: 2.0190, Accuracy: 28.78%

Epoch [47/200], Loss: 2.0150, Accuracy: 28.87%

Epoch [48/200], Loss: 2.0111, Accuracy: 29.00%

Epoch [49/200], Loss: 2.0074, Accuracy: 29.02%

Epoch [50/200], Loss: 2.0037, Accuracy: 29.16%

Epoch [51/200], Loss: 2.0001, Accuracy: 29.21%

Epoch [52/200], Loss: 1.9967, Accuracy: 29.28%

Epoch [53/200], Loss: 1.9933, Accuracy: 29.39%

Epoch [54/200], Loss: 1.9900, Accuracy: 29.46%

Epoch [55/200], Loss: 1.9869, Accuracy: 29.58%

Epoch [56/200], Loss: 1.9837, Accuracy: 29.66%

Epoch [57/200], Loss: 1.9807, Accuracy: 29.79%

Epoch [58/200], Loss: 1.9777, Accuracy: 29.84%

Epoch [59/200], Loss: 1.9749, Accuracy: 29.90%

Epoch [60/200], Loss: 1.9719, Accuracy: 30.02%

Epoch [61/200], Loss: 1.9691, Accuracy: 30.10%

Epoch [62/200], Loss: 1.9665, Accuracy: 30.15%

Epoch [63/200], Loss: 1.9638, Accuracy: 30.25%

Epoch [64/200], Loss: 1.9612, Accuracy: 30.32%

Epoch [65/200], Loss: 1.9587, Accuracy: 30.48%

Epoch [66/200], Loss: 1.9562, Accuracy: 30.58%

Epoch [67/200], Loss: 1.9537, Accuracy: 30.60%

Epoch [68/200], Loss: 1.9514, Accuracy: 30.78%

Epoch [69/200], Loss: 1.9490, Accuracy: 30.80%

Epoch [70/200], Loss: 1.9466, Accuracy: 30.96%

Epoch [71/200], Loss: 1.9444, Accuracy: 30.96%

Epoch [72/200], Loss: 1.9421, Accuracy: 31.08%

Epoch [73/200], Loss: 1.9400, Accuracy: 31.11%

Epoch [74/200], Loss: 1.9378, Accuracy: 31.21%

Epoch [75/200], Loss: 1.9357, Accuracy: 31.33%

Epoch [76/200], Loss: 1.9336, Accuracy: 31.43%

Epoch [77/200], Loss: 1.9315, Accuracy: 31.54%

Epoch [78/200], Loss: 1.9296, Accuracy: 31.63%

Epoch [79/200], Loss: 1.9276, Accuracy: 31.75%

Epoch [80/200], Loss: 1.9257, Accuracy: 31.83%

Epoch [81/200], Loss: 1.9238, Accuracy: 31.85%

Epoch [82/200], Loss: 1.9220, Accuracy: 31.90%

Epoch [83/200], Loss: 1.9202, Accuracy: 31.97%

Epoch [84/200], Loss: 1.9184, Accuracy: 32.10%

Epoch [85/200], Loss: 1.9166, Accuracy: 32.12%

Epoch [86/200], Loss: 1.9148, Accuracy: 32.11%

Epoch [87/200], Loss: 1.9132, Accuracy: 32.25%

Epoch [88/200], Loss: 1.9116, Accuracy: 32.32%

Epoch [89/200], Loss: 1.9100, Accuracy: 32.35%

Epoch [90/200], Loss: 1.9082, Accuracy: 32.47%

Epoch [91/200], Loss: 1.9068, Accuracy: 32.54%

Epoch [92/200], Loss: 1.9052, Accuracy: 32.62%

Epoch [93/200], Loss: 1.9036, Accuracy: 32.71%

Epoch [94/200], Loss: 1.9021, Accuracy: 32.75%

Epoch [95/200], Loss: 1.9006, Accuracy: 32.85%

Epoch [96/200], Loss: 1.8991, Accuracy: 32.90%

Epoch [97/200], Loss: 1.8977, Accuracy: 33.00%

Epoch [98/200], Loss: 1.8963, Accuracy: 33.07%

Epoch [99/200], Loss: 1.8949, Accuracy: 33.08%

Epoch [100/200], Loss: 1.8934, Accuracy: 33.09%

Epoch [101/200], Loss: 1.8921, Accuracy: 33.21%

Epoch [102/200], Loss: 1.8909, Accuracy: 33.23%

Epoch [103/200], Loss: 1.8894, Accuracy: 33.39%

Epoch [104/200], Loss: 1.8882, Accuracy: 33.39%

Epoch [105/200], Loss: 1.8868, Accuracy: 33.51%

Epoch [106/200], Loss: 1.8855, Accuracy: 33.52%

Epoch [107/200], Loss: 1.8842, Accuracy: 33.54%

Epoch [108/200], Loss: 1.8830, Accuracy: 33.66%

Epoch [109/200], Loss: 1.8818, Accuracy: 33.69%

Epoch [110/200], Loss: 1.8805, Accuracy: 33.70%

Epoch [111/200], Loss: 1.8793, Accuracy: 33.80%

Epoch [112/200], Loss: 1.8781, Accuracy: 33.88%

Epoch [113/200], Loss: 1.8768, Accuracy: 33.95%

Epoch [114/200], Loss: 1.8757, Accuracy: 33.93%

Epoch [115/200], Loss: 1.8745, Accuracy: 34.00%

Epoch [116/200], Loss: 1.8733, Accuracy: 34.07%

Epoch [117/200], Loss: 1.8723, Accuracy: 34.09%

Epoch [118/200], Loss: 1.8711, Accuracy: 34.09%

Epoch [119/200], Loss: 1.8700, Accuracy: 34.20%

Epoch [120/200], Loss: 1.8688, Accuracy: 34.23%

Epoch [121/200], Loss: 1.8677, Accuracy: 34.26%

Epoch [122/200], Loss: 1.8665, Accuracy: 34.34%

Epoch [123/200], Loss: 1.8653, Accuracy: 34.39%

Epoch [124/200], Loss: 1.8644, Accuracy: 34.43%

Epoch [125/200], Loss: 1.8633, Accuracy: 34.38%

Epoch [126/200], Loss: 1.8622, Accuracy: 34.51%

Epoch [127/200], Loss: 1.8611, Accuracy: 34.53%

Epoch [128/200], Loss: 1.8600, Accuracy: 34.59%

Epoch [129/200], Loss: 1.8589, Accuracy: 34.61%

Epoch [130/200], Loss: 1.8578, Accuracy: 34.67%

Epoch [131/200], Loss: 1.8568, Accuracy: 34.68%

Epoch [132/200], Loss: 1.8558, Accuracy: 34.74%

Epoch [133/200], Loss: 1.8547, Accuracy: 34.77%

Epoch [134/200], Loss: 1.8536, Accuracy: 34.81%

Epoch [135/200], Loss: 1.8525, Accuracy: 34.93%

Epoch [136/200], Loss: 1.8515, Accuracy: 34.92%

Epoch [137/200], Loss: 1.8506, Accuracy: 34.93%

Epoch [138/200], Loss: 1.8495, Accuracy: 35.01%

Epoch [139/200], Loss: 1.8486, Accuracy: 35.01%

Epoch [140/200], Loss: 1.8475, Accuracy: 35.06%

Epoch [141/200], Loss: 1.8465, Accuracy: 35.08%

Epoch [142/200], Loss: 1.8454, Accuracy: 35.19%

Epoch [143/200], Loss: 1.8444, Accuracy: 35.20%

Epoch [144/200], Loss: 1.8435, Accuracy: 35.25%

Epoch [145/200], Loss: 1.8426, Accuracy: 35.26%

Epoch [146/200], Loss: 1.8416, Accuracy: 35.31%

Epoch [147/200], Loss: 1.8404, Accuracy: 35.34%

Epoch [148/200], Loss: 1.8395, Accuracy: 35.32%

Epoch [149/200], Loss: 1.8384, Accuracy: 35.44%

Epoch [150/200], Loss: 1.8375, Accuracy: 35.43%

Epoch [151/200], Loss: 1.8365, Accuracy: 35.48%

Epoch [152/200], Loss: 1.8355, Accuracy: 35.53%

Epoch [153/200], Loss: 1.8346, Accuracy: 35.63%

Epoch [154/200], Loss: 1.8336, Accuracy: 35.63%

Epoch [155/200], Loss: 1.8326, Accuracy: 35.63%

Epoch [156/200], Loss: 1.8317, Accuracy: 35.65%

Epoch [157/200], Loss: 1.8308, Accuracy: 35.66%

Epoch [158/200], Loss: 1.8299, Accuracy: 35.63%

Epoch [159/200], Loss: 1.8288, Accuracy: 35.69%

Epoch [160/200], Loss: 1.8279, Accuracy: 35.78%

Epoch [161/200], Loss: 1.8269, Accuracy: 35.77%

Epoch [162/200], Loss: 1.8262, Accuracy: 35.82%

Epoch [163/200], Loss: 1.8251, Accuracy: 35.85%

Epoch [164/200], Loss: 1.8241, Accuracy: 35.92%

Epoch [165/200], Loss: 1.8233, Accuracy: 35.94%

Epoch [166/200], Loss: 1.8225, Accuracy: 35.94%

Epoch [167/200], Loss: 1.8213, Accuracy: 35.96%

Epoch [168/200], Loss: 1.8206, Accuracy: 36.10%

Epoch [169/200], Loss: 1.8197, Accuracy: 36.14%

Epoch [170/200], Loss: 1.8186, Accuracy: 36.13%

Epoch [171/200], Loss: 1.8177, Accuracy: 36.14%

Epoch [172/200], Loss: 1.8168, Accuracy: 36.19%

Epoch [173/200], Loss: 1.8159, Accuracy: 36.24%

Epoch [174/200], Loss: 1.8152, Accuracy: 36.20%

Epoch [175/200], Loss: 1.8143, Accuracy: 36.30%

Epoch [176/200], Loss: 1.8133, Accuracy: 36.35%

Epoch [177/200], Loss: 1.8124, Accuracy: 36.36%

Epoch [178/200], Loss: 1.8113, Accuracy: 36.41%

Epoch [179/200], Loss: 1.8105, Accuracy: 36.43%

Epoch [180/200], Loss: 1.8097, Accuracy: 36.50%

Epoch [181/200], Loss: 1.8088, Accuracy: 36.45%

Epoch [182/200], Loss: 1.8079, Accuracy: 36.52%

Epoch [183/200], Loss: 1.8070, Accuracy: 36.49%

Epoch [184/200], Loss: 1.8065, Accuracy: 36.49%

Epoch [185/200], Loss: 1.8053, Accuracy: 36.60%

Epoch [186/200], Loss: 1.8044, Accuracy: 36.56%

Epoch [187/200], Loss: 1.8036, Accuracy: 36.67%

Epoch [188/200], Loss: 1.8027, Accuracy: 36.61%

Epoch [189/200], Loss: 1.8020, Accuracy: 36.64%

Epoch [190/200], Loss: 1.8010, Accuracy: 36.75%

Epoch [191/200], Loss: 1.8002, Accuracy: 36.70%

Epoch [192/200], Loss: 1.7992, Accuracy: 36.68%

Epoch [193/200], Loss: 1.7984, Accuracy: 36.81%

Epoch [194/200], Loss: 1.7975, Accuracy: 36.81%

Epoch [195/200], Loss: 1.7967, Accuracy: 36.85%

Epoch [196/200], Loss: 1.7959, Accuracy: 36.86%

Epoch [197/200], Loss: 1.7951, Accuracy: 36.87%

Epoch [198/200], Loss: 1.7941, Accuracy: 36.93%

Epoch [199/200], Loss: 1.7934, Accuracy: 36.95%

Epoch [200/200], Loss: 1.7924, Accuracy: 37.02%