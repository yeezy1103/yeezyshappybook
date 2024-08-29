# 吴恩达深度学习

## 0.1 notation

a useful convention would be to take the data associated with different training examples

|                     notation                     |                           meaning                            |
| :----------------------------------------------: | :----------------------------------------------------------: |
|                       $m$                        |               Amount of data/training example                |
|                  $n_{x}$ or $n$                  |                   输入的特征向量$x$的维度                    |
| $(x,y),x \in \mathbb{R}^{n_{x}}, y \in \{0,1\} $ |   一个样本，$x$是$n_{x}$维的特征向量，标签$y$值为$0$或$1$    |
|     $(x^{(i)},y^{(i)})...(x^{(m)},y^{(m)})$      |                第$i$个样本 和 最后一个个样本                 |
|                 $m = m_{train}$                  |                训练集的样本数 #train example                 |
|                  $m = m_{test}$                  |                 测试集的样本数 #test example                 |
|       $X = [x^{(1)},x^{(2)},...,x^{(m)}]$        | $X \in \mathbb{R}^{n_{x} \times m}$<br />这是一个$m$列，$n_{x}$行的矩阵，表示所有的训练集（列向量堆叠）的输入形式 |
|       $Y = [y^{(1)},y^{(2)},...,y^{(m)}]$        | $X \in \mathbb{R}^{1 \times m}$<br />这是一个$m$列，$1$行的矩阵，表示输出标签 |
|                       w:=                        |                            更新w                             |



## 0.2 What you'll learn

Courses in this sequence (Specialization微专业) :

1. Neural Networks and Deep Learning 神经网络和深度学习
   1. Week 1 :    Introduction
   2. Week 2 :    Basics of Neural Network programming 神经网络编程框架
   3. Week 3 :    One hidden layer Neural Networks 单隐层神经网络
   4. Week 4 :    Deep Neural Networks 多层神经网络
2. Improving Deep Neural Networks: Hyperparameter tuning, Regularization and Optimization 超参数调整、正则化、优化算法
3. Structuring your Machine Learning project 结构化你的机器学习工程
4. Convolutional Neural Networks 卷积神经网络（CNNs）
5. Natural Language Processing (NLP) : Building sequence models 自然语言处理：序列模型







