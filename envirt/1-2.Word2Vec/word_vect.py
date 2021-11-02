import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import functional as F
x = torch.Tensor([[1,2,3],[1,2,4]])
embedding_size = 2
word_dict = [1,2,3]
class word2vec(nn.Module):
    def __init__(self):
        super(word2vec, self).__init__()
        self.W = nn.Linear(len(word_dict), embedding_size, bias=False)
        # self.WT = nn.Linear(embedding_size,len(word_dict),bias=False)
        self.WT = nn.Linear(embedding_size, len(word_dict), bias=False)
    def forward(self, X):
        X = self.W(X)
        X = self.WT(X)
        return X

model = word2vec()
# 输入为 one hot 的 几个词，如窗口为1则是中心词的左右两个词向量，来预测一个中心词
criterion = nn.CrossEntropyLoss()
print(list(model.parameters()))
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 1000
for i in range(epochs):
    optimizer.zero_grad()
    x = x
    target = torch.tensor([1,2])
    out = model(x)
    loss = criterion(out, target)
    loss.backward()
    optimizer.step()
    if i % 200 == 0:
        print(loss)

target = torch.tensor([0, 2, 3, 1, 4]) # 标签 这里还有一个torch.tensor与torch.Tensor的知识点https://blog.csdn.net/weixin_40607008/article/details/107348254
one_hot = F.one_hot(target).float()