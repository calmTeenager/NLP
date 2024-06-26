import json
import os
import random
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
from PIL import Image
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import json
import os
import random
from collections import Counter, defaultdict
from matplotlib import pyplot as plt
from PIL import Image
from argparse import Namespace
import numpy as np
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.models import resnet50
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from load_data import mktrainval

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 函数：模型训练
def train(model, train_loader, criterion, optimizer, vocab_size):
    '''

    model: 要训练的模型，这里是一个包含两个部分的 nn.ModuleList，第一个部分是图像特征提取器，第二个部分是 Transformer 解码器。
    train_loader: 用于训练的数据加载器
    criterion: 损失函数，这里使用交叉熵损失函数
    optimizer: 优化器，这里使用 Adam 优化器来更新模型的参数
    vocab_size: 词汇表大小，用于指定模型输出的词汇表大小
    '''
    model.train()
    total_loss = 0
    for images, captions, _ in tqdm(train_loader, desc="Training"):
        images, captions = images.to(device), captions.to(device) # 64,3,224,224 64,32
        optimizer.zero_grad()
        features = model[0](images)  # 获取图像特征 64,256
        # !确保特征的维度是正确的，captions.size(1)是描述文本的长度seq_length
        features = features.unsqueeze(1).repeat(1, captions.size(1) - 1, 1) # 64,31,256
        # 将调整后的 features 和 captions 的前 seq_length-1 个单词作为输入，去除每个序列的最后一个单词，这样模型的目标是预测每个单词的下一个单词
        outputs = model[1](features, captions[:, :-1])
        # 计算模型预测输出和真实标签之间的交叉熵损失
        loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

# 函数：模型评估
def evaluate(model, val_loader, criterion, vocab_size):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for images, captions, _ in tqdm(val_loader, desc="Evaluating"):
            images, captions = images.to(device), captions.to(device) # 64,3,224,224 64,32
            features = model[0](images) # 64,256
            features = features.repeat(1, captions.size(1) - 1, 1).view(captions.size(0), -1, features.size(1)) # 64,,31，256
            outputs = model[1](features, captions[:, :-1]) # 64,,31，2471
            # 这里使用 captions[:, :-1] 而不是 captions 的原因是，模型在评估时不需要预测序列的最后一个标记，而是根据前面的标记预测下一个标记
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

if __name__ == '__main__':
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = None # 初始化为None，导入相应的网络类，再连起来
    # 设置超参数
    embed_size = 256 # 表示嵌入层输出的维度大小(图像特征向量或词嵌入向量的维度)
    num_layers = 2
    num_heads = 8
    dropout = 0.1
    learning_rate = 1e-4
    num_epochs = 1
    max_length = 30
    with open('./data/flickr8k/vocab.json', 'r') as f:
        vocab = json.load(f)
    vocab_idx2word = {idx: word for word, idx in vocab.items()}
    vocab_size = len(vocab)
    start_token = vocab['<start>']
    end_token = vocab['<end>']
    train_loder, val_loder, test_loder = mktrainval(data_dir="./data/flickr8k/",
                                                    vocab_path='./data/flickr8k/vocab.json',
                                                    batch_size=64, workers=4)
    # 创建模型 （这里要导入相应的网络类）
    #cnn = CNNFeatureExtractor(embed_size).to(device)
    #transformer = TransformerDecoder(embed_size, vocab_size, num_layers, num_heads, dropout).to(device)
    #model = nn.ModuleList([cnn, transformer])
    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和验证模型
    for epoch in range(num_epochs):
        train_loss = train(model, train_loder, criterion, optimizer, vocab_size)
        val_loss = evaluate(model, val_loder, criterion, vocab_size)
        print(f'Epoch {epoch + 1}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
