
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
from torchvision.models import resnet50, resnet101
from tqdm import tqdm
import numpy as np
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import pandas as pd
import time
import csv
import torch.nn.functional as F
from torchvision.models import densenet121, mobilenet_v2, efficientnet_b0, vgg16

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 基线简化BaselineResNet
class ResNetBlock(nn.Module):
    '''
    ResNetBlock: Infrastructure of ResNet
    '''
    def __init__(self, in_channels, out_channels, stride=1):
        # Inherit the parent class ‘nn.Module’
        super(ResNetBlock, self).__init__()
        # conv1: convolutional layer (kernel_size=3, padding=1, bias=False)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        # bn1: batch normalization layer
        self.bn1 = nn.BatchNorm2d(out_channels)
        # relu: ReLU activation function
        self.relu = nn.ReLU(inplace=True)
        # conv2: convolutional layer (kernel_size=3, padding=1, bias=False)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        # bn2: batch normalization layer
        self.bn2 = nn.BatchNorm2d(out_channels)
        # identity
        self.identity = nn.Sequential()
        # if input channel isn't equal to ouput channel, make them align in order to add together
        if stride != 1 or in_channels != out_channels:
            self.identity = nn.Sequential(
                # Conv + BN
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = self.identity(x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x += identity
        x = self.relu(x)
        return x
class BaselineResNet(nn.Module):
    def __init__(self, embed_size):
        super(BaselineResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 1024, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(1024, embed_size)
        self.sigmod = nn.Sigmoid()
    def forward(self, images): # images: batch,3,224,224
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmod(x)
        return x

# 基线简化BaselineResNet_conv_1：使用1*1卷积代替自适应池化
class BaselineResNet_conv_1(nn.Module):
    def __init__(self, embed_size):
        super(BaselineResNet_conv_1, self).__init__()
        self.conv1 = nn.Conv2d(3, 256, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(256)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = nn.Sequential(
            ResNetBlock(256, 512, stride=2),
            ResNetBlock(512, 1024, stride=2),
        )
        #self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        # 改为1*1卷积 0.5850
        self.conv_1 = nn.Conv2d(1024, 4, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(14*14*4, embed_size)
        self.sigmod = nn.Sigmoid()

    def forward(self, images): # images: batch,3,224,224
        x = self.conv1(images)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x) # 64,256,56,56
        x = self.layer1(x) # 64,256,14,14
        x = self.conv_1(x) # 64,3,14,14
        x = torch.flatten(x, 1)
        x = self.fc(x)
        x = self.sigmod(x)
        return x

# DenseNet121
class DenseNet121(nn.Module):
    def __init__(self, embed_size):
        super(DenseNet121, self).__init__()
        densenet = densenet121(pretrained=False)
        modules = list(densenet.features.children())
        self.densenet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(densenet.classifier.in_features, embed_size)

    def forward(self, images):
        features = self.densenet(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# DenseNet121_conv_1：使用1*1卷积代替自适应池化
class DenseNet121_conv_1(nn.Module):
    def __init__(self, embed_size):
        super(DenseNet121_conv_1, self).__init__()
        densenet = densenet121(pretrained=True)
        modules = list(densenet.features.children())
        self.densenet = nn.Sequential(*modules)
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 改为1*1卷积
        self.conv_1 = nn.Conv2d(1024, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(8*7*7, embed_size)

    def forward(self, images):
        features = self.densenet(images) # 64,1024,7,7
        features = self.conv_1(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# Efficientnet_B0
class Efficientnet_B0(nn.Module):
    def __init__(self, embed_size):
        super(Efficientnet_B0, self).__init__()
        efficientnet = efficientnet_b0(pretrained=False)
        modules = list(efficientnet.children())[:-2]
        self.efficientnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(efficientnet.classifier[1].in_features, embed_size)

    def forward(self, images):
        features = self.efficientnet(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# Efficientnet_B0_conv_1：使用1*1卷积代替自适应池化
class Efficientnet_B0_conv_1(nn.Module):
    def __init__(self, embed_size):
        super(Efficientnet_B0_conv_1, self).__init__()
        efficientnet = efficientnet_b0(pretrained=True)
        modules = list(efficientnet.children())[:-2]
        self.efficientnet = nn.Sequential(*modules)
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 改为1*1卷积
        self.conv_1 = nn.Conv2d(1280, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(8*7*7, embed_size)

    def forward(self, images):
        features = self.efficientnet(images)
        features = self.conv_1(features) #
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# Mobilenet_V2
class Mobilenet_V2(nn.Module):
    def __init__(self, embed_size):
        super(Mobilenet_V2, self).__init__()
        mobilenet = mobilenet_v2(pretrained=False)
        modules = list(mobilenet.features.children())
        self.mobilenet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(mobilenet.classifier[1].in_features, embed_size)

    def forward(self, images):
        features = self.mobilenet(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# Mobilenet_V2_conv_1：使用1*1卷积代替自适应池化
class Mobilenet_V2_conv_1(nn.Module):
    def __init__(self, embed_size):
        super(Mobilenet_V2_conv_1, self).__init__()
        mobilenet = mobilenet_v2(pretrained=True)
        modules = list(mobilenet.features.children())
        self.mobilenet = nn.Sequential(*modules)
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 改为1*1卷积
        self.conv_1 = nn.Conv2d(1280, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(8*7*7, embed_size)

    def forward(self, images):
        features = self.mobilenet(images) # 64,1280,7,7
        features = self.conv_1(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# Resnet50
class Resnet50(nn.Module):
    def __init__(self, embed_size):
        super(Resnet50, self).__init__()
        # 加载预训练的 ResNet50 模型，并移除最后两个全连接层
        resnet = resnet50(pretrained=False)
        modules = list(resnet.children())[:-2]  # 去掉最后两个全连接层
        self.resnet = nn.Sequential(*modules)
        # 添加自适应池化层，将特征图调整为固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，将特征映射到指定的嵌入维度
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images): # images: batch,3,224,224
        # 使用 ResNet 提取特征
        features = self.resnet(images) # 64,2048,7,7
        # 自适应池化层调整特征图大小
        features = self.adaptive_pool(features) # 64,2048,1,1
        # 展平为一维向量
        features = features.view(features.size(0), -1) # 64,2048
        # 通过全连接层将特征映射到嵌入空间
        features = self.fc(features) # batch,256
        return features

# Resnet50_conv_1：使用1*1卷积代替自适应池化
class Resnet50_conv_1(nn.Module):
    def __init__(self, embed_size):
        super(Resnet50_conv_1, self).__init__()
        # 加载预训练的 ResNet50 模型，并移除最后两个全连接层
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        # 添加自适应池化层，将特征图调整为固定大小
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 改为1*1卷积
        self.conv_1 = nn.Conv2d(2048, 10, kernel_size=1, stride=1, padding=0, bias=False)
        # 全连接层，将特征映射到指定的嵌入维度
        self.fc = nn.Linear(10*7*7, embed_size)

    def forward(self, images): # images: batch,3,224,224
        # 使用 ResNet 提取特征
        features = self.resnet(images) # 64,2048,7,7
        # 自适应池化层调整特征图大小
        features = self.conv_1(features) # 64,4,7,7
        # 展平为一维向量
        features = features.view(features.size(0), -1) # 64,2048
        # 通过全连接层将特征映射到嵌入空间
        features = self.fc(features) # batch,256
        return features

# Resnet101
class Resnet101(nn.Module):
    def __init__(self, embed_size):
        super(Resnet101, self).__init__()
        resnet = resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = self.adaptive_pool(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# Resnet101_conv_1：使用1*1卷积代替自适应池化
class Resnet101_conv_1(nn.Module):
    def __init__(self, embed_size):
        super(Resnet101_conv_1, self).__init__()
        resnet = resnet101(pretrained=True)
        modules = list(resnet.children())[:-2]
        self.resnet = nn.Sequential(*modules)
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 改为1*1卷积
        self.conv_1 = nn.Conv2d(2048, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(8*7*7, embed_size)

    def forward(self, images):
        features = self.resnet(images)# 64,2048,7,7
        features = self.conv_1(features)
        features = features.view(features.size(0), -1)
        features = self.fc(features)
        return features

# VGG16
class VGG16(nn.Module):
    def __init__(self, embed_size):
        super(VGG16, self).__init__()
        vgg = vgg16(pretrained=False)
        modules = list(vgg.features.children())
        self.vgg = nn.Sequential(*modules)
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, embed_size)

    def forward(self, images):
        features = self.vgg(images) # 64,512,7,7
        features = self.adaptive_pool(features) # 64,512,1，1
        features = features.view(features.size(0), -1) # 64,512
        features = self.fc(features)
        return features

# VGG16_conv_1：使用1*1卷积代替自适应池化
class VGG16_conv_1(nn.Module):
    def __init__(self, embed_size):
        super(VGG16_conv_1, self).__init__()
        vgg = vgg16(pretrained=False)
        modules = list(vgg.features.children())
        self.vgg = nn.Sequential(*modules)
        #self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 改为1*1卷积
        self.conv_1 = nn.Conv2d(512, 8, kernel_size=1, stride=1, padding=0, bias=False)
        self.fc = nn.Linear(8*7*7, embed_size)

    def forward(self, images):
        features = self.vgg(images) # 64,512,7,7
        features = self.conv_1(features) # 64,512,7，7
        features = features.view(features.size(0), -1) # 64,512
        features = self.fc(features)
        return features

# Transformer 解码器
class TransformerDecoder(nn.Module):
    def __init__(self, embed_size, vocab_size, num_layers, num_heads, dropout=0.1):
        super(TransformerDecoder, self).__init__()
        # 初始化词嵌入层和位置嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.pos_embedding = nn.Embedding(50, embed_size)
        '''
        nn.Embedding 是 PyTorch 中用于创建嵌入层（embedding layer）的类。
        嵌入层通常用于将离散的分类变量（如词汇表中的单词）映射到连续的向量空间中。
        这对于自然语言处理（NLP）任务非常有用，因为它可以将单词表示为密集的向量，从而捕捉词语的语义关系。
        词嵌入层：词汇表中有vocab_size个单词，每个单词用embed_size维向量表示
        位置嵌入层：将每个位置（最多 5000 个）映射到一个 embed_size 维的向量
        '''
        # 初始化 Transformer 编码器
        self.transformer = nn.Transformer(embed_size, num_heads, num_layers, num_layers, dropout=dropout, batch_first=True)
        '''
        使用 batch_first=True 参数时,src 和 tgt（即 features 和 captions_embed）的形状应为 (batch_size, seq_length, embed_size)
        src（features）和 tgt（captions_embed）具有相同的 batch_size 和 embed_size。
        src（features）的 seq_length 可以与 tgt（captions_embed）不同，但是需要保持形状一致。
        '''
        # 全连接层，将 Transformer 输出映射到词汇表大小
        self.fc = nn.Linear(embed_size, vocab_size)
        # 存储嵌入维度
        self.embed_size = embed_size

    # 前向传播：用于训练模型时的前向传播，它接收图像特征和对应的文本序列，输出每个时间步的预测结果
    def forward(self, features, captions):
        # features: (batch_size, feature_dim)
        # captions: (batch_size, seq_length)
        pos = torch.arange(0, captions.size(1)).unsqueeze(0).to(device)  # (1, seq_length)
        captions_embed = self.embedding(captions) + self.pos_embedding(pos)  # (batch_size, seq_length, embed_size)
        features = features.unsqueeze(1).repeat(1, captions.size(1), 1)  # (batch_size, seq_length, feature_dim)
        out = self.transformer(features, captions_embed)  # (batch_size, seq_length, embed_size)
        out = self.fc(out)  # (batch_size, seq_length, vocab_size)
        return out

    # 生成文本说明：用于在生成阶段（如测试或推理时），根据输入的图像特征逐步生成描述文本
    def generate(self, features, max_length, start_token, end_token):
        generated = torch.tensor([start_token]).unsqueeze(0).to(device)  # (1, 1)
        for _ in range(max_length):
            pos = torch.arange(0, generated.size(1)).unsqueeze(0).to(device)  # (1, seq_length)
            out = self.embedding(generated) + self.pos_embedding(pos)  # (1, seq_length, embed_size)
            features_expanded = features.unsqueeze(0).repeat(out.size(0), out.size(1), 1)  # (1, seq_length, feature_dim)
            out = self.transformer(features_expanded, out)  # (1, seq_length, embed_size)
            out = self.fc(out)  # (1, seq_length, vocab_size)
            _, next_word = torch.max(out[:, -1, :], dim=1)  # (1,)
            generated = torch.cat((generated, next_word.unsqueeze(0)), dim=1)  # (1, seq_length+1)
            if next_word.item() == end_token:
                break
        return generated.squeeze(0).tolist()