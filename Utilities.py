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
import csv
import pandas as pd
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
import torch.nn.functional as F

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 函数: 绘制学习曲线
def learning_curve_plot(save_path, csv_path, show=False, image_name="Learning Curve"):
    # 读取csv数据
    loss_data = pd.read_csv(csv_path)
    epoch = loss_data['Epoch']
    train_loss = loss_data['Train Loss']
    valid_loss = loss_data['Validation Loss']
    # 作图
    plt.rcParams['font.family'] = 'Times New Roman'
    fig, ax = plt.subplots()
    ax.plot(epoch, train_loss, linewidth=1.5, label="Training", color='blue')
    ax.plot(epoch, valid_loss, linewidth=1.5, label="Validation", color='red')
    # 横轴、纵轴标题
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss Value")
    # 图例位置
    ax.legend(loc='upper right')
    # 网格
    ax.grid(True, linestyle='--')
    # 标题
    plt.title('Learning Curve')
    # 自动调整子图的布局
    fig.tight_layout()
    # 保存与展示
    plt.savefig(os.path.join(save_path, f"{image_name}.png"), dpi=600)
    plt.savefig(os.path.join(save_path, f"{image_name}.svg"), format="svg", dpi=300)
    if show == True:
        plt.show()
    plt.close(fig)

# 函数：设置随机数种子，方便结果复现
def seed_torch(seed=99):
    """
    Sets random seeds for Python, NumPy, and PyTorch to enable
    reproducible experiments by disabling hash randomization.
    """
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) #  To disable hash randomization for reproducible experiments.
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# 函数：自定义交叉熵损失函数，掩盖无意义部分
def masked_cross_entropy_loss(logits, target, pad_token_idx, eos_token_idx):
    """
    自定义损失函数，忽略填充和结束标记后的损失。

    :param logits: 模型输出的logits，形状为 (batch_size, seq_len, vocab_size)
    :param target: 目标序列，形状为 (batch_size, seq_len)
    :param pad_token_idx: 填充标记的索引
    :param eos_token_idx: 结束标记的索引
    :return: 计算的损失
    """
    # 展平 logits 和 target
    logits = logits.view(-1, logits.size(-1))
    target = target.view(-1)

    # 创建掩码，忽略填充和结束标记之后的损失
    mask = (target != pad_token_idx) & (target != eos_token_idx)

    # 只计算有效位置的损失
    logits = logits[mask]
    target = target[mask]

    return F.cross_entropy(logits, target)

# 函数：计算评价指标
def calculate_bleu(model, data_loader, vocab, start_token, end_token, max_length):
    predict_times = [] # 记录推理时间
    model.eval()
    with torch.no_grad():
        for images, captions, _ in tqdm(data_loader, desc="Calculating BLEU"):
            images, captions = images.to(device), captions.to(device)  # (64,3,224,224) (batch_size=64, seq_length=32)
            features = model[0](images)  # (batch_size, feature_dim)
            features = features.unsqueeze(1).repeat(1, captions.size(1), 1)  # (batch_size, seq_length, feature_dim)
            tgt_mask = nn.Transformer.generate_square_subsequent_mask(captions.size()[1])
            outputs = model[1](features, captions[:, :], tgt_mask=tgt_mask)
            generated_caption = torch.argmax(outputs, dim=2)

            generated_caption = generated_caption[:,:captions.size(1)]
            generated_caption = replace_after_value(generated_caption, end_token)
            reference_caption = captions
            # 计算BLEU
            smoothie = SmoothingFunction().method4
            bleu_scores = {i: [] for i in range(1, 5)}
            for j in range(1, 5):
                for ref, gen in zip(reference_caption, generated_caption):
                    score = cal_bleu_batch([ref], gen)
                    bleu_scores[j].append(score)
                #bleu_score = sentence_bleu([reference_caption], generated_caption, weights=[1 / j] * j, smoothing_function=smoothie)
                #bleu_scores[j].append(bleu_score)
    avg_bleu_scores = {i: np.mean(scores) for i, scores in bleu_scores.items()}
    predict_time = sum(predict_times) / len(data_loader)
    return avg_bleu_scores, predict_time
def replace_after_value(tensor, value):
    # 获取张量的形状
    batch_size, seq_len = tensor.shape

    # 创建一个掩码，初始值为1，表示所有元素都保留
    mask = torch.ones_like(tensor, dtype=torch.bool)

    for i in range(batch_size):
        # 找到当前行中第一个出现指定值的位置
        idx = (tensor[i] == value).nonzero(as_tuple=True)[0]

        if idx.numel() > 0:  # 确保找到了指定值
            # 将指定值后面的所有元素掩盖为0
            mask[i, idx[0]+1:] = 0

    # 使用掩码更新张量，将掩码为0的元素替换为0
    tensor = tensor * mask

    return tensor
def cal_bleu_batch(reference_caption, generated_caption):
    reference = [ref.tolist() for ref in reference_caption]  # Convert to list of lists
    candidate = generated_caption.tolist()  # Convert to list
    return sentence_bleu(reference, candidate)

# 函数：将整数序列转为文本序列
def decode_caption(caption, vocab, reverse_vocab):
    # 过滤掉没有意义的标志
    caption = [item for item in caption if item not in [vocab['<pad>'], vocab['<start>'], vocab['<end>'], vocab['<unk>']]]
    words = [reverse_vocab[idx] for idx in caption]
    return ' '.join(words)

# 函数：记录数据到csv文件
def record_csv(file_name, record_list, title_list):
    if os.path.exists(file_name):
        with open(file_name, 'a', newline='') as f: # ,encoding='utf-8'
            #print("------写入信息------")
            f_csv = csv.writer(f)
            f_csv.writerow(record_list)
    else:
        with open(file_name, 'w', newline='') as f:
            #print("------创建.csv文件------")
            f_csv = csv.writer(f)
            f_csv.writerow(title_list)
            f_csv.writerow(record_list)

