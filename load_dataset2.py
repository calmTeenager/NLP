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
import pandas as pd
import time
import csv

# 函数：创建数据集
def create_dataset(karpathy_json_path, image_folder, output_folder, dataset='flickr8k', captions_per_image=5, min_word_count=5, max_len=30):
    """
    参数：
        dataset:数据集名称
        captions_per_image:每张图片对应的文本描述数
        min_word_count:仅考虑在数据集中（除测试集外）出现5次的词
        max_len:文本描述包含的最大单词数。如果文本描述超过该值则截断
    输出：
        一个词典文件：vocab.json
        三个数据集文件：train_data.json val_data.json test_data.json
    """

    # 读取数据集文本描述的json文件
    with open(file=karpathy_json_path, mode="r") as j:
        data = json.load(fp=j)

    # 初始化三个字典结构，分别用于存储图片路径、图片描述和词频统计
    image_paths = defaultdict(
        list)  # collections.defaultdict() 参考：https://zhuanlan.zhihu.com/p/345741967 ; https://blog.csdn.net/sinat_38682860/article/details/112878842
    image_captions = defaultdict(list)
    vocab = Counter()  # collections.Counter() 主要功能：可以支持方便、快速的计数，将元素数量统计，然后计数并返回一个字典，键为元素，值为元素个数。 参考：https://blog.csdn.net/chl183/article/details/106956807

    # 遍历每张图片的数据，读取其描述，并根据描述的长度进行过滤
    for img in data["images"]:  # 读取每张图片
        split = img["split"]  # split：该图片文本描述的编号 len(spilt)==5
        captions = []
        for c in img["sentences"]:  # 读取图片的文本描述
            # 更新词频，测试集在训练过程中未见到数据集
            if split != "test":  # 只读取train/val
                vocab.update(c['tokens'])  # 更新词表 这里的c['tokens']是一个列表，在这里对这列表中每个元素，也就是每个词使其在词表中的出现个数加一 参考：https://blog.csdn.net/ljr_123/article/details/106209564 ; https://blog.csdn.net/ljr_123/article/details/106209564
            # 不统计超过最大长度限制的词
            if len(c["tokens"]) <= max_len:
                captions.append(c["tokens"])  # 如果每个句子的单词数都不大与max_len，则len(captions)+=5
        if len(captions) == 0:  # 万一有超过的也得往下循环
            continue
        path = os.path.join(image_folder, img[
            'filename'])  # 读取图片路径:"./images/img['filename']" 这里img['filename']为图片名字 os.path.join()函数用于路径拼接文件路径，可以传入多个路径 参考：https://blog.csdn.net/swan777/article/details/89040802
        image_paths[split].append(path)  # 保存每张图片路径
        image_captions[split].append(captions)  # 保存每张图片对应描述文本

    """
    执行完以上步骤后得到了：vocab, image_captions, image_paths
    vocab 为一个字典结构，key为各个出现的词; value为这个词出现的个数
    image_captions 为一个字典结构，key为"train","val"; val为列表，表中元素为一个个文本描述的列表
    image_paths 为一个字典结构，key为"train","val"; val为列表，表中元素为图片路径的字符串

    可运行以下代码验证：
    print(vocab)
    print(image_paths["train"][1])
    print(image_captions["train"][1])
    """

    # 创造词典，增加占位符<pad>，未登录词标识符<unk>，句子首尾标识符<start>和<end>
    words = [w for w in vocab.keys() if vocab[w] > min_word_count]
    vocab = {k: v + 1 for v, k in enumerate(words)}
    vocab['<pad>'] = 0
    vocab['<unk>'] = len(vocab)
    vocab['<start>'] = len(vocab)
    vocab['<end>'] = len(vocab)

    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    # 储存词典
    with open(os.path.join(output_folder, 'vocab.json'), "w") as fw:
        json.dump(vocab, fw)

    # 整理数据集：遍历图片路径和描述，根据描述数量补足或打乱描述，编码后保存到相应的 JSON 文件中
    dataset_sizes = {}
    for split in image_paths:  # 只会循环三次 split = "train" 、 split = "val" 和 split = "test"
        imgpaths = image_paths[split]  # type(imgpaths)=list
        imcaps = image_captions[split]  # type(imcaps)=list
        enc_captions = []

        for i, path in enumerate(imgpaths):

            # 合法性检测，检查图像时候可以被解析
            img = Image.open(path)  # 参考：https://blog.csdn.net/weixin_43723625/article/details/108158375

            # 如果图像对应的描述数量不足，则补足
            if len(imcaps[i]) < captions_per_image:
                filled_num = captions_per_image - len(imcaps[i])
                captions = imcaps[i] + [random.choice(imcaps[i]) for _ in range(0, filled_num)]
            else:
                captions = random.sample(imcaps[i],
                                         k=captions_per_image)  # 打乱文本描述 参考：https://blog.csdn.net/qq_37281522/article/details/85032470

            assert len(captions) == captions_per_image

            for j, c in enumerate(captions):
                # 对文本描述进行编码
                enc_c = [vocab['<start>']] + [vocab.get(word, vocab['<unk>']) for word in c] + [vocab["<end>"]]
                enc_captions.append(enc_c)

        assert len(imgpaths) * captions_per_image == len(enc_captions)

        data = {"IMAGES": imgpaths,
                "CAPTIONS": enc_captions}

        # 储存训练集，验证集，测试集
        with open(os.path.join(output_folder, split + "_data.json"), 'w') as fw:
            json.dump(data, fw)
        # 记录数据集大小
        dataset_sizes[split] = len(imgpaths)
        print(dataset_sizes)

# 类：继承自 PyTorch 的 Dataset （在 Pytorch 中定义数据集十分简单，仅继承 torch.utils.data.Dataset 类，并实现 __getitem__ 和__len__ 两个函数即可）
class ImageTextDataset(Dataset):
    """
    Pytorch 数据类，用于 P有torch Dataloader 来按批次产生数据
    """

    def __init__(self, dataset_path, vocab_path, split, captions_per_image=5, max_len=30, transform=None):
        """
        参数：
            dataset_path: json 格式数据文件路径
            vocab_path: json 格式字典文件路径
            split: "tarin", "val", "test"
            captions_per_image: 每张图片对应的文本描述数
            max_len: 文本描述最大单词量
            transform: 图像预处理方法
        """
        self.split = split
        assert self.split in {"train", "val", "test"}  # assert的应用 参考：https://blog.csdn.net/TeFuirnever/article/details/88883859
        self.cpi = captions_per_image
        self.max_len = max_len

        # 载入图像
        with open(dataset_path, "r") as f:
            self.data = json.load(f)

        # 载入词典
        with open(vocab_path, "r") as f:
            self.vocab = json.load(f)

        # 图像预处理流程
        self.transform = transform

        # 数据量
        self.dataset_size = len(self.data["CAPTIONS"])

    def __getitem__(self, i):
        # 第 i 个样本描述对应第 (i // captions_per_image) 张图片
        img = Image.open(self.data['IMAGES'][i // self.cpi]).convert(
            "RGB")  # 参考：https://blog.csdn.net/nienelong3319/article/details/105458742

        # 如歌有图像预处理流程，进行图像预处理
        if self.transform is not None:
            img = self.transform(img)

        caplen = len(self.data["CAPTIONS"][i])
        pad_caps = [self.vocab['<pad>']] * (self.max_len + 2 - caplen)
        caption = torch.LongTensor(
            self.data["CAPTIONS"][i] + pad_caps)  # 类型转换 参考：https://blog.csdn.net/qq_45138078/article/details/131557441

        return img, caption, caplen

    def __len__(self):
        return self.dataset_size

# 函数：创建训练集、验证集和测试集的 DataLoader （应用数据增强方法）
def mktrainval(data_dir, vocab_path, batch_size, workers=4):
    train_tx = transforms.Compose([
        transforms.Resize(256),  # 缩放
        transforms.RandomCrop(224),  # 随机裁剪
        transforms.ToTensor(),  # 用于对载入的图片数据进行类型转换，将图片数据转换成Tensor数据类型的变量
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 标准化，这里的均值和方差为在ImageNet数据集上抽样计算出来的
    ])

    val_tx = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),  # 中心裁剪
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_set = ImageTextDataset(dataset_path=os.path.join(data_dir, "train_data.json"), vocab_path=vocab_path,
                                 split="train", transform=train_tx)
    vaild_set = ImageTextDataset(dataset_path=os.path.join(data_dir, "val_data.json"), vocab_path=vocab_path,
                                 split="val", transform=val_tx)
    test_set = ImageTextDataset(dataset_path=os.path.join(data_dir, "test_data.json"), vocab_path=vocab_path,
                                split="test", transform=val_tx)

    train_loder = torch.utils.data.DataLoader(
        dataset=train_set, batch_size=batch_size, shuffle=True,
        num_workers=workers, pin_memory=True
    )  # 参考：https://blog.csdn.net/rocketeerLi/article/details/90523649 ; https://blog.csdn.net/zfhsfdhdfajhsr/article/details/116836851

    val_loder = torch.utils.data.DataLoader(
        dataset=vaild_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False
    )  # 验证集和测试集不需要打乱数据顺序：shuffer = False

    test_loder = torch.utils.data.DataLoader(
        dataset=test_set, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True, drop_last=False
    )

    return train_loder, val_loder, test_loder

# 类：CNN 特征提取器
class CNNFeatureExtractor(nn.Module):
    def __init__(self, embed_size):
        super(CNNFeatureExtractor, self).__init__()
        # 加载预训练的 ResNet50 模型，并移除最后两个全连接层
        resnet = resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # 去掉最后两个全连接层
        self.resnet = nn.Sequential(*modules)
        # 添加自适应池化层，将特征图调整为固定大小
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        # 全连接层，将特征映射到指定的嵌入维度
        self.fc = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images): # images: batch,3,224,224
        # 禁用梯度计算
        #with torch.no_grad():
        # 使用 ResNet 提取特征
        features = self.resnet(images) # 64,2048,7,7
        # 自适应池化层调整特征图大小
        features = self.adaptive_pool(features) # 64,2048,1,1
        # 展平为一维向量
        features = features.view(features.size(0), -1) # 64,2048
        # 通过全连接层将特征映射到嵌入空间
        features = self.fc(features) # batch,256
        return features

# 类：Transformer 解码器
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

# 函数：模型训练
def train(model, train_loader, criterion, optimizer, vocab_size):
    model.train()
    total_loss = 0
    for images, captions, _ in tqdm(train_loader, desc="Training"):
        images, captions = images.to(device), captions.to(device)
        optimizer.zero_grad()
        features = model[0](images)  # (batch_size, feature_dim)
        outputs = model[1](features, captions[:, :-1])
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
            images, captions = images.to(device), captions.to(device)
            features = model[0](images)  # (batch_size, feature_dim)
            outputs = model[1](features, captions[:, :-1])
            loss = criterion(outputs.view(-1, vocab_size), captions[:, 1:].reshape(-1))
            total_loss += loss.item()
    return total_loss / len(val_loader)

# 函数：计算评价指标
def calculate_bleu(model, data_loader, vocab, start_token, end_token, max_length):
    predict_times = [] # 记录推理时间
    model.eval()
    smoothie = SmoothingFunction().method4
    bleu_scores = {i: [] for i in range(1, 5)}
    with torch.no_grad():
        for images, captions, _ in tqdm(data_loader, desc="Calculating BLEU"):
            images = images.to(device)
            features = model[0](images)  # (batch_size, feature_dim)
            for i in range(captions.size(0)):
                start_time = time.time()
                generated_caption = model[1].generate(features[i], max_length, start_token, end_token)
                end_time = time.time()
                predict_times.append(end_time - start_time)
                reference_caption = captions[i].tolist()
                for j in range(1, 5):
                    bleu_score = sentence_bleu([reference_caption], generated_caption, weights=[1 / j] * j, smoothing_function=smoothie)
                    bleu_scores[j].append(bleu_score)
    avg_bleu_scores = {i: np.mean(scores) for i, scores in bleu_scores.items()}
    predict_time = sum(predict_times) / 1000
    return avg_bleu_scores, predict_time

# 函数：将整数序列转为文本序列
def decode_caption(caption, vocab, reverse_vocab):
    # 过滤掉没有意义的标志
    caption = [item for item in caption if item not in [vocab['<pad>'], vocab['<start>'], vocab['<end>'], vocab['<unk>']]]
    words = [reverse_vocab[idx] for idx in caption]
    return ' '.join(words)

# 函数：显示图像
def show_imgAcap(tensor, pre_caption, real_caption, save_path=None,show=False,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    # 处理caption
    pre_caption = decode_caption(pre_caption, vocab, reverse_vocab) # vocab, reverse_vocab是全局的
    real_caption = decode_caption(real_caption, vocab, reverse_vocab)
    # 处理图像
    image = tensor.clone()
    image = image.numpy().transpose((1, 2, 0))
    mean = np.array(mean)
    std = np.array(std)
    image = std * image + mean
    image = np.clip(image, 0, 1)
    # 显示图像
    plt.imshow(image)
    plt.title(pre_caption + '\n' + real_caption)
    plt.axis('off')  # Turn off axis numbers and ticks
    if save_path is not None:
        plt.savefig(os.path.join(save_path) + ".png", dpi=600)
    if show == True:
        plt.show()

# 函数：展示结果
def display_result(model, data_loader, vocab, start_token, end_token, max_length):
    model.eval()
    with torch.no_grad():
        for images, captions, _ in tqdm(data_loader, desc="Calculating BLEU"):
            images = images.to(device)
            features = model[0](images)  # (batch_size, feature_dim)
            for i in range(captions.size(0)):
                generated_caption = model[1].generate(features[i], max_length, start_token, end_token)
                reference_caption = captions[i].tolist()
                # 调用show_imgAcap函数
                os.makedirs(os.path.join(result_dir, "show"), exist_ok=True)
                show_imgAcap(tensor=images[i].detach().cpu(), pre_caption=generated_caption,
                             real_caption=reference_caption, save_path=os.path.join(result_dir, "show", str(i)))
            # 跳出循环，不画太多
            break

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

if __name__ == '__main__':
    # 数据集处理完成，不需要多次处理
    '''
    dataset='flickr8k'
    karpathy_json_path = "./archive/dataset_flickr8k.json"  # 读取json文件
    image_folder = "./archive/images/"  # 图片文件夹
    output_folder = "./data/%s" % dataset  # 保存处理结果的文件夹
    # 调用 create_dataset() 函数处理数据集
    create_dataset(karpathy_json_path=karpathy_json_path, image_folder=image_folder, output_folder=output_folder)
    '''
    # 看看效果
    with open('./data/flickr8k/vocab.json', 'r') as f:
        vocab = json.load(f)
        reverse_vocab = {value: key for key, value in vocab.items()} # 反向词表
    vocab_idx2word = {idx: word for word, idx in vocab.items()}
    with open('./data/flickr8k/test_data.json', 'r') as f:
        data = json.load(f)
    # 调用 mktrainval 函数
    train_loder, val_loder, test_loder = mktrainval(data_dir="./data/flickr8k/", vocab_path='./data/flickr8k/vocab.json',
                                                    batch_size=64, workers=4)
    '''
    content_img = Image.open(data['IMAGES'][300])
    plt.imshow(content_img)
    
    print(len(data))
    print(len(data['IMAGES']))
    print(len(data["CAPTIONS"]))
    
    for i in range(5):
        word_indeces = data['CAPTIONS'][300 * 5 + i]
        print(''.join([vocab_idx2word[idx] for idx in word_indeces]))
    '''
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 设置超参数
    embed_size = 256 # 表示嵌入层输出的维度大小(图像特征向量或词嵌入向量的维度)
    num_layers = 2
    num_heads = 8
    dropout = 0.1
    learning_rate = 1e-4
    num_epochs = 50
    vocab_size = len(vocab)
    start_token = vocab['<start>']
    end_token = vocab['<end>']
    max_length = 30
    seed = 1
    result_dir = "result"
    train_before = None # 断点继续训练，相应文件夹目录，没有则为 None
    epoch_before = 2 # 轮数，这里指的是.pth文件的数字部分

    # 创建结果输出目录
    current_time = time.strftime("%Y%m%d_%H%M%S")
    result_dir = result_dir + current_time
    os.makedirs(result_dir, exist_ok=True)

    # 设置随机数种子
    seed_torch(seed=seed)
    # 创建模型
    if train_before is not None:
        cnn = torch.load(os.path.join(train_before, "encoder_" + str(epoch_before) + ".pth"))
        transformer = torch.load(os.path.join(train_before, "transformer_" + str(epoch_before) + ".pth"))
    else:
        cnn = CNNFeatureExtractor(embed_size).to(device)
        transformer = TransformerDecoder(embed_size, vocab_size, num_layers, num_heads, dropout).to(device)
    model = nn.ModuleList([cnn, transformer])
    # 损失函数
    criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
    # 优化器
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # 训练和验证模型
    train_losses = [] # 训练过程中记录每个epoch的训练损失
    val_losses = [] # 训练过程中记录每个epoch的验证损失
    train_times = [] # 训练耗时
    val_times = [] # 验证耗时
    if train_before is not None:
        start_epoch = epoch_before + 1
    else:
        start_epoch = 0
    for epoch in range(start_epoch, num_epochs):
        # 调用train函数进行训练，并且记录耗时
        start_time = time.time()
        train_loss = train(model, train_loder, criterion, optimizer, vocab_size)
        end_time = time.time()
        train_time = end_time - start_time
        # 调用evaluate函数进行训练，并且记录耗时
        start_time = time.time()
        val_loss = evaluate(model, val_loder, criterion, vocab_size)
        end_time = time.time()
        val_time = end_time - start_time
        # 保存模型
        torch.save(cnn, os.path.join(result_dir, "encoder_" + str(epoch) + ".pth"))
        torch.save(transformer, os.path.join(result_dir, "transformer_" + str(epoch) + ".pth"))
        # 打印输出
        print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}')
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_times.append(train_time)
        val_times.append(val_time)
    # 保存损失数据到CSV文件
    epochs = [i for i in range(start_epoch, num_epochs)]
    loss_data = {'Epoch': epochs, 'Train Loss': train_losses, 'Validation Loss': val_losses}
    df = pd.DataFrame(loss_data)
    df.to_csv(os.path.join(result_dir, "loss_data.csv"), index=False)
    # 绘制学习曲线
    learning_curve_plot(save_path=result_dir,
                        csv_path=os.path.join(result_dir, "loss_data.csv"),
                        show=False, image_name="Learning Curve")
    # 计算训练、验证总耗时
    train_all_time = sum(train_times)
    val_all_time = sum(val_times)

    # 展示结果
    display_result(model, data_loader=test_loder, vocab=vocab, start_token=start_token, end_token=end_token, max_length=max_length)
    # 调用calculate_bleu函数计算BLEU评分
    bleu_scores, predict_time = calculate_bleu(model, test_loder, vocab, start_token, end_token, max_length)
    for i in range(1, 5):
        print(f'BLEU-{i} Score: {bleu_scores[i]:.4f}')

    # 记录关键信息到csv文件中
    title_list = ["time",
                  "BLEU-1", "BLEU-2", "BLEU-3", "BLEU-4",
                  "Train Loss", "Validation Loss",
                  "train_times", "val_times", "predict_time"]
    record_list = [current_time,
                   bleu_scores[1], bleu_scores[2], bleu_scores[3], bleu_scores[4],
                   train_losses[-1], val_losses[-1],
                   train_all_time, val_all_time, predict_time]
    record_csv(file_name=os.path.join("result", "all_result.csv"), record_list=record_list, title_list=title_list)
