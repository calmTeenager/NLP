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

if __name__ == '__main__':
    dataset='flickr8k'
    karpathy_json_path = "./archive/dataset_flickr8k.json"  # 读取json文件
    image_folder = "./archive/images/"  # 图片文件夹
    output_folder = "./data/%s" % dataset  # 保存处理结果的文件夹
    # 调用 create_dataset() 函数处理数据集
    create_dataset(karpathy_json_path=karpathy_json_path, image_folder=image_folder, output_folder=output_folder)
    with open('./data/flickr8k/vocab.json', 'r') as f:
        vocab = json.load(f)
    vocab_idx2word = {idx: word for word, idx in vocab.items()}
    with open('./data/flickr8k/test_data.json', 'r') as f:
        data = json.load(f)
    # 调用 mktrainval 函数
    train_loder, val_loder, test_loder = mktrainval(data_dir="./data/flickr8k/", vocab_path='./data/flickr8k/vocab.json',
                                                    batch_size=64, workers=4)