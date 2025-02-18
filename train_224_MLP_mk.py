import numpy as np
import torch
import adversary.cw as cw
from adversary.jsma import SaliencyMapMethod
from adversary.fgsm import Attack
import torchvision
import torch.nn.functional as F
import torch.utils.data as Data
from models.model_mk import PreActResNet18, MLP
from torchvision import transforms


from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader, TensorDataset

from tqdm import tqdm
import glob

import os

CIFAR_CKPT = './checkpoint/cifar_undercover.pth'

CIFAR_UNDERCOVER_CKPT = './checkpoint/cifar_undercover.pth'
device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
# device


# CIFAR 预训练模型路径
CIFAR_UNDERCOVER_CKPT = '/home/intent/advDetect/github项目/DBA-master/checkpoint/cifar_undercover.pth'

CIFAR_CKPT_MLP='/home/intent/advDetect/github项目/DBA-master/checkpoint/cifar_undercover_mlp.pth'

CAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_bk/correct_adversarial_samples.pth'

INCAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_bk/incorrect_adversarial_samples.pth'

CAD_SAMPLE_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_bk/correct_adversarial_samples_test.pth'

INCAD_SAMPLE_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_bk/incorrect_adversarial_samples_test.pth'

FOLDER_TRAIN='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_folder/train'
FOLDER_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_folder/test'

if os.path.exists(CIFAR_UNDERCOVER_CKPT):
    # checkpoint = torch.load(CIFAR_UNDERCOVER_CKPT, map_location=torch.device(device))
    # undercoverNet.load_state_dict(checkpoint['net'])
    print("Checkpoint exsit.")
else:
    print("Checkpoint file does not exist.")


class Config:
    data_root = "/home/intent/advDetect/Caltech_256/256_ObjectCategories"  # 数据集根目录
    batch_size = 64
    num_epochs = 100
    num_classes = 257  # Caltech-256 有 257 个类别（包括 clutter 类别）
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "caltech256_best_model_100epochs.pth"
    train_ratio = 0.9  # 训练集比例
    val_ratio = 0    # 验证集比例
    test_ratio = 0.1   # 测试集比例


# 数据预处理
def get_transforms():
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }


transform_test = transforms.Compose([
    transforms.ToTensor(),
])

mlp = MLP().to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(mlp.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)


undercoverNet = PreActResNet18().to(device)
checkpoint = torch.load(CIFAR_UNDERCOVER_CKPT, map_location=torch.device(device))
undercoverNet.load_state_dict(checkpoint['net'])

undercover_gradient_attacker = Attack(undercoverNet, F.cross_entropy)


normal_samples, adversarial_samples = [], []

################直接加载对抗图片
def get_dba_advjpgloader(data_dir, batch_size=32):
    """
    加载数据并返回一个 DataLoader 对象。

    参数：
    - data_dir: 数据所在的文件夹路径。
    - batch_size: 每个批次的样本数量。

    返回：
    - DataLoader 对象。
    """
    # 定义数据预处理
    transform = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # 加载数据
    dataset = datasets.ImageFolder(data_dir, transform=transform)
    dba_trainloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 获取类别标签映射
    class_names = dataset.classes
    print(f"Classes: {class_names}")
    
    return dba_trainloader
##############


def train(dba_trainloader, epochs=10):
    if os.path.exists(CIFAR_CKPT_MLP):
        checkpoint = torch.load(CIFAR_CKPT_MLP)
        mlp.load_state_dict(checkpoint['net'])
        best_acc = checkpoint['acc']
        print(f"Loaded checkpoint with accuracy: {best_acc:.4f}")
    else:
        best_acc = 0.0
        print("No checkpoint found, starting from scratch.")

    for i in range(epochs):
        epoch_loss = 0
        total, correct = 0, 0
        
        for batch_idx, (x, y) in tqdm(enumerate(dba_trainloader), total=len(dba_trainloader), desc="training"):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y1, V1 = undercoverNet(x, dba=True)
            ylable = y1.argmax(dim=1)
            undercover_adv = undercover_gradient_attacker.fgsm(x, ylable, False, 1/255)
            _, V2 = undercoverNet(undercover_adv, dba=True)
            V = torch.cat([V1, V2, V1 - V2, V1 * V2], axis=-1)
            y_pred = mlp(V)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # 累计损失
            epoch_loss += loss.item()

            # 计算准确率
            total += y.size(0)
            correct += y_pred.argmax(dim=1).eq(y).sum().item()

        # 计算并显示每个 epoch 的平均损失和准确率
        avg_loss = epoch_loss / len(dba_trainloader)
        acc = correct / total
        print(f"Epoch [{i + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        if acc > best_acc:
            best_acc = acc
            state = {
                'net': mlp.state_dict(),
                'acc': acc,
                'epoch': i + 1,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, CIFAR_CKPT_MLP)

# test
def test(dba_testloader):
    total, correct = 0, 0
    for x, y in dba_testloader:
        print("TTTTTTTTTTTT",y.size(),x.size(),y)
        undercoverNet.load_state_dict(checkpoint['net'])

        undercoverNet.eval()

        x, y = x.to(device), y.to(device)
        y1, V1 = undercoverNet(x, dba=True)
        ylable=y1.argmax(dim=1)
        # print("TTTTTTTTTTTT",y.size(),x.size(),y,y1,ylable)
        undercover_adv = undercover_gradient_attacker.fgsm(x, ylable, False, 1/255)
        _, V2 = undercoverNet(undercover_adv, dba=True)
        V = torch.cat([V1, V2, V1 - V2, V1 * V2], axis=-1)
        y_pred = mlp(V).argmax(dim=1)
        
        total += y.size(0)
        correct += y_pred.eq(y).sum().item()
    print(correct / total)

if __name__ == '__main__':
    # device = Config.device
    if os.path.exists(INCAD_SAMPLE):
        print("AAAAAAAAAAAA")


    # 数据加载
    data_transforms = get_transforms()

    # 加载完整数据集
    full_dataset = datasets.ImageFolder(Config.data_root, transform=data_transforms["train"])

    # 划分数据集
    train_size = int(Config.train_ratio * len(full_dataset))
    val_size = int(Config.val_ratio * len(full_dataset))
    test_size = len(full_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        full_dataset, [train_size, val_size, test_size]
    )

    # 应用不同的数据增强
    train_dataset.dataset.transform = data_transforms["train"]
    val_dataset.dataset.transform = data_transforms["val"]
    test_dataset.dataset.transform = data_transforms["test"]

    # 创建数据加载器
    dataloaders = {
        "train": DataLoader(train_dataset, batch_size=Config.batch_size,
                          shuffle=True, num_workers=4, pin_memory=True),
        "val": DataLoader(val_dataset, batch_size=Config.batch_size,
                        shuffle=False, num_workers=4, pin_memory=True),
        "test": DataLoader(test_dataset, batch_size=Config.batch_size,
                         shuffle=False, num_workers=4, pin_memory=True)
    }

    trainloader=dataloaders["train"]
    testloader=dataloaders["test"]

    #########################
    undercover_gradient_attacker = Attack(undercoverNet, F.cross_entropy)

    ##########################
    # construct bim adversarial samples
    # --------------------train---------------------
    normal_samples, adversarial_samples = [], []

    dba_trainloader=get_dba_advjpgloader(FOLDER_TRAIN,64)
    train(dba_trainloader)

    dba_testloader=get_dba_advjpgloader(FOLDER_TEST,64)
    test(dba_testloader)
