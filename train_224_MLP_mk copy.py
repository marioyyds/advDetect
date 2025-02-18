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

from tqdm import tqdm

import os

CIFAR_CKPT = './checkpoint/cifar_undercover.pth'

CIFAR_UNDERCOVER_CKPT = './checkpoint/cifar_undercover.pth'
device = 'cuda:7' if torch.cuda.is_available() else 'cpu'
device


# CIFAR 预训练模型路径
CIFAR_UNDERCOVER_CKPT = '/home/intent/advDetect/github项目/DBA-master/checkpoint/cifar_undercover.pth'

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
    val_ratio = 0.0    # 验证集比例
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

# trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_test)
# trainloader = torch.utils.data.DataLoader(trainset, batch_size=512, shuffle=True, num_workers=4)
# trainiter = iter(trainloader)

# testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
# testloader = torch.utils.data.DataLoader(testset, batch_size=512, shuffle=False, num_workers=4)
# testiter = iter(testloader)


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

print("CCCCCCCCCCCCC")
for batch_idx,(x, y) in tqdm(enumerate(trainloader),total=len(trainloader),desc="BIM"):
# for x, y in trainloader:
    print("BIM")
    x, y = x.to(device), y.to(device)
    y_pred = undercoverNet(x).argmax(dim=1)
    
    eps = 0.3
    x_adv = undercover_gradient_attacker.i_fgsm(x, y, eps=eps, alpha=1/255, iteration=int(min(eps*255 + 4, 1.25*eps*255)))
    y_pred_adv = undercoverNet(x_adv).argmax(dim=1)
    selected = (y == y_pred) & (y != y_pred_adv)
    normal_samples.append(x[selected].detach().cpu())
    adversarial_samples.append(x_adv[selected].detach().cpu())
#     break

normal_x = torch.cat(normal_samples, dim=0)
adversarial_x = torch.cat(adversarial_samples, dim=0)
normal_y = torch.zeros(normal_x.shape[0]).long()
adversarial_y = torch.ones(adversarial_x.shape[0]).long()

dba_trainloader = Data.DataLoader(Data.TensorDataset(torch.cat([normal_x, adversarial_x], dim=0),
                                           torch.cat([normal_y, adversarial_y], dim=0)), 
                                  batch_size=256, shuffle=True, num_workers=4)
dba_trainiter = iter(dba_trainloader)


#######################################
# train the mlp
epochs = 10
for i in range(epochs):
    epoch_loss = 0
    total, correct = 0, 0
    
    for batch_idx,(x, y) in tqdm(enumerate(dba_trainloader),total=len(dba_trainloader),desc="trainning"):
        optimizer.zero_grad()
        x, y = x.to(device), y.to(device)
        _, V1 = undercoverNet(x, dba=True)
        undercover_adv = undercover_gradient_attacker.fgsm(x, y, False, 1/255)
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
    accuracy = correct / total
    print(f"Epoch [{i}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {accuracy:.4f}")

###################################

# ----------------test---------------------
normal_samples, adversarial_samples = [], []
for x, y in testloader:
    x, y = x.to(device), y.to(device)
    y_pred = undercoverNet(x).argmax(dim=1)
    
    eps = 0.3
    x_adv = undercover_gradient_attacker.i_fgsm(x, y, eps=eps, alpha=1/255, iteration=int(min(eps*255 + 4, 1.25*eps*255)))
    y_pred_adv = undercoverNet(x_adv).argmax(dim=1)
    selected = (y == y_pred) & (y != y_pred_adv)
    normal_samples.append(x[selected].detach().cpu())
    adversarial_samples.append(x_adv[selected].detach().cpu())
#     break

normal_x = torch.cat(normal_samples, dim=0)
adversarial_x = torch.cat(adversarial_samples, dim=0)
normal_y = torch.zeros(normal_x.shape[0]).long()
adversarial_y = torch.ones(adversarial_x.shape[0]).long()

dba_testloader = Data.DataLoader(Data.TensorDataset(torch.cat([normal_x, adversarial_x], dim=0),
                                           torch.cat([normal_y, adversarial_y], dim=0)), 
                                  batch_size=256, shuffle=True, num_workers=4)
dba_testiter = iter(dba_testloader)



# test
total, correct = 0, 0
for x, y in dba_testloader:
    x, y = x.to(device), y.to(device)
    _, V1 = undercoverNet(x, dba=True)
    undercover_adv = undercover_gradient_attacker.fgsm(x, y, False, 1/255)
    _, V2 = undercoverNet(undercover_adv, dba=True)
    V = torch.cat([V1, V2, V1 - V2, V1 * V2], axis=-1)
    y_pred = mlp(V).argmax(dim=1)
    
    total += y.size(0)
    correct += y_pred.eq(y).sum().item()
print(correct / total)