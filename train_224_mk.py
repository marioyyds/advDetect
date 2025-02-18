import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torch.autograd import Variable
import torchvision.transforms as transforms
from models.model_mk import PreActResNet18
from adversary.fgsm import Attack

from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models

from tqdm import tqdm


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





def undercover_attack(UndercoverAttack, x, y_true, eps=1/255):
    x = Variable(x.to(device), requires_grad=True)
    y_true = Variable(y_true.to(device), requires_grad=False)
    x_adv = UndercoverAttack.fgsm(x, y_true, False, eps)
    return x_adv


def train(epochs,train_dataloader):
    print('==> Preparing data..',len(train_dataloader))
    # transform_train = transforms.Compose([
    #     transforms.RandomCrop(32, padding=4),
    #     transforms.RandomHorizontalFlip(),
    #     transforms.ToTensor(),
    # ])

    # trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    # trainloader = torch.utils.data.DataLoader(trainset, batch_size=256, shuffle=True,
    #                                           num_workers=4)
    # Model
    print('==> Building model..')
    best_acc = 0.0
    start_epoch = 0
    net = PreActResNet18().to(device)
    ######################原始注释，是否接着上次训练
    # checkpoint = torch.load(CIFAR_CKPT, map_location=torch.device(device))
    # net.load_state_dict(checkpoint['net'])
    # start_epoch = int(checkpoint['epoch'])
    # best_acc = float(checkpoint['acc'])
    #####################

    UndercoverAttack = Attack(net, nn.functional.cross_entropy)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9, weight_decay=5e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=40, gamma=0.1)

    net.train()
    for epoch in range(start_epoch, epochs):
        train_loss = 0
        correct, total = 0, 0
        for batch_idx, (inputs, targets) in tqdm(enumerate(train_dataloader),total=len(train_dataloader),desc="trainning"):
            # print("BBBBB",batch_idx)
            inputs, targets = inputs.to(device), targets.to(device)
            # print("INPuts",inputs.size(),inputs)
            optimizer.zero_grad()
            outputs = net(inputs)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            #############对所有训练样本生成对抗样本
            x_adv = undercover_attack(UndercoverAttack, inputs, targets, eps=0.15)
            ##########计算对抗样本的输出
            adv_outputs = net(x_adv)

            loss1 = criterion(outputs, targets)

            ###########计算对抗样本的损失
            loss2 = criterion(adv_outputs, targets)

            ########两个损失加权
            loss = loss1 + loss2 * 0.8
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            scheduler.step(epoch)
        acc = 1.0 * correct / total
        print('epoch: %d, train loss: %.2f, train acc: %.4f' % (epoch, train_loss, acc))
        if acc > best_acc:
            best_acc = acc
            state = {
                'net': net.state_dict(),
                'acc': acc,
                'epoch': epoch,
            }
            if not os.path.isdir('checkpoint'):
                os.mkdir('checkpoint')
            torch.save(state, CIFAR_CKPT)


def test(test_dataloader):
    # Data
    print('==> Preparing data..')
    # transform_test = transforms.Compose([
    #     transforms.ToTensor(),
    # ])
    # testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)
    # testloader = torch.utils.data.DataLoader(testset, batch_size=256, shuffle=False,
    #                                          num_workers=4)

    # Model
    print('==> Building model..')
    net = PreActResNet18().to(device)
    criterion = nn.CrossEntropyLoss()
    checkpoint = torch.load(CIFAR_CKPT)
    net.load_state_dict(checkpoint['net'])

    net.eval()
    test_loss = 0
    correct, total = 0, 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in tqdm(enumerate(test_dataloader),total=len(test_dataloader),desc="testing"):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    acc = 1.0 * correct / total
    print('test loss: %.2f, test acc: %.4f' % (test_loss, acc))


if __name__ == '__main__':
    CIFAR_CKPT = '/home/intent/advDetect/github项目/DBA-master/checkpoint/cifar_undercover.pth'
    device = Config.device


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



    # train(50,dataloaders["train"])
    test(dataloaders["test"])
