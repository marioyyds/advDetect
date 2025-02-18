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
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
# device


# CIFAR 预训练模型路径
CIFAR_UNDERCOVER_CKPT = '/home/intent/advDetect/github项目/DBA-master/checkpoint/cifar_undercover.pth'

CIFAR_CKPT_MLP='/home/intent/advDetect/github项目/DBA-master/checkpoint/cifar_undercover_mlp.pth'

CAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_mt/correct_adversarial_samples.pth'

INCAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_mt/incorrect_adversarial_samples.pth'

CAD_SAMPLE_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples/correct_adversarial_samples_test.pth'

INCAD_SAMPLE_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples/incorrect_adversarial_samples_test.pth'

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
    train_ratio = 0.009  # 训练集比例
    val_ratio = 0.99    # 验证集比例
    test_ratio = 0.001   # 测试集比例

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


####################################test
def generate_and_save_adversarial_samples(model, dataloader, attacker, device, save_path_correct, save_path_incorrect, eps=0.3, alpha=1/255, iteration=10):
    correct_normal_samples, correct_adversarial_samples = [], []
    incorrect_normal_samples, incorrect_adversarial_samples = [], []

    for batch_idx, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating adversarial samples"):
        x, y = x.to(device), y.to(device)
        y_pred = model(x).argmax(dim=1)
        
        x_adv = attacker.i_fgsm(x, y, eps=eps, alpha=alpha, iteration=iteration)
        y_pred_adv = model(x_adv).argmax(dim=1)
        selected_correct = (y == y_pred) & (y != y_pred_adv)
        selected_incorrect = ~selected_correct
        
        correct_normal_samples.append(x[selected_correct].detach().cpu())
        correct_adversarial_samples.append(x_adv[selected_correct].detach().cpu())
        incorrect_normal_samples.append(x[selected_incorrect].detach().cpu())
        incorrect_adversarial_samples.append(x_adv[selected_incorrect].detach().cpu())

    # 正确分类的样本
    correct_normal_x = torch.cat(correct_normal_samples, dim=0)
    correct_adversarial_x = torch.cat(correct_adversarial_samples, dim=0)
    correct_normal_y = torch.zeros(correct_normal_x.shape[0]).long()
    correct_adversarial_y = torch.ones(correct_adversarial_x.shape[0]).long()
    
    # 错误分类的样本
    incorrect_normal_x = torch.cat(incorrect_normal_samples, dim=0)
    incorrect_adversarial_x = torch.cat(incorrect_adversarial_samples, dim=0)
    incorrect_normal_y = torch.zeros(incorrect_normal_x.shape[0]).long()
    incorrect_adversarial_y = torch.ones(incorrect_adversarial_x.shape[0]).long()
    
    # 存储被错误分类的对抗样本
    torch.save({
        'normal_x': correct_normal_x,
        'adversarial_x': correct_adversarial_x,
        'normal_y': correct_normal_y,
        'adversarial_y': correct_adversarial_y,
    }, save_path_incorrect)
    
    # 存储其他对抗样本
    torch.save({
        'normal_x': incorrect_normal_x,
        'adversarial_x': incorrect_adversarial_x,
        'normal_y': incorrect_normal_y,
        'adversarial_y': incorrect_adversarial_y,
    }, save_path_correct)

    print(f"Correctly classified adversarial samples saved to {save_path_correct}")
    print(f"Incorrectly classified adversarial samples saved to {save_path_incorrect}")

    del correct_normal_samples, correct_adversarial_samples
    del incorrect_normal_samples, incorrect_adversarial_samples
    del correct_normal_x, correct_adversarial_x, correct_normal_y, correct_adversarial_y
    del incorrect_normal_x, incorrect_adversarial_x, incorrect_normal_y, incorrect_adversarial_y
    torch.cuda.empty_cache()


# generate_and_save_adversarial_samples(undercoverNet, trainloader, undercover_gradient_attacker, device, CAD_SAMPLE, INCAD_SAMPLE)


def load_and_concatenate_adversarial_samples(incorrect_save_path, batch_size=Config.batch_size, shuffle=True, num_workers=4):
    # 加载被错误分类的对抗样本
    incorrect_data = torch.load(incorrect_save_path)
    
    incorrect_normal_x = incorrect_data['normal_x']
    incorrect_adversarial_x = incorrect_data['adversarial_x']
    incorrect_normal_y = incorrect_data['normal_y']
    incorrect_adversarial_y = incorrect_data['adversarial_y']

    # 拼接正常样本和对抗样本
    normal_x = torch.cat((incorrect_normal_x,), dim=0)
    adversarial_x = torch.cat((incorrect_adversarial_x,), dim=0)
    normal_y = incorrect_normal_y
    adversarial_y = incorrect_adversarial_y

    # 创建 DataLoader 对象
    dba_trainloader = DataLoader(TensorDataset(torch.cat((normal_x, adversarial_x), dim=0),
                                               torch.cat((normal_y, adversarial_y), dim=0)),
                                 batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    # dba_trainiter = iter(dba_trainloader)

    return dba_trainloader


##################################tttttttt
def generate_and_save_adversarial_samples1(model, dataloader, attacker, device, save_path_correct, save_path_incorrect, eps=0.3, alpha=1/255, iteration=10):
    for batch_idx, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating adversarial samples"):
        x, y = x.to(device), y.to(device)
        y_pred = model(x).argmax(dim=1)
        
        # 生成对抗样本
        x_adv = attacker.i_fgsm(x, y, eps=eps, alpha=alpha, iteration=iteration)
        y_pred_adv = model(x_adv).argmax(dim=1)
        
        # 选择正确分类和错误分类的样本
        selected_correct = (y == y_pred) & (y != y_pred_adv)
        selected_incorrect = ~selected_correct
        
        # 提取当前批次的正确分类和错误分类的样本
        correct_normal_samples = x[selected_correct].detach().cpu()
        correct_adversarial_samples = x_adv[selected_correct].detach().cpu()
        incorrect_normal_samples = x[selected_incorrect].detach().cpu()
        incorrect_adversarial_samples = x_adv[selected_incorrect].detach().cpu()

        # 为当前批次生成标签
        correct_normal_y = torch.zeros(correct_normal_samples.shape[0]).long()
        correct_adversarial_y = torch.ones(correct_adversarial_samples.shape[0]).long()
        incorrect_normal_y = torch.zeros(incorrect_normal_samples.shape[0]).long()
        incorrect_adversarial_y = torch.ones(incorrect_adversarial_samples.shape[0]).long()

        # 保存当前批次的正确分类样本
        # if len(correct_normal_samples) > 0:
        #     torch.save({
        #         'normal_x': correct_normal_samples,
        #         'adversarial_x': correct_adversarial_samples,
        #         'normal_y': correct_normal_y,
        #         'adversarial_y': correct_adversarial_y,
        #     }, f"{save_path_correct}_batch_{batch_idx}.pth")
        #     print(f"Batch {batch_idx}: Correctly classified adversarial samples saved to {save_path_correct}_batch_{batch_idx}.pth")

        # 保存当前批次的错误分类样本
        if len(incorrect_normal_samples) > 0:
            torch.save({
                'normal_x': incorrect_normal_samples,
                'adversarial_x': incorrect_adversarial_samples,
                'normal_y': incorrect_normal_y,
                'adversarial_y': incorrect_adversarial_y,
            }, f"{save_path_incorrect}_batch_{batch_idx}.pth")
            print(f"Batch {batch_idx}: Incorrectly classified adversarial samples saved to {save_path_incorrect}_batch_{batch_idx}.pth")

        # 释放内存
        del correct_normal_samples, correct_adversarial_samples
        del incorrect_normal_samples, incorrect_adversarial_samples
        del correct_normal_y, correct_adversarial_y, incorrect_normal_y, incorrect_adversarial_y
        torch.cuda.empty_cache()




def load_and_concatenate_adversarial_samples1(incorrect_save_path_pattern, batch_size=Config.batch_size, shuffle=True, num_workers=4):
    """
    加载所有批次的 .pth 文件并拼接成原来的格式。

    参数:
        incorrect_save_path_pattern (str): 文件路径模式，例如 "incorrect_samples_batch_*.pth"。
        batch_size (int): DataLoader 的批次大小。
        shuffle (bool): 是否打乱数据。
        num_workers (int): DataLoader 的工作线程数。

    返回:
        DataLoader: 包含拼接后的数据的 DataLoader 对象。
    """
    # 获取所有匹配的文件路径
    file_paths = glob.glob(incorrect_save_path_pattern)
    if not file_paths:
        raise FileNotFoundError(f"No files found matching pattern: {incorrect_save_path_pattern}")

    # 初始化空列表，用于存储每个文件中的数据
    normal_x_list, adversarial_x_list = [], []
    normal_y_list, adversarial_y_list = [], []

    # 遍历所有文件并加载数据
    for file_path in file_paths:
        data = torch.load(file_path)
        normal_x_list.append(data['normal_x'])
        adversarial_x_list.append(data['adversarial_x'])
        normal_y_list.append(data['normal_y'])
        adversarial_y_list.append(data['adversarial_y'])

    # 将所有批次的数据拼接起来
    normal_x = torch.cat(normal_x_list, dim=0)
    adversarial_x = torch.cat(adversarial_x_list, dim=0)
    normal_y = torch.cat(normal_y_list, dim=0)
    adversarial_y = torch.cat(adversarial_y_list, dim=0)

    # 创建 DataLoader 对象
    dba_trainloader = DataLoader(
        TensorDataset(torch.cat((normal_x, adversarial_x), dim=0),
        torch.cat((normal_y, adversarial_y), dim=0)),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers
    )

    return dba_trainloader

##################################


def train(dba_trainloader,epochs=10):
    
    for i in range(epochs):
        epoch_loss = 0
        total, correct = 0, 0
        best_acc = 0.0
        
        for batch_idx,(x, y) in tqdm(enumerate(dba_trainloader),total=len(dba_trainloader),desc="trainning"):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y1, V1 = undercoverNet(x, dba=True)
            ylable=y1.argmax(dim=1)
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
        print(f"Epoch [{i}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        if acc > best_acc:
                best_acc = acc
                state = {
                    'net': mlp.state_dict(),
                    'acc': acc,
                    'epoch': 10,
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


    generate_and_save_adversarial_samples1(undercoverNet, trainloader, undercover_gradient_attacker, device, CAD_SAMPLE, INCAD_SAMPLE)
    # dba_trainloader = load_and_concatenate_adversarial_samples(INCAD_SAMPLE,32)

    # train(dba_trainloader)

    # # generate_and_save_adversarial_samples(undercoverNet, testloader, undercover_gradient_attacker, device, CAD_SAMPLE_TEST, INCAD_SAMPLE_TEST)
    dba_testloader=load_and_concatenate_adversarial_samples1(INCAD_SAMPLE_TEST,32)

    test(dba_testloader)