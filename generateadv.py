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

from PIL import Image
import numpy as np

CIFAR_CKPT = './checkpoint/cifar_undercover.pth'

CIFAR_UNDERCOVER_CKPT = './checkpoint/cifar_undercover.pth'
device = 'cuda:5' if torch.cuda.is_available() else 'cpu'
# device


# CIFAR 预训练模型路径
CIFAR_UNDERCOVER_CKPT = '/home/intent/advDetect/github项目/DBA-master/checkpoint/cifar_undercover.pth'

CIFAR_CKPT_MLP='/home/intent/advDetect/github项目/DBA-master/checkpoint/cifar_undercover_mlp.pth'

CAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples/correct_adversarial_samples.pth'

INCAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples/incorrect_adversarial_samples.pth'

CAD_SAMPLE_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples/correct_adversarial_samples_test.pth'

INCAD_SAMPLE_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples/incorrect_adversarial_samples_test.pth'

FOLDER_TRAIN0='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_folder/train/0'
FOLDER_TRAIN1='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_folder/train/1'
FOLDER_TEST0='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_folder/test/0'
FOLDER_TEST1='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_folder/test/1'

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


import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

def tensor_to_image(tensor):
    """
    将 PyTorch 张量转换为 PIL 图像。
    """
    tensor = tensor.cpu().numpy()
    if tensor.ndim == 4:
        tensor = tensor[0]  # 提取第一个样本
    tensor = tensor.transpose(1, 2, 0)  # 转换为 HWC 格式
    tensor = (tensor * 255).astype(np.uint8)  # 转换为 8 位无符号整数
    return Image.fromarray(tensor)

import os
import torch
from tqdm import tqdm
from PIL import Image
import numpy as np

def tensor_to_image(tensor):
    """
    将 PyTorch 张量转换为 PIL 图像。
    """
    tensor = tensor.cpu().numpy()
    if tensor.ndim == 4:
        tensor = tensor[0]  # 提取第一个样本
    tensor = tensor.transpose(1, 2, 0)  # 转换为 HWC 格式
    tensor = (tensor * 255).astype(np.uint8)  # 转换为 8 位无符号整数
    return Image.fromarray(tensor)

def generate_and_save_adversarial_samples(model, dataloader, attacker, device, save_path_normal, save_path_adversarial, eps=0.3, alpha=1/255, iteration=10):
    # 确保保存目录存在
    os.makedirs(save_path_normal, exist_ok=True)
    os.makedirs(save_path_adversarial, exist_ok=True)
    
    for batch_idx, (x, y) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating adversarial samples"):
        x, y = x.to(device), y.to(device)
        y_pred = model(x).argmax(dim=1)
        
        x_adv = attacker.i_fgsm(x, y, eps=eps, alpha=alpha, iteration=iteration)
        y_pred_adv = model(x_adv).argmax(dim=1)
        selected_correct = (y == y_pred) & (y != y_pred_adv)
        
        correct_normal_samples = x[selected_correct].detach().cpu()
        correct_adversarial_samples = x_adv[selected_correct].detach().cpu()

        # 将自然样本保存为 JPG 文件
        for i, img_tensor in enumerate(correct_normal_samples):
            img = tensor_to_image(img_tensor)
            img.save(os.path.join(save_path_normal, f"correct_normal_batch_{batch_idx}_img_{i}.jpg"))

        # 将对抗样本保存为 JPG 文件
        for i, img_tensor in enumerate(correct_adversarial_samples):
            img = tensor_to_image(img_tensor)
            img.save(os.path.join(save_path_adversarial, f"correct_adversarial_batch_{batch_idx}_img_{i}.jpg"))

        print(f"Batch {batch_idx}: Natural samples saved to {save_path_normal}")
        print(f"Batch {batch_idx}: Adversarial samples saved to {save_path_adversarial}")

    # 清理 GPU 的缓存内存
    torch.cuda.empty_cache()


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

    generate_and_save_adversarial_samples(undercoverNet, trainloader, undercover_gradient_attacker, device, FOLDER_TRAIN0,FOLDER_TRAIN1)
        
    # dba_trainloader = load_and_concatenate_adversarial_samples(INCAD_SAMPLE,32)

    # train(dba_trainloader)

    generate_and_save_adversarial_samples(undercoverNet, testloader, undercover_gradient_attacker, device, FOLDER_TEST0, FOLDER_TEST1)
    # dba_testloader=load_and_concatenate_adversarial_samples(INCAD_SAMPLE_TEST,32)

    # test(dba_testloader)

