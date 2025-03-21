import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms, models
from tqdm import tqdm

# 配置参数
class Config:
    data_root = "/home/intent/advDetect/Caltech_256/256_ObjectCategories"  # 数据集根目录
    batch_size = 64
    num_epochs = 100
    num_classes = 257 
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "caltech256_best_model_100epochs.pth"
    train_ratio = 0.8  # 训练集比例
    val_ratio = 0.1    # 验证集比例
    test_ratio = 0.1   # 测试集比例

# 数据预处理
def get_transforms():
    return {
        "train": transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "val": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        "test": transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
    }

# 创建模型
def create_model(num_classes):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    return model.to(Config.device)

# 训练验证流程
def train_model(model, criterion, optimizer, scheduler, dataloaders):
    best_acc = 0.0

    for epoch in range(Config.num_epochs):
        print(f"\nEpoch {epoch + 1}/{Config.num_epochs}")
        print("-" * 60)

        # 每个epoch有训练和验证阶段
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()  # 训练模式
            else:
                model.eval()  # 验证模式

            running_loss = 0.0
            running_corrects = 0

            # 使用tqdm进度条
            with tqdm(dataloaders[phase], unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(Config.device)
                    labels = labels.to(Config.device)

                    # 梯度清零
                    optimizer.zero_grad()

                    # 前向传播
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs)
                        _, preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                        # 反向传播 + 优化仅在训练阶段
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # 统计指标
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)

                    # 更新进度条描述
                    pbar.set_description(f"{phase} Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

            # 深拷贝模型（如果是最好的模型）
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                torch.save(model.state_dict(), Config.checkpoint_path)
                print(f"New best model saved with val acc: {best_acc:.4f}")

        # 调整学习率
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    print(f"\nTraining complete. Best val Acc: {best_acc:.4f}")

# 主函数
def main():
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

    # 创建模型
    model = create_model(Config.num_classes)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=Config.lr,
                         momentum=Config.momentum,
                         weight_decay=Config.weight_decay)

    # 学习率调度器
    scheduler = lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.1, patience=5, verbose=True
    )

    # 开始训练
    train_model(model, criterion, optimizer, scheduler, dataloaders)

    # 测试集评估
    model.load_state_dict(torch.load(Config.checkpoint_path))
    model.eval()
    test_corrects = 0
    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            test_corrects += torch.sum(preds == labels.data)
    test_acc = test_corrects.double() / len(dataloaders["test"].dataset)
    print(f"Test Accuracy: {test_acc:.4f}")

if __name__ == "__main__":
    main()