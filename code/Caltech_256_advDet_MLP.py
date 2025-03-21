import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms, models
from torchvision.datasets import ImageFolder
from PIL import Image
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

# 配置参数
class Config:
    adv_samples_dir_fgsm = "/data/hdd1/intent_backup/advDetect/Caltech_256/adversarial_samples/fgsm"  # 对抗样本路径
    adv_samples_dir_pgd = "/data/hdd1/intent_backup/advDetect/Caltech_256/adversarial_samples/pgd"  # 对抗样本路径
    natural_samples_dir = "/data/hdd1/intent_backup/advDetect/Caltech_256/256_ObjectCategories"  # 原始样本路径
    batch_size = 64
    num_epochs = 10
    lr = 0.001
    momentum = 0.9
    weight_decay = 1e-4
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "/data/hdd1/intent_backup/advDetect/model/adv_detector_best_model_toFGSMandPGD.pth"
    train_ratio = 0.8  # 训练集比例
    val_ratio = 0.1    # 验证集比例
    test_ratio = 0.1   # 测试集比例

# 自定义数据集
class AdvNaturalDataset(Dataset):
    def __init__(self, adv_dir_fgsm, adv_dir_pgd, natural_dir, transform=None):
        self.adv_samples = [os.path.join(adv_dir_fgsm, f) for f in os.listdir(adv_dir_fgsm)]
        self.adv_samples += [os.path.join(adv_dir_pgd, f) for f in os.listdir(adv_dir_pgd)]
        self.natural_samples = [os.path.join(natural_dir, c, f) 
                                for c in os.listdir(natural_dir) 
                                for f in os.listdir(os.path.join(natural_dir, c))]
        self.samples = self.adv_samples + self.natural_samples
        self.labels = [1] * len(self.adv_samples) + [0] * len(self.natural_samples)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path = self.samples[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label

# 数据预处理
def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# 创建 DenseNet121 模型
def create_model():
    model = models.densenet121(pretrained=True)  # 加载预训练的 DenseNet121
    model.classifier = nn.Linear(model.classifier.in_features, 1)  # 修改最后一层为二分类输出
    return model.to(Config.device)

# 训练验证流程
def train_model(model, criterion, optimizer, scheduler, dataloaders):
    best_auc = 0.0

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
            all_labels = []
            all_probs = []

            # 使用tqdm进度条
            with tqdm(dataloaders[phase], unit="batch") as pbar:
                for inputs, labels in pbar:
                    inputs = inputs.to(Config.device)
                    labels = labels.float().to(Config.device)

                    # 梯度清零
                    optimizer.zero_grad()

                    # 前向传播
                    with torch.set_grad_enabled(phase == "train"):
                        outputs = model(inputs).squeeze()
                        loss = criterion(outputs, labels)

                        # 反向传播 + 优化仅在训练阶段
                        if phase == "train":
                            loss.backward()
                            optimizer.step()

                    # 统计指标
                    running_loss += loss.item() * inputs.size(0)
                    all_labels.extend(labels.cpu().numpy())
                    all_probs.extend(torch.sigmoid(outputs).cpu().detach().numpy())

                    # 更新进度条描述
                    pbar.set_description(f"{phase} Loss: {loss.item():.4f}")

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_auc = roc_auc_score(all_labels, all_probs)

            print(f"{phase} Loss: {epoch_loss:.4f} AUC: {epoch_auc:.4f}")

            # 深拷贝模型（如果是最好的模型）
            if phase == "val" and epoch_auc > best_auc:
                best_auc = epoch_auc
                torch.save(model.state_dict(), Config.checkpoint_path)
                print(f"New best model saved with val AUC: {best_auc:.4f}")

        # 调整学习率
        if isinstance(scheduler, lr_scheduler.ReduceLROnPlateau):
            scheduler.step(epoch_loss)
        else:
            scheduler.step()

    print(f"\nTraining complete. Best val AUC: {best_auc:.4f}")

# 主函数
def main():
    # 数据加载
    transform = get_transforms()
    dataset = AdvNaturalDataset(Config.adv_samples_dir_fgsm, Config.adv_samples_dir_pgd, Config.natural_samples_dir, transform=transform)

    # 划分数据集
    train_size = int(Config.train_ratio * len(dataset))
    val_size = int(Config.val_ratio * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])

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
    model = create_model()

    # 定义损失函数和优化器
    criterion = nn.BCEWithLogitsLoss()
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
    test_labels = []
    test_probs = []
    with torch.no_grad():
        for inputs, labels in dataloaders["test"]:
            inputs = inputs.to(Config.device)
            labels = labels.to(Config.device)
            outputs = model(inputs).squeeze()
            test_labels.extend(labels.cpu().numpy())
            test_probs.extend(torch.sigmoid(outputs).cpu().numpy())

    test_auc = roc_auc_score(test_labels, test_probs)
    print(f"Test AUC: {test_auc:.4f}")

if __name__ == "__main__":
    main()