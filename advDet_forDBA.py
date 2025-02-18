import os
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from torch import nn, optim
from tqdm import tqdm
from PIL import Image
from models.model_mk import PreActResNet18, MLP
from adversary.fgsm import Attack
import torch.nn.functional as F


class Config:
    model_path_k = "/home/intent/advDetect/model/cifar_undercover.pth"
    adv_path = "/home/intent/advDetect/Caltech_256/adversarial_forDBA/bad" #对抗样本
    nature_path = "/home/intent/advDetect/Caltech_256/adversarial_forDBA/nature" #测试样本
    batch_size = 32
    num_epochs = 10
    lr = 0.001
    device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    checkpoint_path = "/home/intent/advDetect/model/adv_detector_MLP.pth"


class CustomDataset(Dataset):
    def __init__(self, adversarial_dir, natural_dir, transform=None):
        self.adversarial_files = [os.path.join(adversarial_dir, f) for f in os.listdir(adversarial_dir)]
        self.natural_files = [os.path.join(natural_dir, f) for f in os.listdir(natural_dir)]
        self.files = self.adversarial_files + self.natural_files
        self.labels = [1] * len(self.adversarial_files) + [0] * len(self.natural_files)
        self.transform = transform
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        img_path = self.files[idx]
        label = self.labels[idx]
        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
        
        return img, label
def load_model_k(model_path):
    model = PreActResNet18().to(Config.device)
    checkpoint = torch.load(model_path, map_location=torch.device(Config.device))
    model.load_state_dict(checkpoint['net'])
    return model

# 数据加载函数
def load_data(adversarial_dir, natural_dir, batch_size=32, transform=None):
    dataset = CustomDataset(adversarial_dir, natural_dir, transform)
    
    train_size = int(0.8 * len(dataset))
    test_size = len(dataset) - train_size 
    
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])
    
    # DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, test_loader

# 训练函数
def train(model, undercoverNet, attack,train_loader, optimizer, criterion, device, epochs=50):
    best_acc = 0
    for epoch in range(epochs):
        epoch_loss = 0
        total, correct = 0, 0

        model.train()
        undercoverNet.train()

        # 使用 tqdm 包装训练迭代
        for batch_idx, (x, y) in tqdm(enumerate(train_loader), total=len(train_loader), desc=f"Training Epoch {epoch + 1}/{epochs}"):
            optimizer.zero_grad()
            x, y = x.to(device), y.to(device)
            y1, V1 = undercoverNet(x, dba=True)
            ylable = y1.argmax(dim=1)

            # 使用对抗攻击生成对抗样本
            undercover_adv = attack.fgsm(x, ylable, False, 1/255)

            # 计算对抗样本的特征
            _, V2 = undercoverNet(undercover_adv, dba=True)

            # 合并两个特征并计算损失
            V = torch.cat([V1, V2, V1 - V2, V1 * V2], axis=-1)
            y_pred = model(V)
            loss = criterion(y_pred, y)
            loss.backward()
            optimizer.step()

            # 累计损失
            epoch_loss += loss.item()

            # 计算准确率
            total += y.size(0)
            correct += y_pred.argmax(dim=1).eq(y).sum().item()

        # 每个 epoch 的平均损失和准确率
        avg_loss = epoch_loss / len(train_loader)
        acc = correct / total
        print(f"Epoch [{epoch + 1}/{epochs}], Loss: {avg_loss:.4f}, Accuracy: {acc:.4f}")
        
        # 保存最佳模型
        if acc > best_acc:
            best_acc = acc
            state = {
                'net': model.state_dict(),
                'acc': acc,
                'epoch': epoch + 1,
            }
            
            torch.save(state, Config.checkpoint_path)

    return best_acc

# 测试函数
def test(model, undercoverNet, attack, test_loader, criterion, device):
    model.eval()
    undercoverNet.eval()
    test_loss = 0
    test_correct = 0
    test_total = 0

    with torch.no_grad():
        # 使用 tqdm 包装测试迭代
        for batch_idx, (x, y) in tqdm(enumerate(test_loader), total=len(test_loader), desc="Testing"):
            x, y = x.to(device), y.to(device)
            y1, V1 = undercoverNet(x, dba=True)
            ylable = y1.argmax(dim=1)
            undercover_adv = attack.fgsm(x, ylable, False, 1/255)
            _, V2 = undercoverNet(undercover_adv, dba=True)
            V = torch.cat([V1, V2, V1 - V2, V1 * V2], axis=-1)
            y_pred = model(V)
            loss = criterion(y_pred, y)
            test_loss += loss.item()
            test_total += y.size(0)
            test_correct += y_pred.argmax(dim=1).eq(y).sum().item()

    test_acc = test_correct / test_total
    print(f"Test Accuracy: {test_acc:.4f}")

    return test_acc

# main 函数
def main():
    # 数据加载
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    train_loader, test_loader = load_data(Config.adv_path, Config.nature_path, Config.batch_size, transform)

    # 初始化模型
    undercoverNet = load_model_k(Config.model_path_k)
    mlp = MLP().to(Config.device)
    attack = Attack(undercoverNet,F.cross_entropy)

    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(mlp.parameters(), lr=Config.lr)

    # 训练模型
    print("Training the model...")
    best_acc = train(mlp, undercoverNet, attack,train_loader, optimizer, criterion, Config.device, Config.num_epochs)

    # 测试模型
    print("Testing the model...")
    test_acc = test(mlp, undercoverNet, attack ,test_loader, criterion, Config.device)

    print(f"Best accuracy during training: {best_acc:.4f}")
    print(f"Test accuracy: {test_acc:.4f}")

if __name__ == '__main__':
    main()
