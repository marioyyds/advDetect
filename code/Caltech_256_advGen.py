import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from torchattacks import FGSM, PGD
from tqdm import tqdm

# 配置参数
class Config:
    data_root = "/home/intent/advDetect/Caltech_256/test"  # 数据集根目录
    model_path = "/home/intent/advDetect/model/caltech256_best_model.pth"  # 训练好的模型路径
    batch_size = 32
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    output_dir = "/home/intent/advDetect/Caltech_256/test_adv"  # 对抗样本保存路径
    eps = 0.03  # 对抗攻击的扰动大小
    alpha = 0.01  # PGD 的步长
    steps = 3  # PGD 的迭代次数


# 定义标准化层
class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.mean = torch.tensor(mean).view(1, -1, 1, 1)
        self.std = torch.tensor(std).view(1, -1, 1, 1)
    
    def forward(self, x):
        mean = self.mean.to(x.device)
        std = self.std.to(x.device)
        return (x - mean) / std
    
# 加载模型
def load_model(model_path, num_classes=257):
    model = models.resnet18(weights=None)  # 不使用预训练权重
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    model.load_state_dict(torch.load(model_path))
    normalize_layer = Normalize(mean=[0.4914, 0.4822, 0.4465], std=[0.2471, 0.2435, 0.2616])
    model = nn.Sequential(normalize_layer, model)
    model = model.to(Config.device)
    model.eval()  # 设置为评估模式
    return model

# 数据预处理
def get_transforms():
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        # transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

# 生成对抗样本
def generate_adversarial_samples_fgsm(model, dataloader, attack_method, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    attack = attack_method(model, eps=Config.eps)
    # attack = attack_method
   

    # for i, (inputs, labels) in enumerate(dataloader):
    for i, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating Adversarial Samples"):

        inputs = inputs.to(Config.device)
        labels = labels.to(Config.device)

        # 生成对抗样本
        adv_inputs = attack(inputs, labels)

        # 保存对抗样本
        for j in range(adv_inputs.size(0)):
            adv_image = transforms.ToPILImage()(adv_inputs[j].cpu())
            adv_image.save(os.path.join(output_dir, f"adv_{i * Config.batch_size + j}.jpg"))

        # print(f"Generated {len(adv_inputs)} adversarial samples.")

def generate_adversarial_samples_pgd(model, dataloader, attack_method, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    # attack = attack_method(model, eps=Config.eps)
    attack = attack_method
   

    # for i, (inputs, labels) in enumerate(dataloader):
    for i, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating Adversarial Samples"):

        inputs = inputs.to(Config.device)
        labels = labels.to(Config.device)

        # 生成对抗样本
        adv_inputs = attack(inputs, labels)

        # 保存对抗样本
        for j in range(adv_inputs.size(0)):
            adv_image = transforms.ToPILImage()(adv_inputs[j].cpu())
            adv_image.save(os.path.join(output_dir, f"adv_{i * Config.batch_size + j}.jpg"))

# 主函数
def main():
    # 加载模型
    model = load_model(Config.model_path)

    # 加载数据集
    transform = get_transforms()
    dataset = datasets.ImageFolder(Config.data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)

    # # # 生成 FGSM 对抗样本
    # print("Generating FGSM adversarial samples...")
    # generate_adversarial_samples_fgsm(model, dataloader, FGSM, os.path.join(Config.output_dir, "fgsm"))

    # 生成 PGD 对抗样本
    print("Generating PGD adversarial samples...")
    pgd_attack = PGD(model, eps=Config.eps, alpha=Config.alpha, steps=Config.steps)
    generate_adversarial_samples_pgd(model, dataloader, pgd_attack, os.path.join(Config.output_dir, "pgd"))

    print("Adversarial sample generation complete.")

if __name__ == "__main__":
    main()