import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
from tqdm import tqdm

from models.model_mk import PreActResNet18
from adversary.fgsm import Attack
import torch.nn.functional as F

# 配置参数
class Config:
    data_root = "/home/intent/advDetect/Caltech_256/256_ObjectCategories"  # 数据集根目录
    model_path = "/home/intent/advDetect/model/cifar_undercover.pth"  # 训练好的模型路径
    batch_size = 32
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    output_dir = "/home/intent/advDetect/Caltech_256/adversarial_forDBA"  # 保存路径
    eps = 0.03  # 对抗攻击的扰动大小
    alpha = 0.01  # PGD 的步长
    steps = 10  # PGD 的迭代次数


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
def load_model(model_path):
    model = PreActResNet18().to(Config.device)
    checkpoint = torch.load(model_path, map_location=torch.device(Config.device))
    model.load_state_dict(checkpoint['net'])
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
def generate_adversarial_samples_fgsm(model, dataloader,output_dir_bad, output_dir_nature):
    os.makedirs(output_dir_bad, exist_ok=True)
    os.makedirs(output_dir_nature,exist_ok=True)

    attack = Attack(model,F.cross_entropy)
   

    # for i, (inputs, labels) in enumerate(dataloader):
    for i, (inputs, labels) in tqdm(enumerate(dataloader), total=len(dataloader), desc="Generating Adversarial Samples"):

        inputs = inputs.to(Config.device)
        labels = labels.to(Config.device)

        adv_inputs = attack.i_fgsm(inputs, labels, eps=Config.eps, alpha=1/255, iteration=int(min(Config.eps * 255 + 4, 1.25 * Config.eps * 255)))

        adv_preds = model(adv_inputs)
        _, adv_preds = torch.max(adv_preds, 1)

        misclassified_indices = (adv_preds != labels).nonzero(as_tuple=True)[0]


        # 保存对抗样本
        for idx in misclassified_indices:
            adv_image = transforms.ToPILImage()(adv_inputs[idx].cpu())
            original_image = transforms.ToPILImage()(inputs[idx].cpu())

            # Save the adversarial image
            adv_image.save(os.path.join(output_dir_bad, f"adv_{i * Config.batch_size + idx.item()}.jpg"))

            # Save the corresponding original image
            original_image.save(os.path.join(output_dir_nature, f"nature_{i * Config.batch_size + idx.item()}.jpg"))

# 主函数
def main():
    # 加载模型
    model = load_model(Config.model_path)

    # 加载数据集
    transform = get_transforms()
    dataset = datasets.ImageFolder(Config.data_root, transform=transform)
    dataloader = DataLoader(dataset, batch_size=Config.batch_size, shuffle=False)

    # # 生成 FGSM 对抗样本
    print("Generating FGSM adversarial samples...")
    generate_adversarial_samples_fgsm(model, dataloader,os.path.join(Config.output_dir, "bad"),os.path.join(Config.output_dir, "nature"))
    print("Adversarial sample generation complete.")

if __name__ == "__main__":
    main()