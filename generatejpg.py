import torch
from PIL import Image
import os

CAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_bk/correct_adversarial_samples.pth'

INCAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_bk/incorrect_adversarial_samples.pth'

CAD_SAMPLE_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_bk/correct_adversarial_samples_test.pth'

INCAD_SAMPLE_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_bk/incorrect_adversarial_samples_test.pth'

SAVE_FILE='/home/intent/advDetect/github项目/DBA-master/JPG_bk'

def tensor_to_image(tensor):
    # 将 Tensor 转换为 PIL 图像
    tensor = tensor.mul(255).byte()
    tensor = tensor.cpu().numpy().transpose((1, 2, 0))
    return Image.fromarray(tensor)

def save_images_from_tensor(tensors, labels, folder, prefix):
    # 创建保存图像的文件夹
    if not os.path.exists(folder):
        os.makedirs(folder)
    
    # 保存每个 Tensor 为图像文件
    for i, (tensor, label) in enumerate(zip(tensors, labels)):
        image = tensor_to_image(tensor)
        image.save(os.path.join(folder, f"{prefix}_{i}_label_{label}.jpg"))

def load_and_save_adversarial_samples_as_images(save_path_incorrect, folder):
    # 加载被错误分类的对抗样本
    incorrect_data = torch.load(save_path_incorrect)
    
    incorrect_normal_x = incorrect_data['normal_x']
    incorrect_adversarial_x = incorrect_data['adversarial_x']
    incorrect_normal_y = incorrect_data['normal_y']
    incorrect_adversarial_y = incorrect_data['adversarial_y']

    # 保存正常样本图像
    save_images_from_tensor(incorrect_normal_x, incorrect_normal_y, folder+"/normal", "incorrect_normal")
    
    # 保存对抗样本图像
    save_images_from_tensor(incorrect_adversarial_x, incorrect_adversarial_y, folder+"/adversarial", "incorrect_adversarial")

    print(f"Adversarial samples saved as images in {folder}")

# 示例用法
load_and_save_adversarial_samples_as_images(INCAD_SAMPLE, SAVE_FILE)
