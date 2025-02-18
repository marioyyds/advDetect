import torch
import io

CAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples/correct_adversarial_samples.pth'

INCAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples_big/incorrect_adversarial_samples.pth'

CAD_SAMPLE_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples/correct_adversarial_samples_test.pth'

INCAD_SAMPLE_TEST='/home/intent/advDetect/github项目/DBA-master/adversarial_samples/incorrect_adversarial_samples_test.pth'

def load_large_tensor(file_path):
    with open(file_path, 'rb') as f:
        buffer = io.BytesIO(f.read())
    return torch.load(buffer)

# incorrect_data = load_large_tensor(CAD_SAMPLE)

import torch

try:
    data = torch.load(INCAD_SAMPLE)
    print("File loaded successfully")
except Exception as e:
    print(f"Error loading file: {e}")


# import torch
# import numpy as np

# # 加载 .pth 文件
# incorrect_data = torch.load(INCAD_SAMPLE)

# # 创建内存映射文件
# correct_normal_x = np.memmap('correct_normal_x.dat', dtype='float32', mode='w+', shape=incorrect_data['normal_x'].shape)
# correct_adversarial_x = np.memmap('correct_adversarial_x.dat', dtype='float32', mode='w+', shape=incorrect_data['adversarial_x'].shape)

# # 将数据写入内存映射文件
# correct_normal_x[:] = incorrect_data['normal_x'].numpy()
# correct_adversarial_x[:] = incorrect_data['adversarial_x'].numpy()

# # 读取内存映射文件
# correct_normal_x = torch.from_numpy(np.memmap('correct_normal_x.dat', dtype='float32', mode='r', shape=incorrect_data['normal_x'].shape))
# correct_adversarial_x = torch.from_numpy(np.memmap('correct_adversarial_x.dat', dtype='float32', mode='r', shape=incorrect_data['adversarial_x'].shape))

