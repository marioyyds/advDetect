import os
import torch
from torch.utils.data import DataLoader, TensorDataset

INCAD_SAMPLE='/home/intent/advDetect/github项目/DBA-master/adversarial_samples/incorrect_adversarial_samples.pth'


try:
    data = torch.load(INCAD_SAMPLE)
    print("File loaded successfully.")
    print(f"Normal X shape: {data['normal_x'].shape}")
    print(f"Adversarial X shape: {data['adversarial_x'].shape}")
except Exception as e:
    print(f"Error loading file: {e}")
