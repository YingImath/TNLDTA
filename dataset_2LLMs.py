import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset

data_path = '/home/flower/lxy/TNLDTA/data'

class DrugTargetDataset(Dataset):
    def __init__(self, dataset, drug_maxlen, target_maxlen, mode='train', device='cuda'):
        self.device = device
        self.drug_maxlen = drug_maxlen
        self.target_maxlen = target_maxlen

        mapping_csv = os.path.join(data_path, dataset, f"{mode}_mapping.csv")
        self.mapping = pd.read_csv(mapping_csv)

        drug_json = os.path.join(data_path, dataset, 'drugs.json')
        with open(drug_json, 'r') as f:
            self.drugs_dict = json.load(f)

        target_json = os.path.join(data_path, dataset, 'targets.json')
        with open(target_json, 'r') as f:
            self.targets_dict = json.load(f)

        self.drugs_ST_dir = os.path.join(data_path, dataset, 'drugs_ST')
        self.targets_ESM_dir = os.path.join(data_path, dataset, 'targets_ESM')

    def __len__(self):
        """返回数据集长度"""
        return len(self.mapping)
    
    def __getitem__(self, idx):
        row = self.mapping.iloc[idx]
        key_drug = row['key_drug']
        key_target = row['key_target']
        affinity = row['affinity']
        
        drug_seq = self.drugs_dict[str(key_drug)]
        
        target_seq = self.targets_dict[str(key_target)]

        drug_pretrained_path = os.path.join(self.drugs_ST_dir, f"{key_drug}.npy")
        drug_pretrained = np.load(drug_pretrained_path)
        
        target_pretrained_path = os.path.join(self.targets_ESM_dir, f"{key_target}.npy")
        target_pretrained = np.load(target_pretrained_path)

        # 转换为PyTorch张量
        return (
            torch.tensor(drug_seq, dtype=torch.long),
            torch.tensor(drug_pretrained, dtype=torch.float, device=self.device),
            torch.tensor(target_seq, dtype=torch.long),
            torch.tensor(target_pretrained, dtype=torch.float, device=self.device),
            torch.tensor([affinity], dtype=torch.float)
        )


if __name__ == '__main__':
    dataset = DrugTargetDataset(
        dataset='KIBA',
        drug_maxlen=100,
        target_maxlen=1000,
        mode='test',
        device='cuda:3'
    )
    
    # 获取第一个样本
    drug_seq, drug_pretrained, target_seq, target_pretrained, affinity = dataset[0]
    print(drug_seq.shape)
    print(drug_pretrained.shape)
    print(target_seq.shape)
    print(target_pretrained.shape)
    print(affinity.shape)
