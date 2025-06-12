import json
import pickle
import re
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from collections import OrderedDict
import torch.nn.functional as F

data_path = '/home/flower/lxy/TNLDTA/data'

##########################################################################
"""ESM"""
# import esm


# class ESM(nn.Module):
#     def __init__(self, model_name='facebook/esm2_t33_650M_UR50D', device="cpu"):
#         super(ESM, self).__init__()
#         self.model, self.alphabet = esm.pretrained.esm2_t33_650M_UR50D()  # 会下载预训练模型和字母表到torch那边
#         # # 获取字母表的词汇表字典
#         # alphabet_dict = self.alphabet.to_dict()
#         # print(alphabet_dict)
#         self.batch_converter = self.alphabet.get_batch_converter()
#         self.device = torch.device(device)
#         self.model.to(self.device)  # 确保模型在正确的设备上

#     def forward(self, sequences):
#         # eg: sequences=[protein1, protein2,...]
#         self.model.eval()
#         batch_labels, batch_strs, batch_tokens = self.batch_converter(sequences)
#         batch_tokens = batch_tokens.to(self.device)  # 移动输入到模型所在设备
#         batch_lens = (batch_tokens != self.alphabet.padding_idx).sum(1)

#         # Extract per-residue representations (on CPU)
#         with torch.no_grad():
#             results = self.model(batch_tokens, repr_layers=[33], return_contacts=True)
#         # print(results.keys())
#         token_representations = results["representations"][33]
#         return token_representations[0, 1:-1, :]

# # 提前计算靶点经过ESM的结果并保存

# device="cuda:0" if torch.cuda.is_available() else "cpu"
# esm_model = ESM(device=device)

# # KIBA数据集
# # 读取蛋白质的序列信息
# def load_target(path):
#     print("Read %s start" % path)

#     proteins = json.load(open(os.path.join(path, "proteins.txt")), object_pairs_hook=OrderedDict)  # OrderedDict保留键的顺序

#     sequences = []  # 存储靶点的序列信息

#     for key, value in proteins.items():
#         sequences.append((key, value))
#     return sequences

# # dataset = 'KIBA'
# # maxlen = 1000
# dataset = 'Davis'
# maxlen = 1200

# path = os.path.join(data_path, dataset)


# sequences = load_target(path)  # 获取靶点的序列信息

# # 提前计算并保存靶点的嵌入
# save_dir = os.path.join(path, 'targets_ESM')
# os.makedirs(save_dir, exist_ok=True)
# print("Saving to directory: ", save_dir)

# for i, sequence in enumerate(sequences):
#     # 如果序列长度大于maxlen，则进行截断
#     if len(sequence[1]) > maxlen:
#         sequence = (sequence[0], sequence[1][:maxlen])  # 创建一个新的元组，包含修改后的序列
#     output = esm_model([sequence]).squeeze(0)  # 去掉批次维度
#     seq_len = output.shape[0]
#     if seq_len > maxlen:
#         output = output[:maxlen]
#     elif seq_len < maxlen:
#         pad_size = maxlen - seq_len
#         output = F.pad(output, (0, 0, 0, pad_size), mode='constant', value=0)
        
#     save_file = os.path.join(save_dir, f'{sequence[0]}.npy')
#     np.save(save_file, output.cpu().detach().numpy())

#     print(f"Processed {i + 1} / {len(sequences)} targets.")
    
#     # 释放内存
#     del output
#     torch.cuda.empty_cache()
    
# print("All targets saved to directory " + save_dir)


##########################################################################
"""SMILES Transformer"""
# from smiles_transformer.pretrain_trfm import TrfmSeq2seq  # 确保导入模型和词汇类
# from smiles_transformer.build_vocab import WordVocab


# class SMILESTransformer(nn.Module):
#     def __init__(self, vocab_path, model_path, model_dim=256, num_layers=4):
#         super(SMILESTransformer, self).__init__()
        
#         # 加载词汇表
#         self.vocab = WordVocab.load_vocab(vocab_path)
#         # 初始化模型
#         self.model = TrfmSeq2seq(len(self.vocab), model_dim, len(self.vocab), num_layers)
#         self.model.load_state_dict(torch.load(model_path))
        
#         # 定义正则表达式来匹配 SMILES 中的多字符符号
#         self.TOKEN_PATTERN = re.compile(r'(\[|\]|Br|Cl|Si|Na|B|P|I|K|C|c|N|n|O|S|F|P|I|B|Na|Si|Se|K|#|=|/|\\|\+|-|\(|\)|\.|:|@|\?|>|\*|\$|%)')
#         # self.TOKEN_PATTERN = re.compile(r'(\[|\]|Br|Cl|Si|Na|B|P|I|K|C|c|N|n|O|S|F|P|I|B|Na|Si|Se|K|#|=|/|\\|\+|-|\(|\)|\.|:|@|\?|>|\*|\$|%|\d|\w)')
    
#     def smiles_to_indices(self, smiles):
#         indices = []
#         for smile in smiles:
#             tokens = self.TOKEN_PATTERN.findall(smile)
#             indices.append([self.vocab.stoi.get(token, self.vocab.unk_index) for token in tokens])
#         return indices
    
#     def indices_to_tensor(self, indices, max_len=None):
#         if max_len is None:
#             max_len = max(len(idx) for idx in indices)
#         tensor = torch.zeros((max_len, len(indices)), dtype=torch.long)
#         for i, idx in enumerate(indices):
#             tensor[:len(idx), i] = torch.tensor(idx, dtype=torch.long)
#         return tensor
    
#     def forward(self, smiles_list):
#         # 转换 SMILES 列表
#         indices = self.smiles_to_indices(smiles_list)
#         inputs = self.indices_to_tensor(indices)
        
#         # 进行预测
#         self.model.eval()  # 切换到评估模式
#         with torch.no_grad():
#             predictions = self.model(inputs, return_hidden=True)  # 确保输入格式正确
#             return predictions

# def load_ligands(path):
#     print("Read %s start" % path)
#     smiles = json.load(open(os.path.join(path, "ligands_can.txt")), object_pairs_hook=OrderedDict)  # OrderedDict保留键的顺序

#     return smiles


# # 初始化 SMILESTransformer 类
# smiles_transformer = SMILESTransformer(
#     vocab_path='smiles_transformer/vocab.pkl',
#     model_path='smiles_transformer/trfm_12_23000.pkl',
#     model_dim=256,
#     num_layers=4
# )

# dataset = 'KIBA'
# maxlen = 100
# # dataset = 'Davis'
# # maxlen = 85
# path = os.path.join(data_path, dataset)
# smiles = load_ligands(path)  # 获取药物的SMILES结构, 以列表形式存储
# print(len(smiles))

# # 提前计算并保存药物的嵌入
# save_dir = os.path.join(path, 'drugs_ST')
# os.makedirs(save_dir, exist_ok=True)
# print("Saving to directory: ", save_dir)

# for idx, seq in smiles.items():
#     save_file = os.path.join(save_dir, f'{idx}.npy')
#     output = smiles_transformer([seq]).squeeze(1)  # 去掉批次维度
#     seq_len = output.shape[0]
#     if seq_len > maxlen:
#         output = output[:maxlen]
#     elif seq_len < maxlen:
#         pad_size = maxlen - seq_len
#         output = F.pad(output, (0, 0, 0, pad_size), mode='constant', value=0)

#     np.save(save_file, output.cpu().detach().numpy())

#     print(f"Processed  drug {idx}.")

# print("All drugs saved to directory "+save_dir)

#########################################################################
"""train test split"""
from sklearn.model_selection import train_test_split


# ESM 用的词汇表
vocab_esm = {'<pad>': 0, '<cls>': 1, '<eos>': 2, '<unk>': 3, 'L': 4, 'A': 5, 
             'G': 6, 'V': 7, 'S': 8, 'E': 9, 'R': 10, 
             'T': 11, 'I': 12, 'D': 13, 'P': 14, 'K': 15, 
             'Q': 16, 'N': 17, 'F': 18, 'Y': 19, 'M': 20, 
             'H': 21, 'W': 22, 'C': 23, 'X': 24, 'B': 25, 
             'U': 26, 'Z': 27, 'O': 28, '.': 29, '-': 30, 
             '<null_1>': 31, '<mask>': 32}
vocab_esm_size = 33

# SMILES Transformer 用的词汇表
vocab_ST = {'<pad>': 0, '<unk>': 1, '<eos>': 2, '<sos>': 3, '<mask>': 4, 'c': 5, 
             'C': 6, '(': 7, ')': 8, 'O': 9, '=': 10, 
             '1': 11, 'N': 12, '2': 13, '3': 14, 'n': 15, 
             '4': 16, '@': 17, '[': 18, ']': 19, 'H': 20, 
             'F': 21, '5': 22, 'S': 23, '\\': 24, 'Cl': 25, 
             's': 26, '6': 27, 'o': 28, '+': 29, '-': 30, 
             '#': 31, '/': 32, '.': 33, 'Br': 34, '7': 35, 
             'P': 36, 'I': 37, '8': 38, 'Na': 39, 'B': 40, 
             'Si': 41, 'Se': 42, '9': 43, 'K': 44}
vocab_ST_size = 45


def process_data(dataset, drug_maxlen, target_maxlen, random_state=None):
    dir_path = os.path.join(data_path, dataset)
    # 1. 读取原始序列
    print("Read %s start" % dir_path)

    key_drugs, seq_drugs = load_drug(dir_path=dir_path, drug_maxlen=drug_maxlen, drugSeq_vocab=vocab_ST)
    drugs_dict = {key: seq for key, seq in zip(key_drugs, seq_drugs)}
    drug_json = os.path.join(dir_path, 'drugs.json')
    with open(drug_json, 'w') as f:
        json.dump(drugs_dict, f)

    key_targets, seq_targets = load_target(dir_path=dir_path, target_maxlen=target_maxlen, targetSeq_vocab=vocab_esm, padding_value=0)
    targets_dict = {key: seq for key, seq in zip(key_targets, seq_targets)}
    target_json = os.path.join(dir_path, 'targets.json')
    with open(target_json, 'w') as f:
        json.dump(targets_dict, f)

    if dataset == 'Davis':
        Y = load_affinity(dir_path, is_log=True)
    else:
        Y = load_affinity(dir_path)
    print(Y, Y.dtype, Y.shape)

    # 3. 生成样本
    # 剔除空数据，将亲和力矩阵转换为列表
    key_drugs, key_targets, affinities = get_samples(key_drugs, key_targets, Y)

    # 打乱数据，划分数据集
    split_train_test(key_drugs, key_targets, affinities, dir_path, test_size=1/6, random_state=random_state)


def load_drug(dir_path, drug_maxlen, drugSeq_vocab, padding_value=0):
    smiles = json.load(open(os.path.join(dir_path, "ligands_can.txt")), object_pairs_hook=OrderedDict)  # OrderedDict保留键的顺序

    # 定义一个正则表达式来匹配 SMILES 中的多字符符号
    TOKEN_PATTERN = re.compile(r'(\[|\]|Br|Cl|Si|Na|B|P|I|K|C|c|N|n|O|S|F|P|I|B|Na|Si|Se|K|#|=|/|\\|\+|-|\(|\)|\.|:|@|\?|>|\*|\$|%)')
    # TOKEN_PATTERN = re.compile(r'(\[|\]|Br|Cl|Si|Na|B|P|I|K|C|c|N|n|O|S|F|P|I|B|Na|Si|Se|K|#|=|/|\\|\+|-|\(|\)|\.|:|@|\?|>|\*|\$|%|\d|\w)')

    key_drugs, seq_drugs = [], []
    for key, seq in smiles.items():
        key_drugs.append(key)

        tokens = TOKEN_PATTERN.findall(seq)
        drug_seq = []
        if len(tokens) >= drug_maxlen:
            for j in range(drug_maxlen):
                token = tokens[j]
                s = drugSeq_vocab[token]
                drug_seq.append(s)
        else:
            for j in range(len(tokens)):
                token = tokens[j]
                s = drugSeq_vocab[token]
                drug_seq.append(s)
            # 填充至最大长度
            drug_seq += [padding_value] * (drug_maxlen - len(tokens))
        seq_drugs.append(drug_seq)
    
    return key_drugs, seq_drugs

def load_target(dir_path, target_maxlen, targetSeq_vocab, padding_value=0):
    # 读取原始靶点序列
    proteins = json.load(open(os.path.join(dir_path, "proteins.txt")), object_pairs_hook=OrderedDict)  # OrderedDict保留键的顺序
    
    key_targets, seq_targets = [], []
    for key, seq in proteins.items():
        key_targets.append(key)

        target_seq = []
        if len(seq) >= target_maxlen:
            for j in range(target_maxlen):
                s = targetSeq_vocab[seq[j]]
                target_seq.append(s)
        else:
            for j in range(len(seq)):
                s = targetSeq_vocab[seq[j]]
                target_seq.append(s)
            # 填充至最大长度
            target_seq += [padding_value] * (target_maxlen - len(seq))
        seq_targets.append(target_seq)
    return key_targets, seq_targets

def load_affinity(dir_path, is_log=False):
    file_path = os.path.join(dir_path, "Y")
    # 从二进制文件中加载亲和力数据
    Y = pickle.load(open(file_path, "rb"), encoding='latin1')  # TODO: read from raw
    if is_log:
        Y = -np.log10(Y / 1e9)
    return Y

def get_samples(drugs, targets, Y):
    key_drugs, key_targets, affinities = [], [], []
    for a in range(len(drugs)):
        for b in range(len(targets)):
            if not np.isnan(Y[a, b]):  # 排除亲和力为NaN的情况
                key_drugs.append(drugs[a])
                key_targets.append(targets[b])
                affinities.append(Y[a, b])
    return key_drugs, key_targets, affinities

def split_train_test(key_drugs, key_targets, affinities, dir_path, test_size=1/6, random_state=None):
    # 将key_drugs, key_targets, affinities组合成一个DataFrame
    data = pd.DataFrame({
        'key_drug': key_drugs,
        'key_target': key_targets,
        'affinity': affinities
    })
    
    # 使用train_test_split划分数据集
    train_data, test_data = train_test_split(data, test_size=test_size, random_state=random_state)
    
    # 将训练集和测试集分别保存为CSV文件
    train_data.to_csv(os.path.join(dir_path, 'train_mapping.csv'), index=False)
    test_data.to_csv(os.path.join(dir_path, 'test_mapping.csv'), index=False)

if __name__ == '__main__':
    process_data('KIBA', drug_maxlen=100, target_maxlen=1000, random_state=42)
    process_data('Davis', drug_maxlen=85, target_maxlen=1200, random_state=42)

##########################################################################
