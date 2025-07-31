import os 
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import pandas as pd
import numpy as np
import random
from collections import defaultdict


import config
from feature_extractor import ResNet50FeatureExtractor

class EmotionDataset(Dataset):
    def __init__(self, stim_info_list, subject_info_list, feature_extractor, video_feat_dict=None):
        self.stim_df = pd.DataFrame(stim_info_list)     # list to df
        self.subject_df = pd.DataFrame(subject_info_list)
        self.feature_extractor = feature_extractor
        if video_feat_dict is None:
            self.video_feat_dict = torch.load(os.path.join(config.CODE_DIR, 'data/video_feat_dict.pt'))
        else:
            self.video_feat_dict = torch.load(video_feat_dict)
        # 快速查找，创建一个stim_lookup字典, 键为 full_name，值为对应的整行数据。
        self.stim_lookup = {row['full_name']: row for _, row in self.stim_df.iterrows()}

    def __len__(self):
        return len(self.subject_df)

    def __getitem__(self, idx):
        subject_data = self.subject_df.iloc[idx] # 根据索引 idx 提供数据项。
        
        # 匹配刺激信息
        stim_full_name = subject_data['full_name'].split('_', 1)[1] # e.g., '415-06014150_stim_02_HAS_1' -> 'stim_02_HAS_1'
        stim_data = self.stim_lookup.get(stim_full_name)
        
        if stim_data is None:
            raise ValueError(f"Stimulus info not found for {stim_full_name}")

        # 提取视频特征
        # video_features = self.feature_extractor(stim_data['video_path'], num_frames=config.SEQ_LEN)
        video_features = self.video_feat_dict[stim_data['video_path'].split('/')[-1]]
        
        # 编码类别特征
        event_type_encoded = torch.tensor(config.EVENT_TYPE_MAP[stim_data['event_type']], dtype=torch.float32)
        group_label_encoded = torch.tensor(config.GROUP_MAP[subject_data['label']], dtype=torch.long)
        stim_type = stim_full_name.split('_')[1] # eg., 'stim_02_HAS_1' -> 02
        stim_type_encoded = torch.tensor(config.STIM_TYPE_MAP[stim_type], dtype=torch.long)
        # 处理VA
        va_tensor = torch.tensor(stim_data['va'], dtype=torch.float32)

        # 真实情绪序列
        true_emo_seq = torch.from_numpy(subject_data['emo_numpy']).float()

        return {
            'video_features': video_features.to(config.DEVICE),          # [120, 2048]
            'event_type': event_type_encoded.to(config.DEVICE),          # scalar  
            'group_label': group_label_encoded.to(config.DEVICE),        # scalar
            'stim_type': stim_type_encoded.to(config.DEVICE),            # scalar
            'va_values': va_tensor.to(config.DEVICE),                    # [2]
            'true_emo_seq': true_emo_seq.to(config.DEVICE),              # [120, 7]
            'group_name': subject_data['label']        # for loss calculation
        }

class GroupBalancedSampler(Sampler): 
    """确保每个batch中两个组的样本数量相等"""
    def __init__(self, dataset, batch_size):  # 接受两个参数：dataset：数据集对象。batch_size：每个批次的样本数量
        if batch_size % 2 != 0:
            # 确保批次大小为偶数，以便能够平衡两个组的样本数量。
            raise ValueError("Batch size must be even for GroupBalancedSampler.") 
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.samples_per_group = batch_size // 2  # 每个组应包含的样本数
        
        # 使用 defaultdict 创建一个字典，键为组标签，值为对应样本的索引列表。通过迭代 DataFrame 的每一行填充这个字典。
        self.group_indices = defaultdict(list)
        for idx, row in self.dataset.subject_df.iterrows():
            self.group_indices[row['label']].append(idx)
        
        # 确定可以形成的批次数量，取决于两个组中样本数量较少的那个。每个批次包含 samples_per_group 个样本。
        self.num_batches = min(len(self.group_indices['Depressed']), len(self.group_indices['Control'])) // self.samples_per_group

    def __iter__(self): # 实现 __iter__ 方法，这个类可以被用于迭代。
        # 获取“抑郁”和“控制”组的样本索引。
        dep_indices = self.group_indices['Depressed']
        ctrl_indices = self.group_indices['Control']
        
        random.shuffle(dep_indices)
        random.shuffle(ctrl_indices)
        
        for i in range(self.num_batches): # 循环遍历每个批次
            # 计算每个组在当前批次中样本的起始索引
            start_dep = i * self.samples_per_group # self.samples_per_group是每个组应包含的样本数
            start_ctrl = i * self.samples_per_group
            
            # 组合批次索引：将Control和Depressed两个组的样本索引组合成一个批次索引列表
            batch_indices = (
                dep_indices[start_dep : start_dep + self.samples_per_group] +
                ctrl_indices[start_ctrl : start_ctrl + self.samples_per_group]
            )
            random.shuffle(batch_indices) # 再次打乱批次索引
            
            # 生成器输出：逐个返回组合后的批次中的样本索引。
            for idx in batch_indices:
                yield idx
                
    def __len__(self):
        return self.num_batches * self.batch_size  # 返回总的样本数量（两组个体数相同），即批次数乘以批次大小。


if __name__ == '__main__':

    pass