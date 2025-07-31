
import os 
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import cv2, av
import numpy as np
import config

class IVideoFeatureExtractor(nn.Module):
    """特征提取器接口，方便未来替换"""
    def __init__(self, device):
        super().__init__()
        self.device = device

    def forward(self, video_path):
        raise NotImplementedError

class ResNet50FeatureExtractor(IVideoFeatureExtractor):
    """使用ResNet50提取视频的逐帧特征"""
    def __init__(self, device):
        super().__init__(device)
        # 加载预训练的ResNet50，并去掉最后的分类层
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1) # 使用官方预训练参数
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1]).to(device)
        self.feature_extractor.eval()

        # 图像预处理流程
        # 使用 transforms.Compose 创建一个图像变换序列，包含：
        self.transform = transforms.Compose([  
            transforms.Resize(256), # 将图像缩放到 256x256。
            transforms.CenterCrop(224), # 从中心裁剪出 224x224 的图像。
            transforms.ToTensor(), # 将图像转换为张量。
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), # 对图像进行标准化，使用预训练模型的均值和标准差。
        ])

    def read_video_frames(self, video_path, num_frames=120):
        """从视频路径读取并抽样到指定帧数"""
        container = av.open(video_path)
        total_frames = container.streams.video[0].frames
        
        # 如果视频帧数不够，就循环播放
        indices = np.arange(total_frames)
        if total_frames < num_frames:
            indices = np.resize(indices, num_frames)
        else:
            # 简单的均匀抽样
            indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)

        frames = []
        for i, frame in enumerate(container.decode(video=0)):
            if i in indices:
                frames.append(frame.to_image())
                if len(frames) == num_frames:
                    break
        container.close()
        return frames

    @torch.no_grad()
    def forward(self, video_path, num_frames=120):
        frames = self.read_video_frames(video_path, num_frames)
        # print('frames:', len(frames))
        if not frames:
            # 返回一个零张量以处理错误
            print('Error in Frames')
            return torch.zeros((num_frames, config.VIDEO_FEAT_DIM), device=self.device)

        batch = torch.stack([self.transform(frame) for frame in frames]).to(self.device)
        features = self.feature_extractor(batch)
        features = features.view(features.size(0), -1) # [num_frames, config.VIDEO_FEAT_DIM] 
        return features

# 使用示例
if __name__ == '__main__':
    
    # video_dir = './stimuli/split_events_for_test/30FPS/'
    # extractor = ResNet50FeatureExtractor(device='cuda:1')

    # video_feat_dict = {}
    # for video in os.listdir(video_dir):
    #     video_path = os.path.join(video_dir, video)
    #     video_features = extractor(video_path)
    #     print(video, video_features.shape)
    #     video_feat_dict[video] = video_features
    # torch.save(video_feat_dict, './data/for_test/video_feat_dict.pt')
    pass
    
  

