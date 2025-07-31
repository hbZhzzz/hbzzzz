# config.py

import torch

# --- DIR ---
CODE_DIR = './code'

# --- 数据相关配置 ---
SEQ_LEN = 120
N_EMOTIONS = 7
BASELINE_FRAMES = 60
RECOVERY_FRAMES = 30 # 最后30帧


# --- 模型维度配置 ---
MODEL_DIM = 256
VIDEO_FEAT_DIM = 2048 # ResNet50输出的特征维度 FEAT(feature)
LATENT_DIM = 64 # 潜变量z的维度
NUM_HEADS = 8
NUM_LAYERS = 8
DROPOUT = 0.1

# --- 训练配置 ---
DEVICE = "cuda:1" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 16 
LEARNING_RATE_G = 1e-4 # 生成器学习率
LEARNING_RATE_D = 1e-4 # 判别器学习率
EPOCHS = 100
PRINT_FREQ = 10

# --- 损失函数权重 ---
LAMBDA_FEATURE = 1.0   # L_feature 的权重  
DEAD_ZONE = 0.1 #
ALPHA_KL_MAX = 0.0001 # KL退火的最大值 
BETA_ADV = 0.0      # 对抗损失的权重,  
GAMMA_FEATURE_MIX = 0.5 # L_feature中，条件损失和差异损失的混合比例 
DELTA_SCALE = 0.5 # L_scale损失的权重  
ETA_VA = 0.1145 # L_va_consistency的权重 

# --- Free Bits 配置 ---
USE_FREE_BITS = True       # 是否启用Free Bits
FREE_BITS_EPSILON = 2.0   # 尝试过的序列：0.1 --> 1.0 --> 2.0

# 特征损失内部权重 
W_HAPPY_REACTIVITY_NORMAL = 2.0
W_SAD_RECOVERY_DEPRESSED = 2.0

W_DEFAULT = 1.0

# --- 类别映射 ---
GROUP_MAP = {'Control': 0, 'Depressed': 1}
STIM_TYPE_MAP = {'01': 0, '02': 1, '03' : 2, '04':0 }  # 01-NEUTRAL, 02-POSITIVE, 03-NEGATIVE
Valence_Shift = {'NVS':-1, 'NO_SHIFT':0, 'PVS':1}
Arousal_Shift = {'LAS':-1, 'NO_SHIFT':0, 'HAS':1}

EVENT_TYPE_MAP = {'NVS': (Valence_Shift['NVS'],Arousal_Shift['NO_SHIFT']), 
                  'PVS': (Valence_Shift['PVS'],Arousal_Shift['NO_SHIFT']), 
                  'HAS': (Valence_Shift['NO_SHIFT'],Arousal_Shift['HAS']), 
                  'LAS': (Valence_Shift['NO_SHIFT'],Arousal_Shift['LAS']),
                  'HAS-PVS':(Valence_Shift['PVS'],Arousal_Shift['HAS']),
                  'LAS-PVS':(Valence_Shift['PVS'],Arousal_Shift['LAS']),
                  'HAS-NVS':(Valence_Shift['NVS'],Arousal_Shift['HAS']),
                  'LAS-NVS':(Valence_Shift['NVS'],Arousal_Shift['LAS']),
                  }



N_GROUPS = len(GROUP_MAP)
N_EVENT_TYPES = len(EVENT_TYPE_MAP)
N_STIM_TYPES = len(STIM_TYPE_MAP)


CLASS_NAMES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
NUM_CLASSES = len(CLASS_NAMES)

# 来自于Warriner等人的工作：Norms of valence, arousal, and dominance for 13,915 English lemmas
EMOTION_TO_VA_MAP = torch.tensor([
    [2.53, 6.2], # Angry (V:-, A:+)
    [3.32, 5.0], # Disgust (V:-, A:+)
    [2.93, 6.14], # Fear (V:-, A:+)
    [8.47, 6.05], # Happy (V:+, A:+)
    [5.5, 3.45], # Neutral (V:~, A:-)
    [2.1, 3.49], # Sad (V:-, A:-)
    [7.44, 6.57]  # Surprise (V:+, A:+)
], device=DEVICE)

DEPRESSED_VA_BIAS = torch.tensor([-2.0, -1.0], device=DEVICE)  # 为抑郁组的VA损失添加偏置，以防惩罚模型为抑郁组生成过于积极的情绪
