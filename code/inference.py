import torch
import torch.nn.functional as F
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

import config
from model import EmotionCVAE
from baselines import ConditionalLSTM, ConditionalTCN, ConditionalTransformer, vqvae, Transformer, T2S_Emotion, MLP, DDPM, RectifiedFlow
from feature_extractor import ResNet50FeatureExtractor

def load_model(model_path, model_name, device):
    """加载训练好的生成器模型"""
    print(f"Loading model from {model_path}...")
    
    # 初始化模型结构
    if model_name == 'EmotionCVAE':
        model = EmotionCVAE().to(device)
        # 加载保存的权重
        checkpoint = torch.load(model_path, map_location=device)
        # 只需要加载生成器的状态字典
        model.load_state_dict(checkpoint['generator_state_dict'])
    
    elif model_name == 'ConditionalLSTM':
        model = ConditionalLSTM().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_name == 'ConditionalTCN':
        model = ConditionalTCN().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_name == 'ConditionalTransformer':
        model = ConditionalTransformer().to(device)
        checkpoint = torch.load(model_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    elif model_name == 'T2S':

        vae_model_path = os.path.join(model_path, 'best_T2S_vae_baseline.pth')
        inputs_model_path = os.path.join(model_path, 'best_T2S_inputs_baseline.pth')
        model_path = os.path.join(model_path, 'best_T2S_DiT_baseline.pth')


        vae_model = vqvae.vqvae().to(device)
        vae_model.load_state_dict(torch.load(vae_model_path)['model'])
        vae_model.eval()

        model_inputs = T2S_Emotion().to(device)
        model_inputs.load_state_dict(torch.load(inputs_model_path)['model'])
        model_inputs.eval()

        model = Transformer().to(device)
        model.encoder =  vae_model.encoder
        model.load_state_dict(torch.load(model_path)['model'])
        model.eval() 
        return vae_model, model_inputs, model

    else:
        raise(NotImplementedError)
    
    # 切换到评估模式
    model.eval()
    
    print("Model loaded successfully.")
    return model

def prepare_inference_input(stim_info, group_label, feature_extractor, device):
    """
    根据单次推理请求准备模型输入。
    
    Args:
        stim_info (dict): 包含单个刺激事件信息的字典。
                          e.g., {'event_type': 'NVS', 'va': (2.0, 7.0), 'video_path': '...'}
        group_label (str): 'Depressed' 或 'Control'
        feature_extractor: 初始化的视频特征提取器。
        device: 'cuda' 或 'cpu'

    Returns:
        dict: 一个包含模型所需输入的batch (batch_size=1)。
    """
    # print("Preparing input...")
    
    # 提取视频特征
    
    video_feat_dict = torch.load(os.path.join(config.CODE_DIR, 'data/video_feat_dict.pt'))
    if stim_info['video_path'].split('/')[-1] in video_feat_dict.keys():
        video_features = video_feat_dict[stim_info['video_path'].split('/')[-1]]
        # print('111')
    else:
        video_features = feature_extractor(stim_info['video_path'], num_frames=config.SEQ_LEN)
    
    # 编码类别特征
    event_type_encoded = torch.tensor(config.EVENT_TYPE_MAP[stim_info['event_type']], dtype=torch.float32)
    group_label_encoded = torch.tensor(config.GROUP_MAP[group_label], dtype=torch.long)
    stim_type_encoded = torch.tensor(config.STIM_TYPE_MAP[stim_info['stim_type']], dtype=torch.long)
    
    # 处理VA
    va_tensor = torch.tensor(stim_info['va'], dtype=torch.float32)

    # 将所有输入构造成一个batch (batch_size=1)
    batch = {
        'video_features': video_features.unsqueeze(0).to(device),
        'event_type': event_type_encoded.unsqueeze(0).to(device),
        'stim_type': stim_type_encoded.unsqueeze(0).to(device),
        'group_label': group_label_encoded.unsqueeze(0).to(device),
        'va_values': va_tensor.unsqueeze(0).to(device),
    }
    
    # print("Input prepared.")
    return batch

def generate_emotion_sequence(model, batch):
    """
    使用加载的模型进行单次推理。
    
    Args:
        model: 
        batch (dict): prepare_inference_input函数返回的输入batch。

    Returns:
        np.array: 生成的情绪概率序列，形状为 [120, 7]。
    """
    # print("Generating emotion sequence...")
    
    with torch.no_grad():
        # 调用模型的generate方法
        # 注意：在model.py中，我们让generate返回了概率，如果返回logits，这里需要加softmax
        generated_probs = model.generate(batch)
        
    # 将结果从GPU移到CPU，并转换为NumPy数组
    # generated_probs的形状是 [batch_size, seq_len, n_emotions]
    generated_sequence = generated_probs.squeeze(0).cpu().numpy()
    
    # print("Sequence generated successfully.")
    return generated_sequence


def generate_emotion_T2S(vae_model, model_inputs, model, batch):
    """
    使用加载的模型进行单次推理。
    
    Args:
        model: 
        batch (dict): prepare_inference_input函数返回的输入batch。

    Returns:
        np.array: 生成的情绪概率序列，形状为 [120, 7]。
    """
    original_seq_len = 120
    original_num_features = 7

    latent_seq_len = 30
    latent_dim = 64
    total_step = 100  # 去噪的总步数
    cfg_scale = 7.0 
    backbone_type = 'flowmatching'
    backbone = {'flowmatching': RectifiedFlow(), 'ddpm': DDPM(total_step, config.DEVICE)}.get(backbone_type)
    
    if backbone_type == 'flowmatching':
        rf = backbone
    elif backbone_type == 'ddpm':
        ddpm = backbone

    
    with torch.no_grad(): 
        for k, v in batch.items():
            if isinstance(v, torch.Tensor):
                batch[k] = v.to(config.DEVICE)
        
        true_emo_seq, inputs_cond = model_inputs(batch)
        batch_size = inputs_cond.shape[0]
        # z, before =  model.encoder(true_emo_seq)
        # x_1_latent = z.permute(0, 2, 1)
        
        x_t = torch.randn(batch_size, latent_seq_len, latent_dim).to(config.DEVICE)


        x_infer_list = []

        # print(f"开始 {total_step} 步去噪循环...")
        for j in range(total_step):
            if backbone_type == 'flowmatching':
                # 计算当前时间步 t
                t_val = j * 1.0 / total_step
                t = torch.full((batch_size,), t_val, device=config.DEVICE)
                
                # Classifier-Free Guidance
                pred_uncond = model(input=x_t, t=t, text_input=None)
                pred_cond = model(input=x_t, t=t, text_input=inputs_cond)
                
                # 结合预测
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                
                # 使用 Euler 方法更新 x_t，
                x_t = rf.euler(x_t, pred, 1.0 / total_step )

            elif backbone_type == 'ddpm':
                # DDPM 的时间步是从 T-1 到 0
                t_val = total_step - 1 - j
                t = torch.full((batch_size,), t_val, dtype=torch.long, device=config.DEVICE)
                
                pred_uncond = model(input=x_t, t=t, text_input=None)
                pred_cond = model(input=x_t, t=t, text_input=inputs_cond)
                pred = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
                
                # 使用 DDPM 的 p_sample 更新 x_t
                x_t = ddpm.p_sample(x_t, pred, t)
            

            if j % 10 == 0 or j == total_step - 1:
                
                
                x_t_permuted = x_t.permute(0, 2, 1) # VAE decoder 需要 [B, C, L]
                decoded_intermediate, _ = vae_model.decoder(x_t_permuted, length=120)
                x_infer_list.append(decoded_intermediate[0].detach().cpu().numpy()) 

    
        
        # 循环结束后，x_t 是最终生成的潜向量
        x_generated_latent = x_t.clone()
        # print(f"最终生成的潜向量 `x_generated_latent` 形状: {x_generated_latent.shape}")
        x_generated_latent_permuted = x_generated_latent.permute(0, 2, 1)
        generated_series, _ = vae_model.decoder(x_generated_latent_permuted, length=original_seq_len)
        
        generated_series = generated_series.permute(1,0)
        probabilities = F.softmax(generated_series, dim=1)
    
    generated_sequence = probabilities.cpu().numpy()
    
    # print("Sequence generated successfully.")
    return generated_sequence


def plot_generated_sequence(sequence, title="Generated Emotion Sequence", figsize=None):
    """
    可视化生成的情绪序列。
    
    Args:
        sequence (np.array): 形状为 [120, 7] 的情绪概率序列。
        title (str): 图表标题。
    """
    if figsize == None:
        plt.figure(figsize=(14, 10))
    else:
        plt.figure(figsize=figsize)
    
    emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
    
    for i, label in enumerate(emotion_labels):
        plt.plot(sequence[:, i], label=label)
        
    # 在第60帧处画一条垂直线，表示事件关键点
    plt.axvline(x=config.BASELINE_FRAMES, color='r', linestyle='--', label='Stimulus Onset')
    
    plt.title(title, fontsize=16)
    plt.xlabel("Frame (Time)", fontsize=12)
    plt.ylabel("Probability", fontsize=12)
    plt.ylim(0, 1)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()





# --- 主执行函数，用于演示 ---
if __name__ == '__main__':
  
    # 

    pass

