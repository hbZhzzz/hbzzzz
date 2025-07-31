import torch
import torch.nn as nn
import torch.nn.functional as F 
import config
from utils import PositionalEncoding, generate_causal_mask

class StimulusEncoder(nn.Module):
    """将多种刺激信息融合成一个表征"""
    def __init__(self):
        super().__init__()
        # 视频特征的线性投射层
        self.video_proj = nn.Linear(config.VIDEO_FEAT_DIM, config.MODEL_DIM)
        # 刺激和事件类型编码的嵌入层
        self.stim_type_embedding = nn.Embedding(config.N_STIM_TYPES, config.MODEL_DIM)
        self.event_type_embedding = nn.Linear(2, config.MODEL_DIM)
        # VA值的线性层
        self.va_proj = nn.Linear(2, config.MODEL_DIM)
        
        self.layer_norm = nn.LayerNorm(config.MODEL_DIM)
        self.dropout = nn.Dropout(config.DROPOUT)
        
    def forward(self, video_features, event_type, va_values, stim_type):
        # video_features: [batch, seq_len, video_feat_dim]
        # event_type: [batch]
        # stim_type: [batch]
        # va_values: [batch, 2]
        
      
        video_encoded = self.video_proj(video_features) # [batch, seq_len, model_dim]
        
     
        event_type_encoded = self.event_type_embedding(event_type).unsqueeze(1).repeat(1, config.SEQ_LEN, 1) # [batch, seq_len, model_dim]
        stim_type_encoded = self.stim_type_embedding(stim_type).unsqueeze(1).repeat(1, config.SEQ_LEN, 1) # [batch, seq_len, model_dim]
        va_encoded = self.va_proj(va_values).unsqueeze(1).repeat(1, config.SEQ_LEN, 1) # [batch, seq_len, model_dim]

       
        fused_stimulus = video_encoded + event_type_encoded + va_encoded + stim_type_encoded
        fused_stimulus = self.dropout(self.layer_norm(fused_stimulus))
        
        return fused_stimulus.transpose(0, 1) # [seq_len, batch, model_dim] (Transformer的Position编码需要这个格式)
    

class LatentEncoder(nn.Module):
    """CVAE的编码器部分，推断潜变量分布"""
    def __init__(self):
        super().__init__()
        # 群体标签编码
        self.group_embedding = nn.Embedding(config.N_GROUPS, config.MODEL_DIM)
        
        # Transformer编码器层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.MODEL_DIM, 
            nhead=config.NUM_HEADS, 
            dim_feedforward=config.MODEL_DIM * 4, 
            dropout=config.DROPOUT,
            batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=config.NUM_LAYERS)
        
        self.pos_encoder = PositionalEncoding(config.MODEL_DIM, config.DROPOUT)
        
        # 输入融合层 (真实情感序列 + 群体标签)
        self.input_proj = nn.Linear(config.N_EMOTIONS, config.MODEL_DIM)
        
        # 输出层，用于生成均值μ和 对数方差log(σ^2)
        self.fc_mu = nn.Linear(config.MODEL_DIM, config.LATENT_DIM) # μ
        self.fc_logvar = nn.Linear(config.MODEL_DIM, config.LATENT_DIM) # 对数方差log(σ^2)

    def forward(self, true_emo_seq, group_label):
        # true_emo_seq: [batch, seq_len=120, n_emotions]
        # group_label: [batch]
        
       
        emo_proj = self.input_proj(true_emo_seq) # [batch, seq_len, model_dim]
        group_proj = self.group_embedding(group_label).unsqueeze(1).repeat(1, config.SEQ_LEN, 1)
        
        # 将情感序列和群体标签的嵌入相加，并转置以符合 Transformer 的输入格式。
        latent_input = (emo_proj + group_proj).transpose(0, 1) # [seq_len, batch, model_dim]
        # 对输入添加位置编码，以便模型能够利用序列中元素的位置信息。
        latent_input = self.pos_encoder(latent_input)
        
        
        # 将经过位置编码的输入传入 Transformer 编码器，得到编码后的表示。
        memory = self.transformer_encoder(latent_input) # [seq_len, batch, model_dim] 
        
        
        pooled_output = memory.mean(dim=0) # [batch, model_dim]
        mu = self.fc_mu(pooled_output)
        logvar = self.fc_logvar(pooled_output)  # [batch, LATENT_DIM]
        
        return mu, logvar

'''
关于变分自编码器的均值 mu 和 对数方差logvar
    1. 潜在变量分布 
    在变分自编码器（VAE）中，潜在变量通常被假设为服从多维正态分布。每个潜在变量的分布由均值 μ 和对数方差 log(σ^2) 描述。
    均值μ：表示潜在空间中每个维度的中心位置。
    对数方差  log(σ^2)：用于控制潜在空间中每个维度的扩展程度。

    2. 具体表示:
    当 LATENT_DIM = 32 时, mu 和 logvar 的每一行（对应一个 batch 中的样本）表示一个 32 维的潜在空间分布：
    mu[i] 是第 i 个样本的潜在变量分布的均值。
    logvar[i] 是第 i 个样本的潜在变量分布的对数方差。

    3. 生成潜在变量：
    在 VAE 中，生成潜在变量 Z 通常使用重参数化技巧：
    std = torch.exp(0.5 * logvar)  # 计算标准差
    epsilon = torch.randn_like(std)  # 生成标准正态分布的随机噪声
    z = mu + std * epsilon  # 生成潜在变量
'''    


class EmotionGenerator(nn.Module):
    """CVAE的解码器部分，生成情绪序列"""
    def __init__(self):
        super().__init__()
        self.group_embedding = nn.Embedding(config.N_GROUPS, config.MODEL_DIM) # 使用嵌入层将群体标签转换为固定维度的向量表示。
        self.latent_proj = nn.Linear(config.LATENT_DIM, config.MODEL_DIM) # 将潜在变量（z）的维度转换为模型维度。
        
        # 解码器输入，从上一帧的预测概率开始
        self.decoder_input_proj = nn.Linear(config.N_EMOTIONS, config.MODEL_DIM) # 将情感序列的维度转换为模型维度。
        
        # Transformer解码器层
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.MODEL_DIM, 
            nhead=config.NUM_HEADS, 
            dim_feedforward=config.MODEL_DIM * 4, 
            dropout=config.DROPOUT,
            batch_first=False
        )

        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.NUM_LAYERS)
        
        self.pos_encoder = PositionalEncoding(config.MODEL_DIM, config.DROPOUT) # 位置编码
        
        # 输出头
        self.output_head = nn.Linear(config.MODEL_DIM, config.N_EMOTIONS)

    def forward(self, z, group_label, stimulus_memory, true_emo_seq):
        """
        前向传播函数，在训练时使用Teacher Forcing。
        """
        # z: [batch, latent_dim]
        # group_label: [batch]
        # stimulus_memory: [seq_len, batch, model_dim]
        # true_emo_seq: [batch, seq_len, n_emotions]
        
  
 
        ## 将真实情感序列先进行投影，然后转置，以符合 Transformer 的输入格式，并添加位置编码。
        decoder_input = self.decoder_input_proj(true_emo_seq).transpose(0, 1) # [seq_len, batch, model_dim]
        decoder_input = self.pos_encoder(decoder_input)

     
        causal_mask = generate_causal_mask(config.SEQ_LEN, device=z.device) # 确保解码器在生成时只关注当前及之前的时间步。
        
     
        # 我们可以将它们融合后，加到解码器的每个输入上，或者作为初始token
        z_proj = self.latent_proj(z)
        group_proj = self.group_embedding(group_label)
        
        # 将潜在变量和群体标签的嵌入相加，形成一个静态上下文，然后使用加法将该信息融合到解码器的每个时间步。
        static_context = (z_proj + group_proj).unsqueeze(0) # [1, batch, model_dim]
        decoder_input = decoder_input + static_context # 这种融合可以帮助解码器在生成情感序列时，考虑到输入的上下文信息，从而生成更符合条件的输出。
        #  注意在这里，static_context是[1, batch, model_dim]， 而decoder_input是[seq_len, batch, model_dim]
        #  PyTorch 会将 static_context 在第一个维度（时间步）上进行扩展，使其形状变为 [seq_len, batch, model_dim]，这样它就能够与 decoder_input 的形状匹配。

 
        output = self.transformer_decoder(
            tgt=decoder_input,
            memory=stimulus_memory,
            tgt_mask=causal_mask
        ) # [seq_len, batch, model_dim]
        

        output_logits = self.output_head(output.transpose(0, 1)) # [batch, seq_len, n_emotions]
        
        return output_logits

    @torch.no_grad()
    def generate(self, z, group_label, stimulus_memory):
        """
        自回归生成函数，在推理/测试时使用。
        """
        self.eval() # 确保处于评估模式
        
        batch_size = z.size(0)
        seq_len = config.SEQ_LEN
        
 
        z_proj = self.latent_proj(z)
        group_proj = self.group_embedding(group_label)
        static_context = (z_proj + group_proj).unsqueeze(0) # [1, batch, model_dim]
        

        # 以一个全零的概率分布作为起始token <SOS>
        generated_probs = torch.zeros(batch_size, seq_len, config.N_EMOTIONS, device=z.device)
        
        for t in range(seq_len):

            ## 从 generated_probs 中提取已生成的序列（包括当前时间步 t）。即在第 t 次迭代中，输入包括从时间步 0 到 t 的所有生成概率。
            current_input_seq = generated_probs[:, :t+1, :] # [batch, t+1, n_emotions]
            
            # 将 current_input_seq 通过线性层投影到模型维度 model_dim， 并添加position编码
            decoder_input = self.decoder_input_proj(current_input_seq).transpose(0, 1) # [t+1, batch, model_dim]
            decoder_input = self.pos_encoder(decoder_input)
            
            #  融合静态上下文
            decoder_input = decoder_input + static_context 


            causal_mask = generate_causal_mask(t + 1, device=z.device)


            ## 将准备好的 decoder_input 和 stimulus_memory 传入 Transformer 解码器。
            output = self.transformer_decoder(
                tgt=decoder_input,
                memory=stimulus_memory[:t+1, :, :], # 只使用到当前时间步的刺激记忆
                tgt_mask=causal_mask
            ) # [t+1, batch, model_dim]



            last_step_output = output[-1, :, :] # [batch, model_dim]
            logits_t = self.output_head(last_step_output) # [batch, n_emotions]
            probs_t = F.softmax(logits_t, dim=-1)


            generated_probs[:, t, :] = probs_t
        
        return generated_probs # 返回生成的概率序列 [batch, seq_len, n_emotions]
    
class Discriminator(nn.Module):
    """判别器，判断序列真假
    暂未使用
    """
    def __init__(self):
        super().__init__()

        self.group_embedding = nn.Embedding(config.N_GROUPS, config.MODEL_DIM)
        self.input_proj = nn.Linear(config.N_EMOTIONS, config.MODEL_DIM)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.MODEL_DIM, nhead=config.NUM_HEADS, batch_first=False
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2) # 可以用更少的层
        
        self.pos_encoder = PositionalEncoding(config.MODEL_DIM, config.DROPOUT)
        

        self.classifier = nn.Sequential(
            nn.Linear(config.MODEL_DIM, config.MODEL_DIM // 2),
            nn.LeakyReLU(0.2),
            nn.Linear(config.MODEL_DIM // 2, 1)
        )

    def forward(self, emo_seq, group_label):
        # emo_seq: [batch, seq_len, n_emotions]
        # group_label: [batch]
        
        emo_proj = self.input_proj(emo_seq)
        group_proj = self.group_embedding(group_label).unsqueeze(1).repeat(1, config.SEQ_LEN, 1)
        
        disc_input = (emo_proj + group_proj).transpose(0, 1)
        disc_input = self.pos_encoder(disc_input)
        
        memory = self.transformer_encoder(disc_input)
        pooled_output = memory.mean(dim=0)
        
        validity = self.classifier(pooled_output)
        return validity


