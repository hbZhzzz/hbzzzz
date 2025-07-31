import torch
import torch.nn as nn
import config
from modules import StimulusEncoder, LatentEncoder, EmotionGenerator

class EmotionCVAE(nn.Module):
    """完整的条件变分自编码器生成器模型"""
    def __init__(self):
        super().__init__()
        self.stimulus_encoder = StimulusEncoder()
        self.latent_encoder = LatentEncoder()
        self.emotion_generator = EmotionGenerator()

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, batch):
        """
        前向传播函数，在训练时调用。
        """

        video_features = batch['video_features']
        event_type = batch['event_type']
        va_values = batch['va_values']
        group_label = batch['group_label']
        stim_type = batch['stim_type']
        true_emo_seq = batch['true_emo_seq']
        

        stimulus_memory = self.stimulus_encoder(video_features, event_type, va_values, stim_type)
        mu, logvar = self.latent_encoder(true_emo_seq, group_label)
        z = self.reparameterize(mu, logvar)

        recon_logits = self.emotion_generator(z, group_label, stimulus_memory, true_emo_seq)
        
        return recon_logits, mu, logvar

    @torch.no_grad()
    def generate(self, batch):
        self.eval() 
        

        video_features = batch['video_features']
        event_type = batch['event_type']
        va_values = batch['va_values']
        stim_type = batch['stim_type']
        group_label = batch['group_label']
        batch_size = group_label.size(0)


        stimulus_memory = self.stimulus_encoder(video_features, event_type, va_values, stim_type)
        

        z = torch.randn(batch_size, config.LATENT_DIM, device=config.DEVICE)
        

        generated_probs = self.emotion_generator.generate(z, group_label, stimulus_memory)
        

        return generated_probs 