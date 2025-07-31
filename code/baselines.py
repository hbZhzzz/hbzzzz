import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.weight_norm as weight_norm
from utils import PositionalEncoding, generate_causal_mask 
import config
from baseline_T2S.model.backbone.rectified_flow import RectifiedFlow
from baseline_T2S.model.backbone.DDPM import DDPM
from baseline_T2S.model.denoiser.transformer import Transformer
from baseline_T2S.model.denoiser.mlp import MLP
from baseline_T2S.model.pretrained import vqvae
# --- Conditional LSTM (CLSTM) ---

class ConditionalLSTM(nn.Module):
    """
    一个条件LSTM基线模型。
    所有条件信息被用来初始化LSTM的隐状态。
    """
    def __init__(self):
        super().__init__()
        
        
        self.group_embedding = nn.Embedding(config.N_GROUPS, config.MODEL_DIM // 4)  ## config.MODEL_DIM // 4 是为了后续拼接起来刚好等于256
        self.event_type_embedding = nn.Linear(2, config.MODEL_DIM // 4)
        self.va_proj = nn.Linear(2, config.MODEL_DIM // 4)

        self.video_proj = nn.Linear(config.VIDEO_FEAT_DIM, config.MODEL_DIM // 4)
        
       
        # LSTM隐状态维度 (num_layers, batch_size, hidden_size)
        lstm_hidden_size = config.MODEL_DIM
        self.lstm_layers = 2
        

        self.init_h = nn.Linear(config.MODEL_DIM, self.lstm_layers * lstm_hidden_size)
        self.init_c = nn.Linear(config.MODEL_DIM, self.lstm_layers * lstm_hidden_size)

        self.lstm_input_proj = nn.Linear(config.N_EMOTIONS, lstm_hidden_size)
        self.lstm = nn.LSTM(
            input_size=lstm_hidden_size,
            hidden_size=lstm_hidden_size,
            num_layers=self.lstm_layers,
            batch_first=True, # [batch, seq_len, features]
            dropout=config.DROPOUT if self.lstm_layers > 1 else 0
        )
        
        self.output_head = nn.Linear(lstm_hidden_size, config.N_EMOTIONS)

    def forward(self, batch):
        """
        前向传播函数, 在训练时使用Teacher Forcing。
        """

        video_features = batch['video_features']    # [batch, seq_len, feat_dim]
        event_type = batch['event_type']            # [batch]
        va_values = batch['va_values']              # [batch, 2]
        group_label = batch['group_label']          # [batch]
        true_emo_seq = batch['true_emo_seq']        # [batch, seq_len, n_emotions]
        batch_size = group_label.size(0)


        video_mean_feat = video_features.mean(dim=1)
        
        video_cond = self.video_proj(video_mean_feat)
        event_cond = self.event_type_embedding(event_type)
        va_cond = self.va_proj(va_values)
        group_cond = self.group_embedding(group_label)
        

        context_vector = torch.cat([video_cond, event_cond, va_cond, group_cond], dim=1) # [batch, model_dim] ， 前面model_dim // 4 的原因
        


        h_0 = self.init_h(context_vector)
        c_0 = self.init_c(context_vector)
        

        h_0 = h_0.view(batch_size, self.lstm_layers, -1).permute(1, 0, 2).contiguous()
        c_0 = c_0.view(batch_size, self.lstm_layers, -1).permute(1, 0, 2).contiguous()


        start_token = torch.zeros(batch_size, 1, config.N_EMOTIONS, device=config.DEVICE)

        lstm_input_seq = torch.cat([start_token, true_emo_seq[:, :-1, :]], dim=1)
        
        lstm_input_proj = self.lstm_input_proj(lstm_input_seq)
        

        lstm_output, _ = self.lstm(lstm_input_proj, (h_0, c_0))
        

        logits = self.output_head(lstm_output) # [batch, seq_len, n_emotions]
        
        return logits
    
    @torch.no_grad()
    def generate(self, batch):
        """
        自回归生成函数。
        """
        self.eval()
        

        video_features = batch['video_features']
        event_type = batch['event_type']
        va_values = batch['va_values']
        group_label = batch['group_label']
        batch_size = group_label.size(0)


        video_mean_feat = video_features.mean(dim=1)
        video_cond = self.video_proj(video_mean_feat)
        event_cond = self.event_type_embedding(event_type)
        va_cond = self.va_proj(va_values)
        group_cond = self.group_embedding(group_label)
        context_vector = torch.cat([video_cond, event_cond, va_cond, group_cond], dim=1)


        h_t = self.init_h(context_vector).view(batch_size, self.lstm_layers, -1).permute(1, 0, 2).contiguous()
        c_t = self.init_c(context_vector).view(batch_size, self.lstm_layers, -1).permute(1, 0, 2).contiguous()



        current_input_prob = torch.zeros(batch_size, 1, config.N_EMOTIONS, device=config.DEVICE)
        generated_probs_list = []

        for _ in range(config.SEQ_LEN):
            lstm_input_proj = self.lstm_input_proj(current_input_prob)
            
            lstm_output, (h_t, c_t) = self.lstm(lstm_input_proj, (h_t, c_t))
            
            logits_t = self.output_head(lstm_output)
            probs_t = F.softmax(logits_t, dim=-1) # [batch, 1, n_emotions]
            
            generated_probs_list.append(probs_t)
            
            current_input_prob = probs_t
        

        generated_probs = torch.cat(generated_probs_list, dim=1) # [batch, seq_len, n_emotions]
        
        return generated_probs
    
# --- Conditional TCN (CTCN) ---

class TCNResidualBlock(nn.Module):
    """TCN的残差块"""
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, dropout):
        super(TCNResidualBlock, self).__init__()

        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=(kernel_size-1) * dilation, dilation=dilation))
        self.chomp1 = nn.ConstantPad1d((0, -(kernel_size-1) * dilation), 0) 
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=(kernel_size-1) * dilation, dilation=dilation))
        self.chomp2 = nn.ConstantPad1d((0, -(kernel_size-1) * dilation), 0)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        
    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class ConditionalTCN(nn.Module):
    """
    一个条件TCN基线模型。
    条件信息作为额外通道与输入序列拼接。
    """
    def __init__(self, num_channels=[config.MODEL_DIM] * 3, kernel_size=3, dropout=config.DROPOUT):
        super(ConditionalTCN, self).__init__()
        
        cond_dim_part = config.MODEL_DIM // 4
        self.group_embedding = nn.Embedding(config.N_GROUPS, cond_dim_part)
        self.event_type_embedding =  nn.Linear(2, cond_dim_part)
        self.va_proj = nn.Linear(2, cond_dim_part)
        self.video_proj = nn.Linear(config.VIDEO_FEAT_DIM, cond_dim_part)
        

        self.tcn_input_proj = nn.Linear(config.N_EMOTIONS, num_channels[0])
        

        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i 
            in_channels = num_channels[i-1] if i > 0 else num_channels[0] + config.MODEL_DIM # 输入通道 = 情绪概率通道 + 条件通道
            out_channels = num_channels[i]
            

            if i == 0:
                in_channels = self.tcn_input_proj.out_features + config.MODEL_DIM

            layers.append(TCNResidualBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size, dropout=dropout))
        
        self.tcn_net = nn.Sequential(*layers)
        

        self.output_head = nn.Linear(num_channels[-1], config.N_EMOTIONS)
    
    def forward(self, batch):
        """
        前向传播函数，在训练时使用Teacher Forcing。
        """

        video_features = batch['video_features']
        event_type = batch['event_type']
        va_values = batch['va_values']
        group_label = batch['group_label']
        true_emo_seq = batch['true_emo_seq']


        video_mean_feat = video_features.mean(dim=1)
        video_cond = self.video_proj(video_mean_feat)
        event_cond = self.event_type_embedding(event_type)
        va_cond = self.va_proj(va_values)
        group_cond = self.group_embedding(group_label)
        context_vector = torch.cat([video_cond, event_cond, va_cond, group_cond], dim=1) # [batch, model_dim]
        

        start_token = torch.zeros(true_emo_seq.size(0), 1, config.N_EMOTIONS, device=config.DEVICE)
        tcn_input_seq = torch.cat([start_token, true_emo_seq[:, :-1, :]], dim=1)
        tcn_input_proj = self.tcn_input_proj(tcn_input_seq)
        

        context_expanded = context_vector.unsqueeze(1).repeat(1, config.SEQ_LEN, 1) # [batch, seq_len, model_dim]
        tcn_input_combined = torch.cat([tcn_input_proj, context_expanded], dim=2)
        

        tcn_input_transposed = tcn_input_combined.transpose(1, 2)
        

        tcn_output = self.tcn_net(tcn_input_transposed)
        


        tcn_output_transposed = tcn_output.transpose(1, 2)
        logits = self.output_head(tcn_output_transposed)
        
        return logits
        
    @torch.no_grad()
    def generate(self, batch):
        """
        自回归生成函数
        """
        self.eval()
        

        video_features, event_type, va_values, group_label = batch['video_features'], batch['event_type'], batch['va_values'], batch['group_label']
        batch_size = group_label.size(0)
        
        video_mean_feat = video_features.mean(dim=1)
        video_cond = self.video_proj(video_mean_feat)
        event_cond = self.event_type_embedding(event_type)
        va_cond = self.va_proj(va_values)
        group_cond = self.group_embedding(group_label)
        context_vector = torch.cat([video_cond, event_cond, va_cond, group_cond], dim=1)
        context_expanded = context_vector.unsqueeze(1).repeat(1, config.SEQ_LEN, 1)


        generated_probs = torch.zeros(batch_size, config.SEQ_LEN, config.N_EMOTIONS, device=config.DEVICE)
        
        for t in range(config.SEQ_LEN):


            current_input_seq = generated_probs
            
            tcn_input_proj = self.tcn_input_proj(current_input_seq)
            tcn_input_combined = torch.cat([tcn_input_proj, context_expanded], dim=2)
            tcn_input_transposed = tcn_input_combined.transpose(1, 2)
            

            tcn_output = self.tcn_net(tcn_input_transposed)
            tcn_output_transposed = tcn_output.transpose(1, 2)
            

            logits_t = self.output_head(tcn_output_transposed[:, t, :])
            probs_t = F.softmax(logits_t, dim=-1)
            

            generated_probs[:, t, :] = probs_t
        
        return generated_probs


# --- Conditional Transformer (CTransformer) ---
class ConditionalTransformer(nn.Module):
    """
    一个条件Transformer基线模型。

    """
    def __init__(self):
        super().__init__()
        
        self.video_proj = nn.Linear(config.VIDEO_FEAT_DIM, config.MODEL_DIM)

        self.group_embedding = nn.Embedding(config.N_GROUPS, config.MODEL_DIM)
        self.event_type_embedding = nn.Linear(2, config.MODEL_DIM)
        self.va_proj = nn.Linear(2, config.MODEL_DIM)
        

        self.decoder_input_proj = nn.Linear(config.N_EMOTIONS, config.MODEL_DIM)
        self.pos_encoder = PositionalEncoding(config.MODEL_DIM, config.DROPOUT)
        

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.MODEL_DIM, 
            nhead=config.NUM_HEADS, 
            dim_feedforward=config.MODEL_DIM * 4, 
            dropout=config.DROPOUT,
            batch_first=False 
        )
        self.transformer_decoder = nn.TransformerDecoder(decoder_layer, num_layers=config.NUM_LAYERS)
        

        self.output_head = nn.Linear(config.MODEL_DIM, config.N_EMOTIONS)
        
    def forward(self, batch):
        """
        前向传播函数
        """
        # --- 提取输入 ---
        video_features = batch['video_features']    # [batch, seq_len, feat_dim]
        event_type = batch['event_type']            # [batch]
        va_values = batch['va_values']              # [batch, 2]
        group_label = batch['group_label']          # [batch]
        true_emo_seq = batch['true_emo_seq']        # [batch, seq_len, n_emotions]
        

        stimulus_memory = self.video_proj(video_features).transpose(0, 1) # [seq_len, batch, model_dim]
        

        decoder_input = self.decoder_input_proj(true_emo_seq).transpose(0, 1) # [seq_len, batch, model_dim]
        decoder_input = self.pos_encoder(decoder_input)
        

        event_cond = self.event_type_embedding(event_type).unsqueeze(0) # [1, batch, model_dim]
        va_cond = self.va_proj(va_values).unsqueeze(0)
        group_cond = self.group_embedding(group_label).unsqueeze(0)
        

        static_context = event_cond + va_cond + group_cond
        decoder_input = decoder_input + static_context
        

        causal_mask = generate_causal_mask(config.SEQ_LEN, device=config.DEVICE)
        

        output = self.transformer_decoder(
            tgt=decoder_input,
            memory=stimulus_memory,
            tgt_mask=causal_mask
        ) # [seq_len, batch, model_dim]
        

        logits = self.output_head(output.transpose(0, 1)) # [batch, seq_len, n_emotions]
        
        return logits

    @torch.no_grad()
    def generate(self, batch):
        """
        自回归生成函数
        """
        self.eval()
        

        video_features = batch['video_features']
        event_type = batch['event_type']
        va_values = batch['va_values']
        group_label = batch['group_label']
        batch_size = group_label.size(0)

      
        stimulus_memory = self.video_proj(video_features).transpose(0, 1)
        event_cond = self.event_type_embedding(event_type).unsqueeze(0)
        va_cond = self.va_proj(va_values).unsqueeze(0)
        group_cond = self.group_embedding(group_label).unsqueeze(0)
        static_context = event_cond + va_cond + group_cond
        

        generated_probs = torch.zeros(batch_size, config.SEQ_LEN, config.N_EMOTIONS, device=config.DEVICE)
        
        for t in range(config.SEQ_LEN):

            current_input_seq = generated_probs[:, :t+1, :] # [batch, t+1, n_emotions]
            decoder_input = self.decoder_input_proj(current_input_seq).transpose(0, 1) # [t+1, batch, model_dim]
            decoder_input = self.pos_encoder(decoder_input)
            decoder_input = decoder_input + static_context # 注入条件
            

            causal_mask = generate_causal_mask(t + 1, device=config.DEVICE)
            

            output = self.transformer_decoder(
                tgt=decoder_input,
                memory=stimulus_memory[:t+1, :, :],
                tgt_mask=causal_mask
            ) # [t+1, batch, model_dim]
            

            logits_t = self.output_head(output[-1, :, :]) # [batch, model_dim]
            probs_t = F.softmax(logits_t, dim=-1)
            

            generated_probs[:, t, :] = probs_t

        return generated_probs



# T2S
class T2S_Emotion(nn.Module):
    """

    """
    def __init__(self):
        super().__init__()
        
        self.model_dim = 128
        
        # 视频特征的投射
        self.video_proj = nn.Linear(config.VIDEO_FEAT_DIM,  self.model_dim)
        # 类别和连续条件的嵌入/投射
        self.group_embedding = nn.Embedding(config.N_GROUPS, self.model_dim)
        self.event_type_embedding = nn.Linear(2,  self.model_dim)
        self.va_proj = nn.Linear(2,  self.model_dim)
        
        self.static_to_t2s_inputs_cond = nn.Linear(120*self.model_dim,self.model_dim)

        
        # T2S 模型相关参数
        self.backbone_type = 'flowmatching' # 'flowmatching' 或 'ddpm'
        self.denoiser_type = 'DiT'
        self.total_step = 100 # 
        self.latent_seq_len = 30 
        self.latent_dim = 64

        self.model = {'DiT': Transformer, 'MLP': MLP}.get(self.denoiser_type)
        self.model = self.model().to(config.DEVICE)
        vae_model = vqvae.vqvae().to(config.DEVICE)
        self.encoder =  vae_model.encoder
        
    def forward(self, batch):
        """
        前向传播函数
        """
        

        video_features = batch['video_features']    # [batch, seq_len, feat_dim]
        event_type = batch['event_type']            # [batch]
        va_values = batch['va_values']              # [batch, 2]
        group_label = batch['group_label']          # [batch]
        if 'true_emo_seq' in batch.keys():     
            true_emo_seq = batch['true_emo_seq']        # [batch, seq_len, n_emotions]
        else:
            true_emo_seq = None
        fact_bs = video_features.shape[0]

        stimulus_memory = self.video_proj(video_features).transpose(0, 1) # [seq_len, batch, model_dim]
        event_cond = self.event_type_embedding(event_type).unsqueeze(0) # [1, batch, model_dim]
        va_cond = self.va_proj(va_values).unsqueeze(0) # [1, batch, model_dim]
        group_cond = self.group_embedding(group_label).unsqueeze(0) # [1, batch, model_dim]
        # print(f'event_cond:{event_cond.shape}, va_cond:{va_cond.shape}, group_cond:{group_cond.shape}')
        
        
        static_context = event_cond + va_cond + group_cond + stimulus_memory  # [seq_len, batch, model_dim]
        static_context = static_context.permute(1, 0, 2) # [batch, seq_len, model_dim] 
        # print(true_emo_seq.shape)
        # print(static_context.shape)
        inputs_cond = self.static_to_t2s_inputs_cond(static_context.view(fact_bs, -1))
        # print('inputs:', inputs_cond.shape)
     
        
        
        return true_emo_seq, inputs_cond















