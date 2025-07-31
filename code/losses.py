import torch
import torch.nn.functional as F
import config

# --- 特征提取函数 ---
def calculate_reactivity(emo_prob_seq):
    # emo_prob_seq: [batch, seq_len] for a single emotion
    # reactivity是个体的峰值概率减去基线状态下的平均概率后的值
    baseline = emo_prob_seq[:, :config.BASELINE_FRAMES].mean(dim=1)
    peak = emo_prob_seq.max(dim=1).values
    return peak - baseline


def calculate_recovery(emo_prob_seq):
    # emo_prob_seq: [batch, seq_len] for a single emotion
    peak = emo_prob_seq.max(dim=1).values
    recovery_phase = emo_prob_seq[:, -config.RECOVERY_FRAMES:].mean(dim=1)
    return recovery_phase - peak    

# --- 损失计算 ---
def loss_fn_scale(pred_seq, true_seq):
    # pred_seq, true_seq: [batch, seq_len, n_emotions]
    
    # 计算每个情绪通道的平均概率
    # 使用 .mean(dim=[0, 1]) 方法沿着第一个（batch）和第二个（seq_len）维度计算平均值
    mean_pred_probs = pred_seq.mean(dim=[0, 1]) # [n_emotions]
    mean_true_probs = true_seq.mean(dim=[0, 1]) # [n_emotions]
    
    # 使用L1损失计算差异
    # L1 损失计算的是两个张量之间的绝对差值的平均值，反映了预测和真实值之间的差异。
    l_scale = F.l1_loss(mean_pred_probs, mean_true_probs)
    return l_scale
    

def loss_fn_feature(pred_seq, true_seq, group_labels):
    
    # pred_seq, true_seq: [batch, seq_len, n_emotions]
    # group_labels: [batch] 0 for Control, 1 for Depressed
    
    # --- 1. 条件化特征损失 ---
    l_feature_cond = 0.0 # 初始化条件化特征损失 = 0.0
    
    # Happy Reactivity：分别计算预测值和真实值
    happy_idx = 3 # 根据EMOTION_LABELS
    pred_happy_react = calculate_reactivity(pred_seq[:, :, happy_idx])
    true_happy_react = calculate_reactivity(true_seq[:, :, happy_idx])
    
    # Sad Recovery：分别计算预测值和真实值
    sad_idx = 5
    pred_sad_recovery = calculate_recovery(pred_seq[:, :, sad_idx])
    true_sad_recovery = calculate_recovery(true_seq[:, :, sad_idx])
    
    # 控制组损失
    mask_norm = (group_labels == config.GROUP_MAP['Control']) # 通过比较 group_labels 和控制组的标签，创建一个布尔掩码 mask_norm，标记哪些样本属于控制组。
    
    # 如果存在属于控制组的样本，进入条件语句。
    if mask_norm.any():
        # 使用加权 L1 损失（F.l1_loss）来计算快乐反应的损失和悲伤恢复的损失
        # config.W_HAPPY_REACTIVITY_NORMAL 和 config.W_DEFAULT 是权重，用于调整不同损失对总损失的贡献。
        # pred_happy_react[mask_norm] 和 true_happy_react[mask_norm] 仅取控制组样本的快乐反应
        # 同样，计算悲伤恢复的损失。
        loss_norm = (
            config.W_HAPPY_REACTIVITY_NORMAL * F.l1_loss(pred_happy_react[mask_norm], true_happy_react[mask_norm]) +
            config.W_DEFAULT * F.l1_loss(pred_sad_recovery[mask_norm], true_sad_recovery[mask_norm])
        )
        l_feature_cond += loss_norm

    # 抑郁组损失
    mask_dep = (group_labels == config.GROUP_MAP['Depressed']) # 创建抑郁组掩码
    if mask_dep.any():
        loss_dep = (
            config.W_DEFAULT * F.l1_loss(pred_happy_react[mask_dep], true_happy_react[mask_dep]) +
            config.W_SAD_RECOVERY_DEPRESSED * F.l1_loss(pred_sad_recovery[mask_dep], true_sad_recovery[mask_dep])
        )
        l_feature_cond += loss_dep
        
    # --- 2. 组间差异损失 ---
    l_feature_diff = 0.0 # 初始化组间差异损失=0.0
    if mask_norm.any() and mask_dep.any():
        # 只有batch中包含两个群体类别的样本时，才计算组间差异，否则该batch的组间差异为0
        # Happy Reactivity 差异
        true_diff_happy = true_happy_react[mask_norm].mean() - true_happy_react[mask_dep].mean()
        pred_diff_happy = pred_happy_react[mask_norm].mean() - pred_happy_react[mask_dep].mean()
        l_feature_diff += F.l1_loss(pred_diff_happy, true_diff_happy)
        # error_diff_happy = F.l1_loss(pred_diff_happy, true_diff_happy, reduction='none')
        # l_feature_diff += F.relu(error_diff_happy - config.DEAD_ZONE).mean()
       
        
        # Sad Recovery 差异
        true_diff_sad = true_sad_recovery[mask_norm].mean() - true_sad_recovery[mask_dep].mean()
        pred_diff_sad = pred_sad_recovery[mask_norm].mean() - pred_sad_recovery[mask_dep].mean()
        l_feature_diff += F.l1_loss(pred_diff_sad, true_diff_sad)
        # error_diff_sad = F.l1_loss(pred_diff_sad, true_diff_sad, reduction='none')
        # l_feature_diff += F.relu(error_diff_sad - config.DEAD_ZONE).mean()
        
    
    # --- 3. 混合 ---
    return config.GAMMA_FEATURE_MIX * l_feature_cond + (1 - config.GAMMA_FEATURE_MIX) * l_feature_diff


def loss_fn_discriminator(D_real_output, D_fake_output):
    # 简单的GAN损失
    l_real = torch.mean(F.relu(1.0 - D_real_output))
    l_fake = torch.mean(F.relu(1.0 + D_fake_output))
    return (l_real + l_fake) / 2


def loss_fn_va_consistency(pred_seq_probs, stim_va, group_labels):
    # pred_seq_probs: [Batch, Seq_Len, 7]
    # stim_va: [Batch, 2] - 原始刺激的VA值
    
    # 计算整个序列的平均感知VA
    # (矩阵乘法: [Batch, Seq_Len, 7] @ [7, 2] -> [Batch, Seq_Len, 2])
    avg_perceived_va = (pred_seq_probs @ config.EMOTION_TO_VA_MAP).mean(dim=1) # [Batch, 2]
    
    target_va = stim_va.clone() # 先复制一份
    
    # 找到抑郁组的样本
    mask_dep = (group_labels == config.GROUP_MAP['Depressed'])
    
    # 对抑郁组的目标VA应用偏置
    if mask_dep.any():
        target_va[mask_dep] = target_va[mask_dep] + config.DEPRESSED_VA_BIAS
        
        # 可选：限制VA值在合理范围，如[1, 9]
        target_va.clamp_(min=1.0, max=9.0)


    # 使用L1损失计算差异
    return F.l1_loss(avg_perceived_va, target_va)



def loss_fn_generator(recon_logits, true_seq, mu, logvar, D_fake_output, group_labels, stim_va):
    # 1. 重构损失 - L_distribution (交叉熵)
    # 预测的是logits，真实的是概率，所以用CrossEntropyLoss很合适
    # 但真实标签是软标签，所以需要手动计算
    recon_probs = F.softmax(recon_logits, dim=-1)  # 将重构情绪的logits转换为概率
    l_distribution = - (true_seq * torch.log(recon_probs + 1e-9)).sum(dim=-1).mean()  # 除数是 batch * seq_len

    # 2. 重构损失 - L_feature
    l_feature = loss_fn_feature(recon_probs, true_seq, group_labels)

    # 3. KL散度损失
    if config.USE_FREE_BITS:
        # free bits
        # --- 使用 Free Bits 技术 ---
        # 逐个维度计算KL散度: KL(q(z_i|x) || p(z_i))
        # 对于高斯分布，其解析解是 -0.5 * (1 + log(σ_i^2) - μ_i^2 - σ_i^2)
        kl_per_dim = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()) # shape: [batch_size, latent_dim]
        
        # 应用Free Bits: 对每个维度，如果KL小于epsilon，就将其钳位到epsilon
        # 这意味着损失至少是epsilon，模型没有动力把它压到更低
        # 更好的实现是 max(kl_per_dim, epsilon)
        # torch.clamp(input, min, max, out=None)将输入input张量每个元素的范围限制到区间 [min,max]
        free_bits_kl = torch.clamp(kl_per_dim, min=config.FREE_BITS_EPSILON) # shape: [batch_size, latent_dim]
        
        # 将所有维度的损失相加，然后在batch上取平均
        l_kl = torch.sum(free_bits_kl, dim=1).mean()
    else:
        l_kl = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / recon_logits.size(0)

    # 4. 对抗损失
    if config.BETA_ADV > 0:
        l_adv = -torch.mean(D_fake_output)
    else:
        l_adv = torch.tensor(0.0, device=recon_logits.device)

    # 5. 尺度匹配损失
    l_scale = loss_fn_scale(pred_seq=recon_probs, true_seq=true_seq)

    # 6. VA一致性损失
    l_va = loss_fn_va_consistency(recon_probs, stim_va, group_labels)
    
    return l_distribution, l_feature, l_kl, l_adv, l_scale, l_va