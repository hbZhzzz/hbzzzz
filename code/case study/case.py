# case_study_plot.py
import os 
import sys
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), "../../")), 'code'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import torch

import config

# --- 1. 准备和处理数据 ---

def prepare_plot_data(generated_data_list, stim_key):
    """
    从原始数据列表中为指定的刺激准备绘图所需的数据。
    """
    all_data = []
    for item in generated_data_list:
        if stim_key in item:
            group = item['group_label']
            prob_seq = item[stim_key]
            for t, frame_probs in enumerate(prob_seq):
                for emotion_idx, prob in enumerate(frame_probs):
                    all_data.append({
                        'group': group,
                        'time': t,
                        'emotion_idx': emotion_idx,
                        'probability': prob
                    })
    
    df = pd.DataFrame(all_data)
    return df

# --- 2. 核心绘图函数 ---

def plot_case_study_figure(data_pvs, data_nvs, class_names):
    """
    绘制2x2展示图(只绘制Happy, Sad, Neutral三条关键曲线)。
    """

    KEY_EMOTIONS = ['Happy', 'Sad', 'Neutral']
    
    color_map = {
        'Happy': '#2ca02c',
        'Sad': '#d62728',   
        'Neutral': '#7f7f7f'  
    }
    #全局字体和字号设置
    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Times New Roman', 
        'axes.labelsize': 12,  
        'axes.titlesize': 14,  
        'xtick.labelsize': 10,  
        'ytick.labelsize': 10,  
        'legend.fontsize': 9,   
        'legend.title_fontsize': 10
    })
    
    # 创建一个只包含关键情绪的色板
    key_emotion_palette = [color_map[name] for name in KEY_EMOTIONS]
    key_emotion_indices = [class_names.index(name) for name in KEY_EMOTIONS]

    plt.style.use('seaborn-v0_8-whitegrid')
    plt.rcParams.update({
        'font.family': 'Times New Roman',
        'axes.labelsize': 16, 
        'axes.titlesize': 18, 
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
        'legend.fontsize': 13,
        'legend.title_fontsize': 14
    })

    #创建2x2的子图布局 
    fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharex=True, sharey=True)
    fig.suptitle('Case Study: Generated Emotional Dynamics under Different Stimuli', fontsize=24, y=1.0)

    plot_configs = [
        {'data': data_pvs, 'title': 'Positive Stimulus'},
        {'data': data_nvs, 'title': 'Negative Stimulus'}
    ]
    groups = ['Control', 'Depressed']


    for col, config in enumerate(plot_configs):
        for row, group_name in enumerate(groups):
            ax = axes[row, col]
            

            df_full = config['data']
            df_subset = df_full[
                (df_full['group'] == group_name) & 
                (df_full['emotion_idx'].isin(key_emotion_indices))
            ].copy() 
            
          
            df_subset['emotion'] = df_subset['emotion_idx'].map({idx: name for idx, name in enumerate(class_names)})

            # 使用seaborn的lineplot绘制均值和置信区间
            sns.lineplot(
                data=df_subset,
                x='time',
                y='probability',
                hue='emotion', 
                hue_order=KEY_EMOTIONS, 
                palette=color_map, 
                ax=ax,
                linewidth=3.0, 
                errorbar=('ci', 95)
            )
            

            ax.axvline(x=60, color='black', linestyle='--', linewidth=2.0, alpha=0.9)
            

            ax.get_legend().remove()
            
            # 设置子图标题和坐标轴标签
            if row == 0:
                ax.set_title(config['title'], fontweight='bold')
            if col == 0:
                ax.set_ylabel(f'{group_name} Group\nProbability', fontweight='bold')
            else:
                ax.set_ylabel(f'{group_name} Group', fontweight='bold')

            ax.set_xlabel("Time (Frames)" if row == 1 else "")
            
            #调整Y轴范围 
            ax.set_ylim(-0.02, 0.6) # 
            ax.set_xlim(-5, 125)
            # ax.grid(False)


    # 创建共享图例
    from matplotlib.lines import Line2D
    
    legend_elements = [
        Line2D([0], [0], color=color_map['Happy'], lw=3, label='Happy'),
        Line2D([0], [0], color=color_map['Sad'], lw=3, label='Sad'),
        Line2D([0], [0], color=color_map['Neutral'], lw=3, label='Neutral'),
        Line2D([0], [0], color='black', linestyle='--', lw=1.5, label='Stimulus Onset')
    ]
    # fig.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, 0.95), ncol=4, frameon=False) 
    fig.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(0.99, 0.95), ncol=4, frameon=True, shadow=True)

    plt.tight_layout(rect=[0, 0.02, 1, 0.95]) # 调整布局
    plt.show()



if __name__ == '__main__':

    CLASS_NAMES = config.CLASS_NAMES
    generated_data = torch.load(os.path.join(config.CODE_DIR,'analysis/generated_sample_lambda-feature_1.0_alpha-kl-max_0.0001_beta-adv_0.0_delta-scale_0.5_eta-va_0.1145_freebits_2.0_num_100.pt'))
   
        

    df_pvs = prepare_plot_data(generated_data, 'stim_02_NVS_1')
    df_nvs = prepare_plot_data(generated_data, 'stim_03_HAS-NVS_1')


    plot_case_study_figure(df_pvs, df_nvs, CLASS_NAMES)
