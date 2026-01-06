#!/usr/bin/env python3
"""
Qwen2.5-VL 7B 多注意力头GT区域分析（修复版）
修复tokenization问题和bfloat16转换问题
"""

import os
import json
import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoProcessor, AutoModelForVision2Seq, AutoTokenizer
from qwen_vl_utils import process_vision_info
import base64
from io import BytesIO

# 配置
MODEL_PATH = "/data2/shaos/data/Qwen/Qwen2.5-VL-7B-Instruct"  # 修改为实际路径
IMAGE_BASE_DIR = "/data2/ouyangxc/data/coco2017/train2017"
GT_FILE = "/data2/shaos/data/coco/annotations/small_objects.jsonl"
OUTPUT_BASE_DIR = "/data1/shaos/labs/GLIMPSE/head_select/output"

device = "cuda" if torch.cuda.is_available() else "cpu"
# 使用float32以避免bfloat16转换问题
compute_dtype = torch.float16  # 改为float16避免bfloat16转换问题

def encode_base64(image):
    """将PIL图像编码为base64字符串"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

def prepare_qwen2_5_input(messages, processor):
    """准备Qwen2.5-VL输入"""
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs = process_vision_info(messages)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt")
    return inputs

def load_samples(gt_file, max_samples=None):
    """加载GT样本"""
    samples = []
    with open(gt_file, 'r') as f:
        for line in f:
            if line.strip():
                samples.append(json.loads(line.strip()))
    
    if max_samples:
        samples = samples[:max_samples]
    
    print(f"加载了 {len(samples)} 个样本")
    return samples

def get_resized_input_image(image, size=(336, 336)):
    """获取resize后的图像"""
    return image.resize(size, resample=Image.BILINEAR)

def find_object_tokens_in_input(text, object_text, tokenizer):
    """
    在tokenized输入中查找object token的位置
    返回token位置和token IDs
    """
    # 首先获取完整的tokenized输入
    tokens = tokenizer(text, add_special_tokens=False).input_ids
    
    # 单独tokenize object text
    object_tokens = tokenizer(object_text, add_special_tokens=False).input_ids
    
    print(f"完整文本: {text[:100]}...")
    print(f"对象文本: {object_text}")
    print(f"对象token IDs: {object_tokens}")
    print(f"完整token IDs长度: {len(tokens)}")
    
    # 查找object token在完整token序列中的位置
    obj_positions = []
    
    # 简单的子序列搜索
    if len(object_tokens) > 0:
        for i in range(len(tokens) - len(object_tokens) + 1):
            if tokens[i:i+len(object_tokens)] == object_tokens:
                obj_positions.extend(range(i, i+len(object_tokens)))
                print(f"找到对象token在位置: {i}")
    
    return obj_positions, object_tokens

def compute_head_wise_attention_direct(attentions, general_attentions,
                                      vision_start_pos, vision_end_pos,
                                      obj_positions=None):
    """
    直接计算每个头的注意力
    从最后一个token（生成token）到图像token计算相对注意力
    """
    num_layers = len(attentions)
    if num_layers == 0:
        print("警告: 没有获取到注意力权重")
        return None, None
    
    batch_size, num_heads, seq_len, _ = attentions[0].shape
    
    print(f"注意力形状: {attentions[0].shape}")
    print(f"层数: {num_layers}, 头数: {num_heads}")
    print(f"图像token位置: {vision_start_pos} 到 {vision_end_pos}")
    
    # 图像序列长度
    img_seq_len = vision_end_pos - vision_start_pos
    print(f"图像序列长度: {img_seq_len}")
    
    all_head_attentions = []
    
    for layer_idx in range(num_layers):
        layer_att = attentions[layer_idx][0]  # [num_heads, seq_len, seq_len]
        general_layer_att = general_attentions[layer_idx][0]
        
        # 方法1: 从最后一个token（生成token）到图像token
        last_token_idx = -1
        
        # 提取注意力
        att_to_img = layer_att[:, last_token_idx, vision_start_pos:vision_end_pos]  # [num_heads, img_seq_len]
        general_att_to_img = general_layer_att[:, last_token_idx, vision_start_pos:vision_end_pos]
        
        # 转换为float32避免bfloat16问题
        att_to_img = att_to_img.to(torch.float32)
        general_att_to_img = general_att_to_img.to(torch.float32)
        
        # 计算相对注意力 (att / general_att)
        # 添加小值避免除零
        relative_att = att_to_img / (general_att_to_img + 1e-8)
        
        # 转换为numpy
        all_head_attentions.append(relative_att.detach().cpu().numpy())
    
    if not all_head_attentions:
        print("警告: 无法计算头部注意力")
        return None, None
    
    # 堆叠所有层 [num_layers, num_heads, img_seq_len]
    head_attentions_np = np.stack(all_head_attentions, axis=0)
    
    # 计算平均注意力
    avg_attention_np = head_attentions_np.mean(axis=(0, 1))
    
    print(f"成功计算 {num_layers} 层 × {num_heads} 头的注意力")
    print(f"注意力矩阵形状: {head_attentions_np.shape}")
    
    return head_attentions_np, avg_attention_np

def analyze_heads_in_gt_region(head_attentions, gt_bbox, H_patch, W_patch,
                              orig_image_size, resized_image, object_text, output_dir):
    """
    分析所有层的所有头在GT区域中的贡献
    """
    print(f"\n" + "="*60)
    print(f"开始分析GT区域头部贡献")
    print(f"GT区域: {gt_bbox}")
    print(f"原图尺寸: {orig_image_size}")
    print(f"Patch网格: {H_patch}×{W_patch}")
    print("="*60)
    
    num_layers, num_heads, img_seq_len = head_attentions.shape
    
    # 检查注意力形状
    print(f"注意力矩阵形状: {head_attentions.shape}")
    
    # 检查img_seq_len是否等于H_patch * W_patch
    expected_patches = H_patch * W_patch
    if img_seq_len != expected_patches:
        print(f"注意: 注意力长度({img_seq_len})与期望的patch数({expected_patches})不匹配")
        print(f"实际网格: {H_patch}×{W_patch} = {expected_patches}")
        print(f"将调整网格为正方形...")
        
        # 尝试找到最接近的正方形
        side = int(math.sqrt(img_seq_len))
        if side * side == img_seq_len:
            H_patch, W_patch = side, side
            print(f"调整为正方形网格: {H_patch}×{W_patch}")
        else:
            # 尝试找到合适的因子
            for h in range(int(math.sqrt(img_seq_len)), 0, -1):
                if img_seq_len % h == 0:
                    H_patch, W_patch = h, img_seq_len // h
                    print(f"调整为网格: {H_patch}×{W_patch}")
                    break
    
    # 解析GT bbox格式 [x, y, width, height]
    gt_x, gt_y, gt_width, gt_height = gt_bbox
    orig_w, orig_h = orig_image_size
    
    # 默认resize大小
    resize_w, resize_h = 336, 336
    
    # 计算从原图到resize图像的缩放
    scale_to_resize_x = resize_w / orig_w
    scale_to_resize_y = resize_h / orig_h
    
    # 在resize图像上的GT坐标
    gt_x_resized = gt_x * scale_to_resize_x
    gt_y_resized = gt_y * scale_to_resize_y
    gt_x2_resized = gt_x_resized + gt_width * scale_to_resize_x
    gt_y2_resized = gt_y_resized + gt_height * scale_to_resize_y
    
    print(f"在resize图像({resize_w}x{resize_h})上的GT坐标:")
    print(f"  x1: {gt_x_resized:.2f}, y1: {gt_y_resized:.2f}")
    print(f"  x2: {gt_x2_resized:.2f}, y2: {gt_y2_resized:.2f}")
    
    # 计算每个patch在resize图像上的像素范围
    patch_width = resize_w / W_patch
    patch_height = resize_h / H_patch
    
    print(f"\n每个patch的像素尺寸: {patch_width:.2f}×{patch_height:.2f}")
    
    # 找到所有"蹭到"GT区域的patch
    patch_x1 = int(np.floor(gt_x_resized / patch_width))
    patch_y1 = int(np.floor(gt_y_resized / patch_height))
    patch_x2 = int(np.ceil(gt_x2_resized / patch_width))
    patch_y2 = int(np.ceil(gt_y2_resized / patch_height))
    
    # 确保在patch网格范围内
    patch_x1 = max(0, patch_x1)
    patch_y1 = max(0, patch_y1)
    patch_x2 = min(W_patch-1, patch_x2)
    patch_y2 = min(H_patch-1, patch_y2)
    
    print(f"\n在patch网格上受影响的GT区域:")
    print(f"  patch坐标: ({patch_x1}, {patch_y1}) -> ({patch_x2}, {patch_y2})")
    print(f"  覆盖patch数: {(patch_x2-patch_x1+1)}×{(patch_y2-patch_y1+1)} = {(patch_x2-patch_x1+1)*(patch_y2-patch_y1+1)}")
    
    # 计算每个头在GT区域的贡献
    gt_head_contributions = []
    
    for layer_idx in range(num_layers):
        for head_idx in range(num_heads):
            # 获取该头的注意力图
            try:
                head_att = head_attentions[layer_idx, head_idx].reshape(H_patch, W_patch)
            except ValueError as e:
                print(f"警告: 无法重塑头 {layer_idx}-{head_idx} 的注意力图: {e}")
                continue
            
            # 提取GT区域内的所有patch
            gt_attention_values = []
            
            for patch_y in range(patch_y1, patch_y2+1):
                for patch_x in range(patch_x1, patch_x2+1):
                    if patch_y < H_patch and patch_x < W_patch:
                        gt_attention_values.append(head_att[patch_y, patch_x])
            
            if gt_attention_values:
                # 计算统计信息
                gt_attention_array = np.array(gt_attention_values)
                mean_attention = gt_attention_array.mean()
                max_attention = gt_attention_array.max()
                total_attention = gt_attention_array.sum()
                
                # 计算全局统计用于对比
                global_mean = head_att.mean()
                global_max = head_att.max()
                
                # 计算相对贡献
                relative_mean = mean_attention / (global_mean + 1e-8)
                relative_max = max_attention / (global_max + 1e-8)
                
                # 计算GT区域占全图注意力的比例
                attention_ratio = total_attention / (head_att.sum() + 1e-8)
                
                # 计算覆盖的patch数量
                covered_patches = len(gt_attention_values)
                
                gt_head_contributions.append({
                    'layer': layer_idx,
                    'head': head_idx,
                    'mean_attention': float(mean_attention),
                    'max_attention': float(max_attention),
                    'total_attention': float(total_attention),
                    'global_mean': float(global_mean),
                    'global_max': float(global_max),
                    'relative_mean': float(relative_mean),
                    'relative_max': float(relative_max),
                    'attention_ratio': float(attention_ratio),
                    'covered_patches': covered_patches,
                    'patch_bbox': (patch_x1, patch_y1, patch_x2, patch_y2)
                })
    
    print(f"计算了 {len(gt_head_contributions)} 个头在GT区域的贡献")
    
    if not gt_head_contributions:
        print("警告: 没有计算到任何头在GT区域的贡献")
        return None
    
    # 保存分析结果
    save_gt_head_analysis(gt_head_contributions, output_dir, num_layers, num_heads,
                         patch_x1, patch_y1, patch_x2, patch_y2, 
                         gt_bbox, object_text, H_patch, W_patch)
    
    # 可视化分析结果
    visualize_gt_head_analysis(head_attentions, gt_head_contributions, num_layers, num_heads,
                             H_patch, W_patch, patch_x1, patch_y1, patch_x2, patch_y2,
                             resized_image, output_dir, object_text)
    
    return gt_head_contributions

def save_gt_head_analysis(gt_head_contributions, output_dir, num_layers, num_heads,
                         patch_x1, patch_y1, patch_x2, patch_y2, gt_bbox, object_text,
                         H_patch, W_patch):
    """
    保存GT区域的头部贡献分析
    """
    os.makedirs(output_dir, exist_ok=True)
    
    with open(os.path.join(output_dir, "gt_region_head_analysis.txt"), 'w') as f:
        f.write("Qwen2.5-VL GT区域头部注意力分析\n")
        f.write("="*100 + "\n\n")
        
        f.write(f"GT Bounding Box (pixel): {gt_bbox}\n")
        f.write(f"GT Patch Region: ({patch_x1}, {patch_y1}) -> ({patch_x2}, {patch_y2})\n")
        f.write(f"Object: {object_text}\n")
        f.write(f"Total Layers: {num_layers}, Total Heads: {num_heads}\n")
        f.write(f"Patch Grid: {H_patch}×{W_patch}\n")
        f.write("\n" + "="*100 + "\n\n")
        
        # 按注意力占比排序
        f.write("Top 1000 Heads by Attention Ratio in GT Region:\n")
        f.write("Rank\tLayer\tHead\tAttRatio\tMeanAtt\tTotalAtt\tRelativeMean\tPatches\n")
        f.write("-"*100 + "\n")
        
        sorted_by_ratio = sorted(gt_head_contributions, 
                               key=lambda x: x['attention_ratio'], reverse=True)
        
        for i, head_info in enumerate(sorted_by_ratio[:1000]):
            f.write(f"{i+1}\t{head_info['layer']}\t{head_info['head']}\t"
                   f"{head_info['attention_ratio']:.6f}\t"
                   f"{head_info['mean_attention']:.6f}\t"
                   f"{head_info['total_attention']:.6f}\t"
                   f"{head_info['relative_mean']:.2f}\t"
                   f"{head_info['covered_patches']}\n")
        
        f.write("\n" + "="*100 + "\n\n")
        
        # 保存总体统计
        if gt_head_contributions:
            all_ratios = [h['attention_ratio'] for h in gt_head_contributions]
            all_means = [h['mean_attention'] for h in gt_head_contributions]
            
            f.write("Overall Statistics:\n")
            f.write(f"  Average Attention Ratio: {np.mean(all_ratios):.6f}\n")
            f.write(f"  Max Attention Ratio: {np.max(all_ratios):.6f}\n")
            f.write(f"  Min Attention Ratio: {np.min(all_ratios):.6f}\n")
            f.write(f"  Average Mean Attention: {np.mean(all_means):.6f}\n")
            
            # 找出最好的头
            best_by_ratio = sorted_by_ratio[0] if sorted_by_ratio else None
            if best_by_ratio:
                f.write(f"\nBest Head by Attention Ratio:\n")
                f.write(f"  Layer {best_by_ratio['layer']}, Head {best_by_ratio['head']}\n")
                f.write(f"  Ratio: {best_by_ratio['attention_ratio']:.6f}\n")
                f.write(f"  Mean Attention: {best_by_ratio['mean_attention']:.6f}\n")
    
    # 保存CSV格式
    with open(os.path.join(output_dir, "gt_region_head_analysis.csv"), 'w') as csv_f:
        csv_f.write("layer,head,mean_attention,max_attention,total_attention,"
                   "global_mean,global_max,relative_mean,relative_max,"
                   "attention_ratio,covered_patches\n")
        
        for head_info in sorted_by_ratio:
            csv_f.write(f"{head_info['layer']},{head_info['head']},"
                       f"{head_info['mean_attention']:.6f},{head_info['max_attention']:.6f},"
                       f"{head_info['total_attention']:.6f},{head_info['global_mean']:.6f},"
                       f"{head_info['global_max']:.6f},{head_info['relative_mean']:.2f},"
                       f"{head_info['relative_max']:.2f},{head_info['attention_ratio']:.6f},"
                       f"{head_info['covered_patches']}\n")
    
    print(f"已保存分析结果到: {output_dir}")

def visualize_gt_head_analysis(head_attentions, gt_head_contributions, num_layers, num_heads,
                             H_patch, W_patch, patch_x1, patch_y1, patch_x2, patch_y2,
                             resized_image, output_dir, object_text):
    """
    可视化GT区域的头部贡献分析
    """
    # 按注意力占比排序
    sorted_by_ratio = sorted(gt_head_contributions, 
                           key=lambda x: x['attention_ratio'], reverse=True)
    
    # 创建汇总可视化
    fig = plt.figure(figsize=(20, 15))
    
    # 1. 原始图像和GT区域
    ax1 = plt.subplot(3, 4, 1)
    if resized_image is not None:
        ax1.imshow(resized_image)
        
        # 绘制GT区域
        rect = plt.Rectangle((patch_x1 * (336/W_patch), 
                            patch_y1 * (336/H_patch)), 
                           (patch_x2-patch_x1) * (336/W_patch), 
                           (patch_y2-patch_y1) * (336/H_patch),
                           linewidth=2, edgecolor='red', facecolor='none', alpha=0.7)
        ax1.add_patch(rect)
        
        ax1.set_title(f"Image with GT Region\n{object_text}")
        ax1.axis('off')
    
    # 2. 最佳头的注意力图
    ax2 = plt.subplot(3, 4, 2)
    if sorted_by_ratio:
        best_ratio_head = sorted_by_ratio[0]
        try:
            best_ratio_att = head_attentions[best_ratio_head['layer'], 
                                            best_ratio_head['head']].reshape(H_patch, W_patch)
            
            im2 = ax2.imshow(best_ratio_att, cmap='jet')
            
            # 绘制GT区域边界
            rect = plt.Rectangle((patch_x1-0.5, patch_y1-0.5), 
                               patch_x2-patch_x1+1, patch_y2-patch_y1+1,
                               linewidth=2, edgecolor='white', facecolor='none')
            ax2.add_patch(rect)
            
            ax2.set_title(f"Best Head (Ratio: {best_ratio_head['attention_ratio']:.4f})\n"
                         f"Layer {best_ratio_head['layer']}, Head {best_ratio_head['head']}")
            ax2.axis('off')
            plt.colorbar(im2, ax=ax2)
        except Exception as e:
            ax2.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
            ax2.axis('off')
    
    # 3. 注意力占比分布
    ax3 = plt.subplot(3, 4, 3)
    ratios = [h['attention_ratio'] for h in gt_head_contributions]
    ax3.hist(ratios, bins=50, alpha=0.7, color='green')
    ax3.set_xlabel('Attention Ratio')
    ax3.set_ylabel('Number of Heads')
    ax3.set_title(f'Attention Ratio Distribution\nMean: {np.mean(ratios):.4f}')
    ax3.grid(True, alpha=0.3)
    
    # 4. 层-头热力图
    ax4 = plt.subplot(3, 4, 4)
    layer_head_matrix = np.zeros((num_layers, num_heads))
    for head_info in gt_head_contributions:
        layer_head_matrix[head_info['layer'], head_info['head']] = head_info['attention_ratio']
    
    im4 = ax4.imshow(layer_head_matrix, cmap='viridis', aspect='auto')
    ax4.set_xlabel('Head Index')
    ax4.set_ylabel('Layer Index')
    ax4.set_title('Attention Ratio Heatmap')
    plt.colorbar(im4, ax=ax4, label='Attention Ratio')
    
    # 5. 前4个高占比头的注意力图
    if sorted_by_ratio:
        for i in range(min(4, len(sorted_by_ratio))):
            ax = plt.subplot(3, 4, 5+i)
            head_info = sorted_by_ratio[i]
            try:
                head_att = head_attentions[head_info['layer'], head_info['head']].reshape(H_patch, W_patch)
                
                im = ax.imshow(head_att, cmap='jet')
                
                # 绘制GT区域边界
                rect = plt.Rectangle((patch_x1-0.5, patch_y1-0.5), 
                                   patch_x2-patch_x1+1, patch_y2-patch_y1+1,
                                   linewidth=1, edgecolor='white', facecolor='none')
                ax.add_patch(rect)
                
                ax.set_title(f"#{i+1}: L{head_info['layer']}H{head_info['head']}\n"
                            f"Ratio: {head_info['attention_ratio']:.4f}")
                ax.axis('off')
            except Exception as e:
                ax.text(0.5, 0.5, f"Error: {e}", ha='center', va='center')
                ax.axis('off')
    
    plt.tight_layout()
    summary_path = os.path.join(output_dir, "gt_region_head_analysis_summary.png")
    plt.savefig(summary_path, bbox_inches='tight', dpi=150)
    plt.close()
    
    print(f"已保存可视化结果: {summary_path}")

def process_sample(sample, model, processor, output_base_dir):
    """
    处理单个样本：使用output_attentions方法计算各个头在GT区域的情况
    """
    sample_id = sample['image_id']
    sample_output_dir = os.path.join(output_base_dir, f"sample_{sample_id}")
    os.makedirs(sample_output_dir, exist_ok=True)
    
    print(f"\n{'='*60}")
    print(f"处理样本 {sample_id}: {sample['file_name']}")
    print(f"类别: {sample['category_name']}")
    print(f"GT BBox: {sample['bbox']}")
    print(f"{'='*60}")
    
    # 获取图像路径和GT信息
    image_path = os.path.join(IMAGE_BASE_DIR, sample['file_name'])
    gt_bbox = sample['bbox']
    object_text = sample['category_name']
    
    try:
        # 1. 加载图像
        orig_image = Image.open(image_path).convert("RGB")
        orig_image_size = orig_image.size
        
        # 2. 构建特定prompt和通用prompt
        specific_prompt = f"Identify the {object_text} in the image."
        general_prompt = "Describe the image."
        
        # 将图像编码为base64
        image_str = encode_base64(orig_image)
        
        # 特定prompt的消息
        specific_messages = [{"role": "user", 
                             "content": [
                                 {"type": "image", "image": f'data:image;base64,{image_str}'},
                                 {"type": "text", "text": specific_prompt}
                             ]}]
        
        # 通用prompt的消息
        general_messages = [{"role": "user", 
                            "content": [
                                {"type": "image", "image": f'data:image;base64,{image_str}'},
                                {"type": "text", "text": general_prompt}
                            ]}]
        
        # 3. 准备输入
        specific_inputs = prepare_qwen2_5_input(specific_messages, processor)
        general_inputs = prepare_qwen2_5_input(general_messages, processor)
        
        # 移动到设备
        specific_inputs = {k: v.to(device) for k, v in specific_inputs.items()}
        general_inputs = {k: v.to(device) for k, v in general_inputs.items()}
        
        # 4. 获取tokenizer
        tokenizer = processor.tokenizer
        
        # 5. 获取图像token位置
        vision_start_token_id = tokenizer.convert_tokens_to_ids('<|vision_start|>')
        vision_end_token_id = tokenizer.convert_tokens_to_ids('<|vision_end|>')
        
        input_ids = specific_inputs['input_ids'][0].tolist()
        
        # 查找vision token位置
        try:
            pos = input_ids.index(vision_start_token_id) + 1
            pos_end = input_ids.index(vision_end_token_id)
        except ValueError as e:
            print(f"警告: 无法找到vision token位置: {e}")
            # 使用默认位置
            pos = 15  # Qwen2.5-VL通常在这个位置
            pos_end = pos + 234  # 默认图像序列长度
        
        img_seq_len = pos_end - pos
        print(f"图像token位置: {pos} 到 {pos_end} (长度: {img_seq_len})")
        
        # 6. 获取图像网格信息
        image_grid_thw = specific_inputs['image_grid_thw']
        if image_grid_thw is not None:
            att_shape = (image_grid_thw[0, 1:] / 2).cpu().numpy().astype(int).tolist()
            H_patch, W_patch = att_shape[0], att_shape[1]
        else:
            # 如果没有grid信息，使用默认值
            H_patch, W_patch = 18, 13  # Qwen2.5-VL的默认网格
        
        print(f"Patch网格: {H_patch}×{W_patch}")
        
        # 7. 前向传播，获取注意力权重
        torch.cuda.empty_cache()
        
        print("运行模型前向传播...")
        with torch.no_grad():
            if device == "cuda":
                with torch.autocast(device_type="cuda", dtype=compute_dtype):
                    specific_outputs = model(**specific_inputs, output_attentions=True, use_cache=False)
                    general_outputs = model(**general_inputs, output_attentions=True, use_cache=False)
            else:
                specific_outputs = model(**specific_inputs, output_attentions=True, use_cache=False)
                general_outputs = model(**general_inputs, output_attentions=True, use_cache=False)
        
        # 8. 计算每个头的注意力
        print("计算每个头的注意力分布...")
        head_attentions, avg_attention_np = compute_head_wise_attention_direct(
            specific_outputs.attentions, general_outputs.attentions,
            pos, pos_end
        )
        
        if head_attentions is None:
            print("警告: 无法计算头部注意力")
            return False
        
        # 9. 分析GT区域中的头部贡献
        print("分析GT区域中的头部贡献...")
        resized_image = get_resized_input_image(orig_image, (336, 336))
        gt_head_contributions = analyze_heads_in_gt_region(
            head_attentions, gt_bbox, H_patch, W_patch,
            orig_image_size, resized_image, object_text, sample_output_dir
        )
        
        if gt_head_contributions is None:
            print("警告: 无法分析GT区域贡献")
            return False
        
        # 10. 保存数据
        data_dir = os.path.join(sample_output_dir, "data")
        os.makedirs(data_dir, exist_ok=True)
        
        np.save(os.path.join(data_dir, "head_attentions.npy"), head_attentions)
        np.save(os.path.join(data_dir, "avg_attention.npy"), avg_attention_np)
        
        metadata = {
            'sample_id': sample_id,
            'file_name': sample['file_name'],
            'category_name': sample['category_name'],
            'gt_bbox': gt_bbox,
            'area': sample['area'],
            'orig_image_size': orig_image_size,
            'H_patch': H_patch,
            'W_patch': W_patch,
            'num_layers': head_attentions.shape[0],
            'num_heads': head_attentions.shape[1],
            'object_text': object_text,
            'specific_prompt': specific_prompt,
            'general_prompt': general_prompt,
            'vision_start_pos': pos,
            'vision_end_pos': pos_end
        }
        
        with open(os.path.join(data_dir, "metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"样本 {sample_id} 处理完成!")
        
        # 清理内存
        del specific_inputs, general_inputs, specific_outputs, general_outputs, head_attentions
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        print(f"处理样本 {sample_id} 时出错: {e}")
        import traceback
        traceback.print_exc()
        
        error_file = os.path.join(sample_output_dir, "error.txt")
        with open(error_file, 'w') as f:
            f.write(f"处理样本 {sample_id} 时出错:\n{str(e)}\n")
            traceback.print_exc(file=f)
        
        return False

def main():
    """主函数"""
    print("="*60)
    print("Qwen2.5-VL 7B GT区域多注意力头分析（修复版）")
    print("="*60)
    
    os.makedirs(OUTPUT_BASE_DIR, exist_ok=True)
    
    # 加载样本
    # samples = load_samples(GT_FILE, max_samples=3)  # 先测试3个样本
    samples = load_samples(GT_FILE, max_samples=100000)  # 先测试3个样本
    
    # 初始化模型和处理器
    print("初始化Qwen2.5-VL模型...")
    try:
        processor = AutoProcessor.from_pretrained(
            MODEL_PATH,
            use_fast=True  # 使用fast processor
        )
    except:
        processor = AutoProcessor.from_pretrained(MODEL_PATH)
    
    # 使用eager attention implementation以避免warning
    model = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        torch_dtype=compute_dtype,
        attn_implementation="eager"  # 使用eager避免warning
    ).to(device).eval()
    
    print(f"模型加载完成，使用设备: {device}")
    print(f"模型dtype: {model.dtype}")
    
    # 处理样本
    successful = 0
    failed = 0
    
    for i, sample in enumerate(samples):
        print(f"\n处理样本 {i+1}/{len(samples)}")
        success = process_sample(sample, model, processor, OUTPUT_BASE_DIR)
        
        if success:
            successful += 1
        else:
            failed += 1
        
        # 定期清理GPU缓存
        if device.startswith("cuda"):
            print("清理GPU缓存...")
            torch.cuda.empty_cache()
    
    # 创建汇总报告
    summary_file = os.path.join(OUTPUT_BASE_DIR, "summary.txt")
    with open(summary_file, 'w') as f:
        f.write("Qwen2.5-VL GT区域头部注意力分析汇总\n")
        f.write("="*60 + "\n\n")
        f.write(f"总样本数: {len(samples)}\n")
        f.write(f"成功处理: {successful}\n")
        f.write(f"失败: {failed}\n\n")
        
        f.write("样本列表:\n")
        for i, sample in enumerate(samples):
            f.write(f"{i+1}. {sample['file_name']} - {sample['category_name']}\n")
    
    print(f"\n处理完成! 成功: {successful}, 失败: {failed}")
    print(f"结果保存在: {OUTPUT_BASE_DIR}")

if __name__ == "__main__":
    main()