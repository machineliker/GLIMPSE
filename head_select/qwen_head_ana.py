import os
import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict
import json
import numpy as np
from scipy import stats

# ================= 配置区域 =================
# 数据根目录
ROOT_DIR = "/data2/shaos/labs/mllms_know_94.1_head_select/head_analysisi/"
# 要分析的目标文件名
TARGET_FILENAME = "gt_region_head_analysis.txt"
# 图表保存目录
OUTPUT_DIR = "/data2/shaos/labs/mllms_know_94.1_head_select/head_analysisi/head_analysis_results"
# 注意力贡献前 n 的数量列表 - 保存多个不同n值的结果
TOP_N_VALUES = list(range(1, 101))
# 指定要分析的层范围 (从第 Lx 层到第 Ly 层，Lx <= Ly)
LAYER_RANGE = (0, 27)  # 例如从 L1 到 L5
# ===========================================

def parse_file_with_attention_ratio(file_path: str) -> List[Dict]:
    """
    解析文件，提取注意力占比相关的数据
    优先按注意力占比(attention_ratio)排序
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"读取文件失败: {file_path}, 错误: {e}")
        return []

    heads_data = []
    
    # 方法1: 尝试从CSV文件读取（更可靠）
    csv_file = file_path.replace(".txt", ".csv")
    if os.path.exists(csv_file):
        try:
            csv_df = pd.read_csv(csv_file)
            
            # 检查是否有需要的列
            required_columns = ['layer', 'head', 'attention_ratio']
            if all(col in csv_df.columns for col in required_columns):
                for _, row in csv_df.iterrows():
                    record = {
                        'Layer': int(row['layer']),
                        'Head': int(row['head']),
                        'AttentionRatio': float(row['attention_ratio'])
                    }
                    
                    # 添加其他可选字段
                    if 'mean_attention' in csv_df.columns:
                        record['MeanAttention'] = float(row['mean_attention'])
                    if 'relative_mean' in csv_df.columns:
                        record['RelativeContribution'] = float(row['relative_mean'])
                    if 'contrast_ratio' in csv_df.columns:
                        record['Contrast'] = float(row['contrast_ratio'])
                    if 'covered_patches' in csv_df.columns:
                        record['CoveredPatches'] = int(row['covered_patches'])
                    if 'total_attention' in csv_df.columns:
                        record['TotalAttention'] = float(row['total_attention'])
                    
                    heads_data.append(record)
                
                print(f"✅ 从CSV文件加载数据: {csv_file} ({len(heads_data)} 条记录)")
                return heads_data
        except Exception as e:
            print(f"读取CSV文件失败: {csv_file}, 错误: {e}")

    # 方法2: 如果CSV文件不存在或读取失败，尝试解析TXT文件
    print(f"尝试解析TXT文件: {file_path}")
    
    # 尝试多种表格格式
    patterns_to_try = [
        # 格式1: Top 50 Heads by Attention Ratio
        (r"Top \d+ Heads by Attention Ratio[\s\S]*?Rank\s+Layer\s+Head\s+AttRatio\s+MeanAtt\s+TotalAtt\s+GlobalTotal\s+Contrast\s+Patches[\s\S]*?(\d+\s+\d+\s+\d+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+\d+)"),
        
        # 格式2: Top 30 Heads by Mean Attention (包含AttRatio)
        (r"Top \d+ Heads by Mean Attention[\s\S]*?Rank\s+Layer\s+Head\s+MeanAtt\s+RelMean\s+AttRatio\s+Contrast\s+Patches[\s\S]*?(\d+\s+\d+\s+\d+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+\d+)"),
        
        # 格式3: 通用表格行匹配
        (r"^\s*(\d+)\s+(\d+)\s+(\d+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+(\d+)", re.MULTILINE)
    ]
    
    for pattern_idx, pattern in enumerate(patterns_to_try):
        try:
            if pattern_idx < 2:  # 前两个模式是找特定部分
                if pattern in content:
                    section_match = re.search(pattern, content)
                    if section_match:
                        # 提取表格数据部分
                        table_section = section_match.group(1)
                        # 在表格部分中匹配行
                        row_pattern = r"^\s*\d+\s+\d+\s+\d+\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+([0-9\.]+)\s+(\d+)"
                        row_matches = re.findall(row_pattern, table_section, re.MULTILINE)
                        
                        for match in row_matches:
                            try:
                                record = {
                                    'Layer': int(match[0]),  # 需要根据实际格式调整索引
                                    'Head': int(match[1]),
                                    'AttentionRatio': float(match[2]),
                                    'MeanAttention': float(match[3]),
                                    'TotalAttention': float(match[4]),
                                    'Contrast': float(match[5]),
                                    'CoveredPatches': int(match[6])
                                }
                                heads_data.append(record)
                            except Exception as e:
                                continue
                        
                        if heads_data:
                            print(f"✅ 从TXT文件解析数据成功 (模式{pattern_idx+1}): {len(heads_data)} 条记录")
                            return heads_data
            else:  # 第三个模式是通用匹配
                matches = re.findall(pattern, content, re.MULTILINE)
                for match in matches:
                    try:
                        # 跳过表头行
                        if match[0].isdigit() and int(match[0]) > 0:
                            record = {
                                'Layer': int(match[1]),
                                'Head': int(match[2]),
                                'AttentionRatio': float(match[3]),
                                'MeanAttention': float(match[4]),
                                'TotalAttention': float(match[5]),
                                'GlobalTotal': float(match[6]),
                                'Contrast': float(match[7]),
                                'CoveredPatches': int(match[8])
                            }
                            heads_data.append(record)
                    except Exception as e:
                        continue
                
                if heads_data:
                    print(f"✅ 从TXT文件解析数据成功 (通用模式): {len(heads_data)} 条记录")
                    return heads_data
        except Exception as e:
            continue
    
    # 如果以上方法都失败，尝试简单的行匹配
    print(f"⚠️  使用简单模式解析TXT文件: {file_path}")
    simple_pattern = r"(\d+)\s+(\d+)\s+(\d+)\s+([0-9\.]+)"
    matches = re.findall(simple_pattern, content)
    
    for match in matches:
        try:
            # 确保是数据行而不是表头
            if match[0].isdigit() and int(match[0]) > 0:
                record = {
                    'Layer': int(match[1]),
                    'Head': int(match[2]),
                    'AttentionRatio': float(match[3])
                }
                heads_data.append(record)
        except Exception as e:
            continue
    
    if heads_data:
        print(f"✅ 从TXT文件解析数据成功 (简单模式): {len(heads_data)} 条记录")
    else:
        print(f"❌ 无法从文件中解析数据: {file_path}")
    
    return heads_data

def save_top_n_heads_to_jsonl(ranked_df: pd.DataFrame, title: str, n_values: List[int], layer_range: tuple):
    """
    将指定层范围内的多个不同n值的前n个头的贡献数据保存到jsonl文件中
    按注意力占比排序
    
    参数:
        ranked_df: 排序后的DataFrame
        title: 文件标题
        n_values: 要保存的n值列表
        layer_range: 层范围
    """
    # 筛选指定层范围的数据
    filtered_df = ranked_df[(ranked_df['Layer'] >= layer_range[0]) & 
                           (ranked_df['Layer'] <= layer_range[1])].copy()
    
    if filtered_df.empty:
        print(f"⚠️  在层范围 L{layer_range[0]}-L{layer_range[1]} 内没有数据")
        return []
    
    # 按注意力占比降序排序
    filtered_df = filtered_df.sort_values(by='AttRatio_mean', ascending=False).reset_index(drop=True)
    
    saved_files = []
    
    for n in n_values:
        # 限制n不超过数据长度
        actual_n = min(n, len(filtered_df))
        
        # 取前actual_n个
        top_n_df = filtered_df.head(actual_n)
        
        # 保存到JSONL文件
        jsonl_file = os.path.join(OUTPUT_DIR, f"{title}_top_{actual_n}_heads_by_ratio_L{layer_range[0]}_to_L{layer_range[1]}.jsonl")
        
        with open(jsonl_file, 'w', encoding='utf-8') as f:
            for _, row in top_n_df.iterrows():
                # 创建保存的记录
                record = {
                    'Layer': int(row['Layer']),
                    'Head': int(row['Head']),
                    'AttentionRatio': float(row['AttRatio_mean']),
                    'HeadLabel': f"L{int(row['Layer'])}-H{int(row['Head'])}",
                    'Rank': int(_ + 1)  # 添加排名信息
                }
                
                # 添加其他可用字段
                if 'MeanAttention' in row:
                    record['MeanAttention'] = float(row['MeanAttention'])
                if 'RelativeContribution' in row:
                    record['RelativeContribution'] = float(row['RelativeContribution'])
                if 'TotalAttention' in row:
                    record['TotalAttention'] = float(row['TotalAttention'])
                if 'Contrast' in row:
                    record['Contrast'] = float(row['Contrast'])
                if 'CoveredPatches' in row:
                    record['CoveredPatches'] = int(row['CoveredPatches'])
                if 'Frequency' in row:
                    record['Frequency'] = int(row['Frequency'])
                
                json.dump(record, f, ensure_ascii=False)
                f.write("\n")
        
        print(f"✅ 前 {actual_n} 个注意力头（按注意力占比排序，层范围: L{layer_range[0]} 到 L{layer_range[1]}）已保存至: {jsonl_file}")
        
        # 同时保存为CSV格式以便查看
        csv_file = jsonl_file.replace('.jsonl', '.csv')
        top_n_df.to_csv(csv_file, index=False)
        print(f"✅ CSV格式已保存至: {csv_file}")
        
        saved_files.append({
            'n': actual_n,
            'jsonl_file': jsonl_file,
            'csv_file': csv_file,
            'num_heads': len(top_n_df)
        })
    
    # 生成一个汇总文件，记录所有n值的结果
    summary_file = os.path.join(OUTPUT_DIR, f"{title}_top_n_summary_L{layer_range[0]}_to_L{layer_range[1]}.txt")
    with open(summary_file, 'w', encoding='utf-8') as f:
        f.write(f"Top-N Heads Summary for {title}\n")
        f.write(f"Layer Range: L{layer_range[0]} to L{layer_range[1]}\n")
        f.write(f"Total available heads: {len(filtered_df)}\n")
        f.write("="*60 + "\n\n")
        
        for saved in saved_files:
            f.write(f"Top {saved['n']} heads:\n")
            f.write(f"  JSONL file: {os.path.basename(saved['jsonl_file'])}\n")
            f.write(f"  CSV file: {os.path.basename(saved['csv_file'])}\n")
            f.write(f"  Number of heads saved: {saved['num_heads']}\n")
            
            # 添加统计信息
            if saved['n'] <= len(filtered_df):
                top_n_data = filtered_df.head(saved['n'])
                total_ratio = top_n_data['AttRatio_mean'].sum()
                percentage = (total_ratio / filtered_df['AttRatio_mean'].sum() * 100) if filtered_df['AttRatio_mean'].sum() > 0 else 0
                f.write(f"  Total attention ratio: {total_ratio:.6f}\n")
                f.write(f"  Percentage of total ratio: {percentage:.2f}%\n")
            
            f.write("\n")
    
    print(f"✅ Top-N汇总文件已保存至: {summary_file}")
    
    return saved_files

def generate_report_and_plots(df: pd.DataFrame, title: str):
    """
    生成排行榜文本和可视化图表
    按注意力占比排序
    """
    if df.empty:
        print(f"No data for {title}")
        return
    
    print(f"\n{'='*60}")
    print(f"分析数据集: {title}")
    print(f"总数据点数: {len(df)}")
    print(f"唯一头数: {df[['Layer', 'Head']].drop_duplicates().shape[0]}")
    
    if 'AttentionRatio' not in df.columns:
        print("❌ 数据中没有AttentionRatio列")
        return
    
    print(f"注意力占比范围: {df['AttentionRatio'].min():.6f} 到 {df['AttentionRatio'].max():.6f}")
    print(f"平均注意力占比: {df['AttentionRatio'].mean():.6f}")
    print(f"{'='*60}")

    # 1. 数据聚合 - 按(Layer, Head)分组
    # 先收集所有可能的分组列
    agg_dict = {
        'AttentionRatio': ['mean', 'max', 'min', 'std', 'count']
    }
    
    # 动态添加其他列
    additional_columns = []
    if 'MeanAttention' in df.columns:
        agg_dict['MeanAttention'] = 'mean'
        additional_columns.append('MeanAttention_mean')
    if 'RelativeContribution' in df.columns:
        agg_dict['RelativeContribution'] = 'mean'
        additional_columns.append('RelativeContribution_mean')
    if 'Contrast' in df.columns:
        agg_dict['Contrast'] = 'mean'
        additional_columns.append('Contrast_mean')
    if 'CoveredPatches' in df.columns:
        agg_dict['CoveredPatches'] = 'mean'
        additional_columns.append('CoveredPatches_mean')
    if 'TotalAttention' in df.columns:
        agg_dict['TotalAttention'] = 'mean'
        additional_columns.append('TotalAttention_mean')
    
    # 执行分组聚合
    agg_df = df.groupby(['Layer', 'Head']).agg(agg_dict).reset_index()
    
    # 扁平化列名
    agg_df.columns = ['Layer', 'Head'] + [f'AttRatio_{stat}' for stat in ['mean', 'max', 'min', 'std', 'count']] + additional_columns
    
    # 重命名count列为Frequency
    agg_df = agg_df.rename(columns={'AttRatio_count': 'Frequency'})
    
    # 按平均注意力占比降序排序
    ranked_df = agg_df.sort_values(by='AttRatio_mean', ascending=False).reset_index(drop=True)
    
    # 创建 Head 的标签列
    ranked_df['HeadLabel'] = ranked_df.apply(lambda x: f"L{int(x['Layer'])}-H{int(x['Head'])}", axis=1)

    # --- 打印排行榜 ---
    print(f"\n{'='*20} 🏆 注意力占比排行榜: {title} {'='*20}")
    print(f"按平均注意力占比(AttRatio_mean)排序")
    
    # 创建显示列
    display_columns = ['HeadLabel', 'AttRatio_mean', 'AttRatio_max', 'AttRatio_std', 'Frequency']
    
    # 动态添加其他统计列
    if 'MeanAttention_mean' in ranked_df.columns:
        display_columns.append('MeanAttention_mean')
    if 'RelativeContribution_mean' in ranked_df.columns:
        display_columns.append('RelativeContribution_mean')
    if 'Contrast_mean' in ranked_df.columns:
        display_columns.append('Contrast_mean')
    
    # 创建格式化函数
    def format_float(x):
        if isinstance(x, (int, np.integer)):
            return f"{x}"
        elif abs(x) < 0.001:
            return f"{x:.6f}"
        elif abs(x) < 1:
            return f"{x:.4f}"
        elif abs(x) < 100:
            return f"{x:.2f}"
        else:
            return f"{x:.1f}"
    
    print(ranked_df[display_columns].head(50).to_string(
        index=False, 
        formatters={col: format_float for col in display_columns if col != 'HeadLabel'}
    ))

    # --- 生成统计摘要 ---
    print(f"\n{'='*20} 📊 统计摘要: {title} {'='*20}")
    print(f"注意力占比统计:")
    print(f"  平均值: {ranked_df['AttRatio_mean'].mean():.6f}")
    print(f"  中位数: {ranked_df['AttRatio_mean'].median():.6f}")
    print(f"  最大值: {ranked_df['AttRatio_mean'].max():.6f} ({ranked_df.iloc[0]['HeadLabel']})")
    print(f"  最小值: {ranked_df['AttRatio_mean'].min():.6f}")
    print(f"  标准差: {ranked_df['AttRatio_mean'].std():.6f}")
    
    # 计算百分位数
    percentiles = [25, 50, 75, 90, 95, 99]
    print(f"\n百分位数:")
    for p in percentiles:
        value = np.percentile(ranked_df['AttRatio_mean'], p)
        print(f"  {p}%: {value:.6f}")
    
    print(f"\n出现频率统计:")
    print(f"  总头数: {len(ranked_df)}")
    print(f"  平均出现次数: {ranked_df['Frequency'].mean():.1f}")
    print(f"  最常出现头: {ranked_df.loc[ranked_df['Frequency'].idxmax()]['HeadLabel']} (出现{ranked_df['Frequency'].max()}次)")

    # --- 可视化 1: 所有 Heads 的注意力占比柱状图 ---
    num_heads = len(ranked_df)
    fig_width = max(18, min(num_heads * 0.3, 50))  # 限制最大宽度
    
    plt.figure(figsize=(fig_width, 8))
    
    # 创建颜色映射
    if num_heads > 0:
        colors = plt.cm.Reds((ranked_df['AttRatio_mean'] - ranked_df['AttRatio_mean'].min()) / 
                             (ranked_df['AttRatio_mean'].max() - ranked_df['AttRatio_mean'].min() + 1e-8))
    else:
        colors = 'skyblue'
    
    bars = plt.bar(range(num_heads), ranked_df['AttRatio_mean'], color=colors, edgecolor='black')
    
    plt.title(f'Attention Heads by Attention Ratio in GT Region ({title})', fontsize=14, fontweight='bold')
    plt.xlabel('Attention Head (Sorted by Attention Ratio)', fontsize=12)
    plt.ylabel('Attention Ratio (GT/Total)', fontsize=12)
    
    # X 轴标签
    plt.xticks(range(num_heads), ranked_df['HeadLabel'], rotation=90, fontsize=8)
    
    # 添加网格
    plt.grid(True, alpha=0.3, axis='y')
    
    # 添加平均值线
    mean_val = ranked_df['AttRatio_mean'].mean()
    plt.axhline(y=mean_val, color='green', linestyle='--', alpha=0.7, label=f'Mean: {mean_val:.4f}')
    
    # 标记前5名
    for i in range(min(5, num_heads)):
        plt.text(i, ranked_df.iloc[i]['AttRatio_mean'] + 0.001, f'#{i+1}', 
                ha='center', va='bottom', fontsize=8, fontweight='bold', color='darkred')
    
    plt.legend()
    plt.tight_layout()
    
    bar_chart_path = os.path.join(OUTPUT_DIR, f"{title}_all_heads_attention_ratio_bar.png")
    plt.savefig(bar_chart_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 柱状图已保存: {bar_chart_path}")

    # --- 可视化 2: Layer-Head 热力图 (按注意力占比) ---
    try:
        pivot_table = ranked_df.pivot(index='Head', columns='Layer', values='AttRatio_mean')
        
        plt.figure(figsize=(14, 10))
        sns.heatmap(
            pivot_table, 
            cmap='Reds', 
            annot=False, 
            cbar_kws={'label': 'Attention Ratio (GT/Total)'},
            square=True
        )
        plt.title(f'Attention Head Heatmap by Attention Ratio ({title})\n(Darker Red = Higher Ratio in GT Region)', 
                  fontsize=14, fontweight='bold')
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Head Index', fontsize=12)
        plt.tight_layout()
        
        heatmap_path = os.path.join(OUTPUT_DIR, f"{title}_attention_ratio_heatmap.png")
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 热力图已保存: {heatmap_path}")
    except Exception as e:
        print(f"⚠️  创建热力图失败: {e}")

    # --- 可视化 3: 注意力占比分布直方图 ---
    plt.figure(figsize=(10, 6))
    plt.hist(ranked_df['AttRatio_mean'], bins=50, alpha=0.7, color='skyblue', edgecolor='black')
    plt.axvline(x=mean_val, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_val:.4f}')
    plt.axvline(x=ranked_df['AttRatio_mean'].median(), color='green', linestyle='--', linewidth=2, 
                label=f'Median: {ranked_df["AttRatio_mean"].median():.4f}')
    
    plt.title(f'Distribution of Attention Ratio in GT Region ({title})', fontsize=14, fontweight='bold')
    plt.xlabel('Attention Ratio (GT/Total)', fontsize=12)
    plt.ylabel('Number of Heads', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    
    hist_path = os.path.join(OUTPUT_DIR, f"{title}_attention_ratio_distribution.png")
    plt.savefig(hist_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"✅ 分布直方图已保存: {hist_path}")

    # --- 可视化 4: 注意力占比 vs 层数的散点图 ---
    if len(ranked_df) > 1:
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(ranked_df['Layer'], ranked_df['AttRatio_mean'], 
                             c=ranked_df['Head'], cmap='viridis', s=50, alpha=0.7)
        
        # 添加回归线
        try:
            slope, intercept, r_value, p_value, std_err = stats.linregress(
                ranked_df['Layer'], ranked_df['AttRatio_mean']
            )
            x_line = np.array([ranked_df['Layer'].min(), ranked_df['Layer'].max()])
            y_line = slope * x_line + intercept
            plt.plot(x_line, y_line, 'r--', 
                    label=f'Linear fit: y={slope:.4f}x+{intercept:.4f}\nR²={r_value**2:.4f}')
            
            # 打印回归分析结果
            print(f"\n{'='*20} 📈 回归分析: 注意力占比 vs 层数 {'='*20}")
            print(f"斜率 (每层变化): {slope:.6f}")
            print(f"截距: {intercept:.6f}")
            print(f"R²值: {r_value**2:.4f}")
            print(f"p值: {p_value:.6f}")
            if p_value < 0.05:
                print("✅ 注意力占比与层数的关系具有统计显著性 (p < 0.05)")
            else:
                print("⚠️  注意力占比与层数的关系不具有统计显著性")
        except Exception as e:
            print(f"⚠️  回归分析失败: {e}")
        
        plt.title(f'Attention Ratio vs Layer Index ({title})', fontsize=14, fontweight='bold')
        plt.xlabel('Layer Index', fontsize=12)
        plt.ylabel('Attention Ratio (GT/Total)', fontsize=12)
        plt.colorbar(scatter, label='Head Index')
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        
        scatter_path = os.path.join(OUTPUT_DIR, f"{title}_layer_vs_attention_ratio.png")
        plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ 散点图已保存: {scatter_path}")
    else:
        print("⚠️  数据点不足，无法创建散点图")

    # --- 可视化 5: 注意力占比 Top 20 的详细条形图 ---
    top_n = min(50, len(ranked_df))
    if top_n > 0:
        plt.figure(figsize=(14, 8))
        top_data = ranked_df.head(top_n)
        
        colors = plt.cm.Reds((top_data['AttRatio_mean'] - top_data['AttRatio_mean'].min()) / 
                             (top_data['AttRatio_mean'].max() - top_data['AttRatio_mean'].min() + 1e-8))
        
        bars = plt.bar(range(top_n), top_data['AttRatio_mean'], color=colors, edgecolor='black')
        
        # 添加数值标签
        for i, (_, row) in enumerate(top_data.iterrows()):
            plt.text(i, row['AttRatio_mean'] + 0.001, 
                    f'{row["AttRatio_mean"]:.4f}\nL{row["Layer"]}H{row["Head"]}', 
                    ha='center', va='bottom', fontsize=8)
        
        plt.title(f'Top {top_n} Heads by Attention Ratio ({title})', fontsize=14, fontweight='bold')
        plt.xlabel('Head Rank', fontsize=12)
        plt.ylabel('Attention Ratio (GT/Total)', fontsize=12)
        plt.xticks(range(top_n), [f'#{i+1}' for i in range(top_n)])
        plt.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        
        top50_path = os.path.join(OUTPUT_DIR, f"{title}_top50_attention_ratio.png")
        plt.savefig(top50_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"✅ Top {top_n} 条形图已保存: {top50_path}")

    # 保存多个不同n值的头到jsonl文件，限制为指定层范围
    saved_files_info = save_top_n_heads_to_jsonl(ranked_df, title, TOP_N_VALUES, LAYER_RANGE)
    
    # 打印前N个头的详细信息（以最大n值显示）
    max_n = max(TOP_N_VALUES)
    top_n_df = ranked_df.head(min(max_n, len(ranked_df)))
    
    if top_n_df is not None and not top_n_df.empty:
        print(f"\n{'='*20} 🎯 前 {len(top_n_df)} 个注意力头详情 (最大n={max_n}) {'='*20}")
        for i, (_, row) in enumerate(top_n_df.iterrows()):
            if i < 20:  # 只显示前20个详细信息
                print(f"{i+1}. L{int(row['Layer'])}-H{int(row['Head'])}: 注意力占比={row['AttRatio_mean']:.6f}")
        
        print(f"... (共 {len(top_n_df)} 个头，详细信息见保存的文件)")
    
    return ranked_df

def main():
    """
    主函数：遍历目录，解析文件，分析注意力占比
    """
    # 创建输出目录
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        print(f"创建输出目录: {OUTPUT_DIR}")
    
    # 创建详细的日志文件
    log_file = os.path.join(OUTPUT_DIR, "analysis_log.txt")
    
    all_data = []
    processed_files = 0
    failed_files = 0
    
    print(f"开始扫描目录: {ROOT_DIR}")
    print(f"查找文件: {TARGET_FILENAME}")
    print(f"输出目录: {OUTPUT_DIR}")
    print(f"将保存以下n值的结果: {TOP_N_VALUES}")
    print(f"层范围: L{LAYER_RANGE[0]}-L{LAYER_RANGE[1]}")
    print(f"{'='*60}")
    
    with open(log_file, 'w', encoding='utf-8') as log_f:
        log_f.write(f"注意力占比分析日志\n")
        log_f.write(f"扫描目录: {ROOT_DIR}\n")
        log_f.write(f"目标文件: {TARGET_FILENAME}\n")
        log_f.write(f"输出目录: {OUTPUT_DIR}\n")
        log_f.write(f"将保存以下n值的结果: {TOP_N_VALUES}\n")
        log_f.write(f"层范围: L{LAYER_RANGE[0]}-L{LAYER_RANGE[1]}\n")
        log_f.write(f"{'='*60}\n\n")
    
    # 遍历目录
    for root, dirs, files in os.walk(ROOT_DIR):
        if TARGET_FILENAME in files:
            file_path = os.path.join(root, TARGET_FILENAME)
            
            print(f"\n处理文件: {file_path}")
            
            with open(log_file, 'a', encoding='utf-8') as log_f:
                log_f.write(f"\n处理文件: {file_path}\n")
            
            # 判断样本类型（根据目录名）
            dir_name = os.path.basename(root)
            sample_type = 'Unknown'
            if any(keyword in dir_name.lower() for keyword in ['incorrect', 'false', 'wrong', 'neg']):
                sample_type = 'Incorrect'
            elif any(keyword in dir_name.lower() for keyword in ['correct', 'true', 'right', 'pos']):
                sample_type = 'Correct'
            else:
                # 尝试从父目录判断
                parent_dir = os.path.basename(os.path.dirname(root))
                if any(keyword in parent_dir.lower() for keyword in ['incorrect', 'false', 'wrong', 'neg']):
                    sample_type = 'Incorrect'
                elif any(keyword in parent_dir.lower() for keyword in ['correct', 'true', 'right', 'pos']):
                    sample_type = 'Correct'
            
            print(f"  样本类型: {sample_type}")
            
            # 解析数据
            heads = parse_file_with_attention_ratio(file_path)
            
            if heads:
                for h in heads:
                    h['Type'] = sample_type
                    h['FilePath'] = file_path
                    h['SampleID'] = dir_name
                    all_data.append(h)
                
                processed_files += 1
                print(f"  ✅ 成功解析，找到 {len(heads)} 个数据点")
                
                with open(log_file, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"  ✅ 成功解析，找到 {len(heads)} 个数据点\n")
            else:
                failed_files += 1
                print(f"  ⚠️  解析失败或没有数据")
                
                with open(log_file, 'a', encoding='utf-8') as log_f:
                    log_f.write(f"  ⚠️  解析失败或没有数据\n")

    # 转换为 Pandas DataFrame
    if all_data:
        full_df = pd.DataFrame(all_data)
        
        print(f"\n{'='*60}")
        print(f"处理完成!")
        print(f"成功处理文件: {processed_files}")
        print(f"失败文件: {failed_files}")
        print(f"总数据点数: {len(full_df)}")
        print(f"将保存 {len(TOP_N_VALUES)} 个不同n值的结果文件")
        print(f"{'='*60}")
        
        with open(log_file, 'a', encoding='utf-8') as log_f:
            log_f.write(f"\n{'='*60}\n")
            log_f.write(f"处理完成!\n")
            log_f.write(f"成功处理文件: {processed_files}\n")
            log_f.write(f"失败文件: {failed_files}\n")
            log_f.write(f"总数据点数: {len(full_df)}\n")
            log_f.write(f"将保存 {len(TOP_N_VALUES)} 个不同n值的结果文件\n")
            log_f.write(f"{'='*60}\n\n")
        
        # 1. 分析全部样本
        print("\n" + "="*60)
        print("开始分析全部样本...")
        print("="*60)
        all_ranked_df = generate_report_and_plots(full_df, "All_Samples")
        
        # 2. 分析正确样本
        correct_df = full_df[full_df['Type'] == 'Correct']
        if not correct_df.empty:
            print("\n" + "="*60)
            print(f"开始分析正确样本... (共 {len(correct_df)} 个数据点)")
            print("="*60)
            correct_ranked_df = generate_report_and_plots(correct_df, "Correct_Samples")
        else:
            print("\n⚠️  没有找到正确样本数据")
        
        # 3. 分析错误样本
        incorrect_df = full_df[full_df['Type'] == 'Incorrect']
        if not incorrect_df.empty:
            print("\n" + "="*60)
            print(f"开始分析错误样本... (共 {len(incorrect_df)} 个数据点)")
            print("="*60)
            incorrect_ranked_df = generate_report_and_plots(incorrect_df, "Incorrect_Samples")
        else:
            print("\n⚠️  没有找到错误样本数据")
        
        # 4. 保存完整数据
        full_csv_path = os.path.join(OUTPUT_DIR, "all_data_complete.csv")
        full_df.to_csv(full_csv_path, index=False)
        print(f"\n✅ 完整数据已保存至: {full_csv_path}")
        
        # 5. 生成汇总报告
        summary_path = os.path.join(OUTPUT_DIR, "analysis_summary.txt")
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("注意力占比分析汇总报告\n")
            f.write("="*60 + "\n\n")
            
            f.write(f"扫描目录: {ROOT_DIR}\n")
            f.write(f"目标文件: {TARGET_FILENAME}\n")
            f.write(f"输出目录: {OUTPUT_DIR}\n")
            f.write(f"处理时间: {pd.Timestamp.now()}\n\n")
            
            f.write(f"处理统计:\n")
            f.write(f"  成功处理文件数: {processed_files}\n")
            f.write(f"  失败文件数: {failed_files}\n")
            f.write(f"  总数据点数: {len(full_df)}\n")
            f.write(f"  唯一样本数: {full_df['SampleID'].nunique()}\n")
            f.write(f"  唯一头数 (Layer-Head组合): {full_df[['Layer', 'Head']].drop_duplicates().shape[0]}\n\n")
            
            f.write(f"样本类型分布:\n")
            type_counts = full_df['Type'].value_counts()
            for type_name, count in type_counts.items():
                f.write(f"  {type_name}: {count} 个数据点 ({count/len(full_df)*100:.1f}%)\n")
            
            if 'AttentionRatio' in full_df.columns:
                f.write(f"\n注意力占比总体统计:\n")
                f.write(f"  平均值: {full_df['AttentionRatio'].mean():.6f}\n")
                f.write(f"  中位数: {full_df['AttentionRatio'].median():.6f}\n")
                f.write(f"  标准差: {full_df['AttentionRatio'].std():.6f}\n")
                f.write(f"  最小值: {full_df['AttentionRatio'].min():.6f}\n")
                f.write(f"  最大值: {full_df['AttentionRatio'].max():.6f}\n")
                
                f.write(f"\n层范围统计 (L{LAYER_RANGE[0]}-L{LAYER_RANGE[1]}):\n")
                layer_filtered = full_df[(full_df['Layer'] >= LAYER_RANGE[0]) & (full_df['Layer'] <= LAYER_RANGE[1])]
                if not layer_filtered.empty:
                    f.write(f"  数据点数: {len(layer_filtered)}\n")
                    f.write(f"  注意力占比平均值: {layer_filtered['AttentionRatio'].mean():.6f}\n")
                else:
                    f.write(f"  在指定层范围内没有数据\n")
            
            # 找出最常出现的头
            f.write(f"\n最常出现的前10个头 (跨所有样本):\n")
            head_counts = full_df.groupby(['Layer', 'Head']).size().reset_index(name='Count')
            head_counts = head_counts.sort_values('Count', ascending=False).head(10)
            for i, (_, row) in enumerate(head_counts.iterrows()):
                f.write(f"  {i+1}. L{int(row['Layer'])}-H{int(row['Head'])}: {int(row['Count'])} 次\n")
            
            # 记录保存的文件数量
            f.write(f"\n保存的文件统计:\n")
            f.write(f"  保存了 {len(TOP_N_VALUES)} 个不同n值的结果文件\n")
            f.write(f"  n值列表: {TOP_N_VALUES}\n")
            f.write(f"  每种类型样本保存 {len(TOP_N_VALUES) * 2} 个文件 (JSONL + CSV)\n")
            if 'Type' in full_df.columns:
                type_list = full_df['Type'].unique()
                f.write(f"  涉及 {len(type_list)} 种样本类型\n")
        
        print(f"✅ 汇总报告已保存至: {summary_path}")
        
    else:
        print("未找到任何数据，请检查路径和文件内容。")
        with open(log_file, 'a', encoding='utf-8') as log_f:
            log_f.write("未找到任何数据，请检查路径和文件内容。\n")
    
    print(f"\n✅ 分析完成! 所有结果保存在: {OUTPUT_DIR}")
    print(f"📝 详细日志: {log_file}")
    print(f"📊 共保存了 {len(TOP_N_VALUES)} 个不同n值的结果文件")

if __name__ == "__main__":
    main()