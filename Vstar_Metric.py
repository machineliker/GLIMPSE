import json
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import defaultdict
import matplotlib

# 设置无头模式，防止在 Linux 服务器无显示器环境下报错
matplotlib.use('Agg') 

# ==========================================
# 1. 基础工具模块
# ==========================================

def read_multi_line_json_objects(file_path):
    """
    读取可能包含格式错误的 JSONL 文件 (Robust Reader)
    """
    data = []
    if not os.path.exists(file_path):
        print(f"❌ 文件不存在: {file_path}")
        return []
        
    print(f"📂 正在读取文件: {os.path.basename(file_path)} ...")
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read().strip()
        # 修复连在一起的 JSON 对象 (例如 '}\n{')
        objects = content.replace('}\n{', '}|-|-|{').split('|-|-|')
        
        for obj_str in objects:
            if not obj_str.strip(): continue
            try:
                parsed_obj = json.loads(obj_str)
                # 过滤掉非字典类型的脏数据
                if isinstance(parsed_obj, dict):
                    data.append(parsed_obj)
            except: pass 
            
    print(f"✅ 成功加载 {len(data)} 条有效样本")
    return data

def normalize_answer(ans):
    """标准化答案格式 (处理 list 或 str, 统一转大写)"""
    if isinstance(ans, list): 
        ans = ans[-1]
    return str(ans).strip().upper()

def get_method_key(data):
    """自动检测数据中的主要 Key (如 HiDe_s3_t0.7)"""
    for item in data:
        if "step_answers" in item:
            keys = list(item["step_answers"].keys())
            if keys: return keys[0]
    return None

# ==========================================
# 2. 核心分析模块
# ==========================================

def get_accuracy_at_specific_threshold(data, method_key, target_threshold):
    """
    功能：输入一个置信度阈值，返回该设置下的准确率详情。
    """
    print("\n" + "="*80)
    print(f"🎯 查询特定阈值表现 (Target Threshold: {target_threshold})")
    print("="*80)
    
    samples = [d for d in data if "step_answers" in d and method_key in d["step_answers"]]
    total = len(samples)
    
    if total == 0:
        print("❌ 没有有效样本。")
        return

    correct_last = 0
    correct_max = 0
    early_exit_count = 0  # 统计有多少样本触发了早停

    for item in samples:
        gt = normalize_answer(item["Ground truth"])
        ans_list = item["step_answers"][method_key]
        conf_list = item["confidence_history"][method_key]
        
        # 模拟早停逻辑
        hit = False
        curr_ans = None
        
        for i, conf in enumerate(conf_list):
            if conf >= target_threshold:
                hit = True
                curr_ans = normalize_answer(ans_list[i])
                break
        
        if hit:
            early_exit_count += 1
            # 触发早停时，两种策略结果一样
            pred_last = curr_ans
            pred_max = curr_ans
        else:
            # 未触发早停，进入保底逻辑
            pred_last = normalize_answer(ans_list[-1]) # 策略 A: 坚持到底
            pred_max = normalize_answer(ans_list[np.argmax(conf_list)]) # 策略 B: 回溯最高置信度
        
        if pred_last == gt: correct_last += 1
        if pred_max == gt: correct_max += 1

    acc_last = correct_last / total
    acc_max = correct_max / total
    exit_rate = early_exit_count / total

    print(f"📊 样本总数: {total}")
    print(f"⚡ 早停触发率 (Early Exit Rate): {exit_rate:.2%} (即 {early_exit_count} 个样本在置信度 >= {target_threshold} 时停止)")
    print("-" * 65)
    print(f"{'Strategy (保底策略)':<30} | {'Accuracy (准确率)':<20}")
    print("-" * 65)
    print(f"{'Fallback: Last Step (最后一步)':<30} | {acc_last:.2%} ({correct_last}/{total})")
    print(f"{'Fallback: Max Conf  (最高置信度)':<30} | {acc_max:.2%} ({correct_max}/{total})")
    print("-" * 65)
    
    return acc_last, acc_max

def analyze_optimal_threshold_curve(data, method_key):
    """
    功能：遍历 0-1 的所有阈值，寻找最佳点并画图
    """
    print("\n" + "="*80)
    print("📈 分析: 最佳置信度阈值搜索 (Global Search)")
    print("="*80)
    
    samples = [d for d in data if "step_answers" in d and method_key in d["step_answers"]]
    thresholds = np.linspace(0, 1.0, 101)
    acc_last = []
    acc_max = []

    for t in thresholds:
        c_last = 0
        c_max = 0
        total = len(samples)
        
        for item in samples:
            gt = normalize_answer(item["Ground truth"])
            ans_list = item["step_answers"][method_key]
            conf_list = item["confidence_history"][method_key]
            
            # 模拟早停
            hit = False
            curr_ans = None
            for i, conf in enumerate(conf_list):
                if conf >= t:
                    hit = True
                    curr_ans = normalize_answer(ans_list[i])
                    break
            
            pred_last = curr_ans if hit else normalize_answer(ans_list[-1])
            pred_max = curr_ans if hit else normalize_answer(ans_list[np.argmax(conf_list)])
            
            if pred_last == gt: c_last += 1
            if pred_max == gt: c_max += 1
            
        acc_last.append(c_last / total)
        acc_max.append(c_max / total)
        
    best_idx_last = np.argmax(acc_last)
    best_idx_max = np.argmax(acc_max)
    
    print(f"策略 A (Fallback Last) | 最佳阈值: {thresholds[best_idx_last]:.2f} | 最高准确率: {acc_last[best_idx_last]:.2%}")
    print(f"策略 B (Fallback Max)  | 最佳阈值: {thresholds[best_idx_max]:.2f} | 最高准确率: {acc_max[best_idx_max]:.2%}")

    # 画图
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, acc_last, label='Fallback: Last', color='blue', linewidth=2)
    plt.plot(thresholds, acc_max, label='Fallback: Max', color='red', linestyle='--', linewidth=2)
    
    # 标记最高点
    plt.scatter(thresholds[best_idx_last], acc_last[best_idx_last], c='blue', s=100, zorder=5)
    plt.scatter(thresholds[best_idx_max], acc_max[best_idx_max], c='red', s=100, zorder=5)
    
    plt.title(f'Accuracy vs Threshold ({method_key})')
    plt.xlabel('Confidence Threshold')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 1.05)
    
    save_name = 'accuracy_curve.png'
    plt.savefig(save_name, dpi=300)
    print(f"🖼️  曲线图已保存为: {save_name}")

def analyze_category_accuracy(data, method_key):
    """
    功能：按类别统计 Ori, Last, MaxConf 的准确率
    """
    print("\n" + "="*80)
    print("📊 分析: 不同类别的准确率 (Category Accuracy Table)")
    print("="*80)
    
    stats = defaultdict(lambda: defaultdict(lambda: {'correct': 0, 'total': 0}))
    strategies = ['Ori', 'Last', 'MaxConf']
    
    for item in data:
        if "step_answers" not in item or method_key not in item["step_answers"]:
            continue
            
        cat = item.get("category", item.get("Category", "Unknown"))
        gt = normalize_answer(item["Ground truth"])
        
        ans_list = item["step_answers"][method_key]
        conf_list = item["confidence_history"][method_key]
        
        preds = {}
        preds['Ori'] = normalize_answer(item["answer"].get("ori", "INVALID"))
        preds['Last'] = normalize_answer(ans_list[-1])
        preds['MaxConf'] = normalize_answer(ans_list[np.argmax(conf_list)])
        
        for strat in strategies:
            stats[cat][strat]['total'] += 1
            stats['Overall'][strat]['total'] += 1 
            if preds[strat] == gt:
                stats[cat][strat]['correct'] += 1
                stats['Overall'][strat]['correct'] += 1

    rows = []
    sorted_cats = sorted([c for c in stats.keys() if c != 'Overall']) + ['Overall']
    
    for cat in sorted_cats:
        row = {'Category': cat}
        for strat in strategies:
            c = stats[cat][strat]['correct']
            t = stats[cat][strat]['total']
            acc = c / t if t > 0 else 0
            # 格式: 85.00% (17/20)
            row[f'{strat}'] = f"{acc:.2%} ({c}/{t})"
        rows.append(row)
        
    df = pd.DataFrame(rows)
    # 打印表格
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_colwidth', 30)
    print(df.to_string(index=False))

def export_error_samples(data, method_key, target_strategy='MaxConf'):
    """
    功能：导出预测错误的样本 ID
    """
    print("\n" + "="*80)
    print(f"📋 分析: 提取错误样本 ID (Strategy: {target_strategy})")
    print("="*80)
    
    error_ids = []
    for item in data:
        if "step_answers" not in item or method_key not in item["step_answers"]:
            continue
            
        sample_id = item.get("id", "unknown")
        gt = normalize_answer(item["Ground truth"])
        ans_list = item["step_answers"][method_key]
        conf_list = item["confidence_history"][method_key]
        
        pred = ""
        if target_strategy == 'Last':
            pred = normalize_answer(ans_list[-1])
        elif target_strategy == 'MaxConf':
            max_idx = np.argmax(conf_list)
            pred = normalize_answer(ans_list[max_idx])
        elif target_strategy == 'Ori':
             pred = normalize_answer(item["answer"].get("ori", ""))
             
        if pred != gt:
            error_ids.append(str(sample_id))
            
    filename = f"error_ids_{target_strategy}.txt"
    with open(filename, "w") as f:
        f.write("\n".join(error_ids))
        
    print(f"发现 {len(error_ids)} 个错误样本。前10个ID: {error_ids[:10]}")
    print(f"💾 已保存至: {filename}")

# ==========================================
# 主执行入口
# ==========================================

if __name__ == "__main__":
    # 🔴 设置：JSONL 文件路径
    file_path = r"/data1/shaos/labs/HiDe_3/data1/shaos/labs/HiDe/Hide/HR_8k_results_1confidence_prompt_easy_all_step_answer_save_top_6_heads.json"
    
    # 🔴 设置：你想查询的具体阈值
    my_target_threshold = 0.95

    # 读取数据
    data = read_multi_line_json_objects(file_path)
    
    if data:
        # 1. 自动获取主 Key (e.g. HiDe_s3_t0.7)
        key = get_method_key(data)
        
        if key:
            print(f"🔍 检测到方法 Key: {key}")
            
            # 功能 1: 查询指定阈值下的准确率
            get_accuracy_at_specific_threshold(data, key, my_target_threshold)
            
            # 功能 2: 绘制最佳阈值曲线 (生成 accuracy_curve.png)
            analyze_optimal_threshold_curve(data, key)
            
            # 功能 3: 打印分类别准确率表格
            analyze_category_accuracy(data, key)
            
            # 功能 4: 导出错误样本 ID (默认导出 MaxConf 策略下的错误)
            export_error_samples(data, key, target_strategy='MaxConf')
            
        else:
            print("❌ 数据中未找到 'step_answers' 字段，请检查推理代码是否正确保存了每步结果。")