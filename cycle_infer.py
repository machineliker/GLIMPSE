import os
import time
import random
import traceback
import subprocess
import numpy as np
import torch.multiprocessing as mp
from multiprocessing import Pool
from tqdm import tqdm
from utiles import load_dataset_Vstar_json
from inference import cycle_epoch_infer

# --- 🔥 新增：子进程保护壳函数 ---
def worker_wrapper(func, *args, **kwargs):
    try:
        return func(*args, **kwargs)
    except Exception as e:
        print(f"\n❌ [子进程报错] 捕获到异常: {e}")
        print("="*60)
        traceback.print_exc()
        print("="*60)
        raise e

def get_available_gpus(max_memory_mb=1000, max_gpus=None):
    """获取可用GPU列表 (已包含屏蔽0号卡逻辑)"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=memory.used', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, check=True
        )
        used_memory = [int(x.strip()) for x in result.stdout.strip().split('\n')]
        gpu_memory_pairs = [(i, mem) for i, mem in enumerate(used_memory)]
        gpu_memory_pairs.sort(key=lambda x: x[1])
        # 屏蔽 0 号卡
        available_gpus = [gpu_id for gpu_id, mem in gpu_memory_pairs if mem < max_memory_mb and gpu_id != 0]
        if max_gpus is not None:
            available_gpus = available_gpus[:max_gpus]
        return available_gpus
    except Exception as e:
        print(f"Error detecting GPU memory: {e}")
        return []

def main(datasetdir, savedir, max_pixels, Parallels, sig, thre, head_config_path, para_nums=6):
    # 设置环境变量
    if head_config_path:
        if os.path.exists(head_config_path):
            os.environ["HEAD_CONFIG_PATH"] = head_config_path
            print(f"🔧 [Config] Head 配置文件路径: {head_config_path}")
        else:
            print(f"⚠️ [Warning] Head 配置文件不存在: {head_config_path}")
            # 如果配置文件必须存在才能跑，这里建议直接 return，防止跑出错误结果
            return 

    if not Parallels: para_nums = 1
    
    # 加载数据集
    dataset = load_dataset_Vstar_json(datasetdir)
    random.shuffle(dataset)
    
    available_gpus = get_available_gpus(max_memory_mb=1000, max_gpus=para_nums)
    if len(available_gpus) == 0:
        print("❌ 没有找到符合条件的空闲 GPU")
        return
        
    print(f"✅ 找到 {len(available_gpus)} 个可用 GPU: {available_gpus}")
    
    splits = np.array_split(dataset, len(available_gpus))
    print("文件加载完成")
    
    if not Parallels:
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            cycle_epoch_infer(gpu_id, rank, dataset_part, savedir, max_pixels, sig, thre)
    else:
        pool = Pool(processes=len(available_gpus))
        results = []
        for rank, gpu_id in tqdm(enumerate(available_gpus)):
            dataset_part = splits[rank]
            res = pool.apply_async(
                worker_wrapper,
                args=(cycle_epoch_infer, gpu_id, rank, dataset_part, savedir, max_pixels, sig, thre),
                error_callback=lambda e: print(f"⚠️ 主进程感知到子进程错误: {e}") 
            )
            results.append(res)
        
        pool.close()
        for res in tqdm(results, desc="等待所有进程完成"):
            res.wait()
        pool.join()

if __name__ == "__main__":
    mp.set_start_method('spawn', force=True)
    
    # 固定参数设置
    maxp = 16384
    Parallels = True
    sigma = [3]
    threshold = [0.7]
    seed = 2077
    #datasetdir = f"/data2/shaos/data/vstar_bench/test_questions_converted.json"
    datasetdir = f"/data2/shaos/data/vstar_bench/relative_position_154.jsonl"
    # 🔥🔥🔥 循环运行 1 到 100 🔥🔥🔥

    print(f"\n{'#'*30}")
    print(f"🚀 开始运行: Top Heads")
    print(f"{'#'*30}\n")
    
    # 1. 动态设置 Head Config 路径
    head_config_jsonl = f"/data2/shaos/labs/mllms_know_94.1_head_select/head_analysisi/head_analysis_results/All_Samples_top_6_heads_by_ratio_L0_to_L27.jsonl"
    
    # 2. 动态设置保存路径 (防止覆盖)
    # 注意：我也把输出文件名改成了 ...top_{n}_heads.json
    savejson = f"vstar_results_1confidence_prompt_easy_all_step_answer_save_top_6_heads.json"
    
    # 3. 检查文件是否存在，避免报错
    if not os.path.exists(head_config_jsonl):
        print(f"❌ 跳过: 配置文件未找到 -> {head_config_jsonl}")

    try:
        # 每次循环重置随机种子，确保除Head外其他条件一致
        random.seed(seed)
        
        # 执行主函数
        main(datasetdir, savejson, maxp, Parallels, sigma, threshold, head_config_jsonl, 4)
        
        print(f"✅ 完成: Top 6 Heads -> 结果已保存至 {savejson}")
        
    except Exception as e:
        print(f"❌ 运行 Top 6时发生错误: {e}")
        traceback.print_exc()
    
    # 可选：稍作停顿让显存稍微释放一下，虽然 spawn 模式下子进程结束会自动释放
    time.sleep(2)

    print("\n🎉🎉🎉 所有 1-100 任务执行完毕！")