import os
import shutil
import base64
import io
import time
import re
import torch
from PIL import Image
from tqdm import tqdm
from transformers import AutoProcessor
from modeling_qwen2_5_vl_re_infer import Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info
from utiles import serialize_dict
import numpy as np

# 引用 Get_box 中的函数
from Get_box import (
    messages2out, 
    messages2att, 
    from_img_and_att_get_cropbox, 
    get_inputs,
    annotate_image_with_text, 
    get_candidates_from_raw_att, 
    pil_to_base64,
    verify_object_existence,
    visualize_attention_overlay,
    draw_all_candidates,
    stitch_components_spatially 
)

# --- 辅助函数：保存图片 ---
def save_images_to_folder(base_folder, sample_id, image_list, step_prefix=""):
    try:
        sample_folder = os.path.join(base_folder, f"sample_{sample_id}_process")
        if not os.path.exists(sample_folder):
            os.makedirs(sample_folder, exist_ok=True)
        
        print(f"   💾 Saving step: {step_prefix} ({len(image_list)} images)...")
        
        for idx, img_item in enumerate(image_list):
            try:
                suffix = ""
                if idx == 0: suffix = "_Original"
                elif idx == 1: suffix = "_MergedAll"
                elif idx == 2: suffix = "_MergedRefined"
                
                file_name = f"{step_prefix}_img{idx}{suffix}.png"
                save_path = os.path.join(sample_folder, file_name)
                
                if isinstance(img_item, str) and len(img_item) < 300 and os.path.exists(img_item):
                    shutil.copy(img_item, save_path)
                    continue

                if isinstance(img_item, str) and len(img_item) > 200:
                    b64_str = img_item
                    if "base64," in b64_str:
                        b64_str = b64_str.split("base64,")[1]
                    b64_str = "".join(b64_str.split())
                    missing_padding = len(b64_str) % 4
                    if missing_padding:
                        b64_str += '=' * (4 - missing_padding)
                    
                    img_data = base64.b64decode(b64_str)
                    img = Image.open(io.BytesIO(img_data)).convert("RGB")
                    img.save(save_path)
            except Exception:
                 pass
    except Exception as main_e:
        print(f"❌ [Folder Error] {main_e}")

# --- 辅助函数：询问哪个物体不清楚 ---
def ask_which_object_unclear(model, processor, ques, entities, current_answer):
    options_str = "\n".join([f"- {e}" for e in entities])
    prompt = f"""
    The user asked: "{ques}"
    You tentatively answered: "{current_answer}".
    
    We need to verify specific objects.
    Below is the list of REMAINING CANDIDATES:
    
    {options_str}
    
    TASK: Select the SINGLE object from the list above that is most critical to check.
    CONSTRAINTS:
    1. You MUST choose exactly one option from the list above.
    2. Output ONLY the exact text of the option.
    3. Wrap your answer in <FINAL_OUTPUT> tags.
    """
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], padding=True, return_tensors="pt").to(model.device)
    
    output_text, _, _ = messages2out(model, processor, inputs)
    raw_output = output_text[0]
    
    if "<FINAL_OUTPUT>" in raw_output and "</FINAL_OUTPUT>" in raw_output:
        entity = raw_output.split("<FINAL_OUTPUT>")[-1].split("</FINAL_OUTPUT>")[0]
    else:
        entity = raw_output
        
    entity = entity.strip().lower()
    entity = entity.replace("<", "").replace(">", "").replace('"', "").replace("'", "").replace(".", "")
    return entity

# --- 核心推理逻辑 ---
def once_infer(model, qwen_processor, sample, messages, img_url, ori_img_url, ques, sig, thre, save_root_dir, sample_id):
    prompt_ques = '''
You are a visual entity extractor. Extract the key physical objects mentioned in the question that are necessary to look at.
**Rules:**
1. Keep adjectives attached to their nouns. (e.g., "yellow balloon", "red car").
2. Do not use "with color" or "with size" formats. Use natural phrases.
3. Output the entities as a comma-separated list within <FINAL_OUTPUT> tags.

Example 1: "Is the drum on the left of the yellow balloon?" -> <FINAL_OUTPUT>drum, yellow balloon</FINAL_OUTPUT>
Example 2: "What is the man in the red shirt holding?" -> <FINAL_OUTPUT>man in red shirt</FINAL_OUTPUT>

Question: {input_text}
'''
# 2. 提取实体
    prompt_messages = [{"role": "user","content": [{"type": "text", "text": prompt_ques.format(input_text=ques)}],},]
    inputs = get_inputs(prompt_messages, qwen_processor, model)[3]
    prompt_output_text, _, _ = messages2out(model, qwen_processor, inputs)
    answer_out = prompt_output_text[0].split("<FINAL_OUTPUT>")[-1].split("</FINAL_OUTPUT>")[0]
    
    outputs = {}
    initial_entity_list = [e.strip().lower() for e in answer_out.split(',')] if answer_out else []
    search_query = answer_out if answer_out else ques

    # 3. 第一次 HiDe 注意力计算
    search_msg = [m.copy() for m in messages]
    search_msg[-1]["content"] = search_msg[-1]["content"][:-1]
    search_msg[-1]["content"].append({"type": "text", "text": "Search the following entities in the images: " + search_query})
    
    inputs = get_inputs(search_msg, qwen_processor, model)[3]
    attention, idx2word_dicts, img_start, img_end = messages2att(model, qwen_processor, inputs)
    
    # 传入 target_entities 进行聚合
    results = from_img_and_att_get_cropbox(
        inputs, attention, idx2word_dicts, img_url, img_start, img_end, sig, thre,
        target_entities=initial_entity_list,
        debug_dir=os.path.join(save_root_dir, f"sample_{sample_id}_process")
    )
    
    for s in sig:
        for t in thre:
            data = results[str(s)][str(t)]
# ========================================================
            # 🔥🔥🔥 新增：基于已识别实体的注意力叠加 (修正版) 🔥🔥🔥
            # ========================================================
            # 注意：data[5] 是 clean_raw_maps，存储了 { "entity_name": (img_idx, att_map) }
            # 这是 Get_box.py 内部已经匹配好的结果，比我们在外部重新匹配 Token 更可靠。
            if len(data) > 5:
                clean_raw_maps = data[5] 
                
                maps_save_dir = os.path.join(save_root_dir, f"sample_{sample_id}_process", f"S{s}_Object_Attention_Maps")
                if not os.path.exists(maps_save_dir):
                    os.makedirs(maps_save_dir, exist_ok=True)

                print(f"   💾 Aggregating EXTRACTED ENTITIES to {maps_save_dir} ...")
                
                # 按图片索引组织数据: { img_idx: [map1, map2, ...] }
                img_to_maps = {}
                found_entities = []

                # 1. 收集所有实体的 Attention Map
                for entity_name, (img_idx, att_map) in clean_raw_maps.items():
                    if img_idx not in img_to_maps:
                        img_to_maps[img_idx] = []
                    
                    # 确保是 numpy float32 且维度正确
                    map_data = np.array(att_map, dtype=np.float32)
                    map_data = np.nan_to_num(map_data, nan=0.0)
                    if map_data.ndim == 3: map_data = map_data.squeeze()
                    
                    if map_data.ndim == 2:
                        img_to_maps[img_idx].append(map_data)
                        found_entities.append(entity_name)

                print(f"   🎯 Found maps for entities: {found_entities}")

                # 2. 对每张图进行叠加并保存
                if not img_to_maps:
                    print("   ⚠️ clean_raw_maps is empty! (Get_box failed to match tokens)")
                    # 调试：打印一下 idx2word_dicts 的前几个看看 Token 长什么样
                    print(f"   🔍 Debug Token Samples: {list(idx2word_dicts.values())[:10]}")
                
                for img_idx, map_list in img_to_maps.items():
                    if img_idx >= len(img_url): continue
                    
                    current_img_path = img_url[img_idx]
                    
                    # 叠加所有物体的 Map
                    aggregated_map = None
                    for m in map_list:
                        if aggregated_map is None:
                            aggregated_map = m
                        else:
                            # 形状对齐
                            if aggregated_map.shape != m.shape:
                                m = cv2.resize(m, (aggregated_map.shape[1], aggregated_map.shape[0]))
                            aggregated_map += m  # 累加
                    
                    if aggregated_map is not None:
                        save_name = f"Img{img_idx}_Merged_Objects_{len(map_list)}_Entities.jpg"
                        save_path = os.path.join(maps_save_dir, save_name)
                        
                        visualize_attention_overlay(current_img_path, aggregated_map, save_path, alpha=0.6)
                        print(f"   ✅ Saved Merged Object Map for Img {img_idx}")
            # ========================================================
            initial_components = data[3] 
            raw_att_maps = data[5]
                    
            refined_entities = set()
            MAX_LOOPS = 5 
            #CONFIDENCE_THRESHOLD = 0.95
            CONFIDENCE_THRESHOLD = 1.
            
            final_confidence = 0.0
            final_output_text = ""
            final_messages = []
            
            confidence_history = []
            answer_history = []  # 🔥 [新增] 初始化回答历史列表
            
            all_focus_components = list(initial_components)
            refined_components = []
            
            def build_current_image_set():
                imgs = list(ori_img_url)
                # 使用带 label 逻辑的 stitch 函数
                merged_all = stitch_components_spatially(all_focus_components)
                if merged_all: imgs.append(merged_all)
                if refined_components:
                    merged_refined = stitch_components_spatially(refined_components)
                    if merged_refined: imgs.append(merged_refined)
                return imgs

            current_image_set = build_current_image_set()
            save_images_to_folder(save_root_dir, sample_id, current_image_set, step_prefix=f"S{s}_T{t}_Step0_Initial")

            loop_idx = 0
            print(f"\n🚀 [Refine Start] Maps: {list(raw_att_maps.keys())}")

            while loop_idx < MAX_LOOPS:
                loop_messages = [{"role": "user", "content": []}]
                for img in current_image_set:
                    loop_messages[-1]["content"].append({"type": "image", "image": img})
                loop_messages[-1]["content"].append({"type": "text", "text": ques + "\nAnswer with the option's letter from the given choices directly."})
                
                inputs = get_inputs(loop_messages, qwen_processor, model)[3]
                output_text, _, confidence = messages2out(model, qwen_processor, inputs)
                
                # 🔥 [新增] 记录当前步骤的回答
                answer_history.append(output_text[0])
                
                final_confidence = confidence
                final_output_text = output_text
                final_messages = loop_messages
                confidence_history.append(confidence)
                
                print(f"   📊 Loop {loop_idx} Confidence: {confidence:.4f} | Answer: {output_text[0]}")
                
                available_entities = [e for e in initial_entity_list if e not in refined_entities]
                if confidence >= CONFIDENCE_THRESHOLD: break
                if not available_entities: break
                
                unclear_entity = ask_which_object_unclear(model, qwen_processor, ques, available_entities, output_text[0])
                print(f"👉 Focus: '{unclear_entity}'")
                refined_entities.add(unclear_entity)
                
                target_data = None
                if unclear_entity in raw_att_maps: target_data = raw_att_maps[unclear_entity]
                else:
                    for k in raw_att_maps:
                        if unclear_entity in k or k in unclear_entity:
                            target_data = raw_att_maps[k]; break
                if not target_data: loop_idx += 1; continue
                
                target_img_idx, target_att_map = target_data
                
                safe_entity = "".join([c if c.isalnum() else "_" for c in unclear_entity])
                debug_sub_folder = os.path.join(save_root_dir, f"sample_{sample_id}_process", f"S{s}_Loop{loop_idx}_{safe_entity}")
                if not os.path.exists(debug_sub_folder): os.makedirs(debug_sub_folder, exist_ok=True)
                
                visualize_attention_overlay(img_url[target_img_idx], target_att_map, os.path.join(debug_sub_folder, "Att_Overlay.jpg"))
                candidates_list = get_candidates_from_raw_att(target_att_map, img_url, target_img_idx, n_candidates=20)
                draw_all_candidates(img_url[target_img_idx], candidates_list, os.path.join(debug_sub_folder, "All_Candidates.jpg"))
                
                best_conf = -100.0
                best_item = None
                
                for item in candidates_list:
                    cand_b64 = item['b64']
                    cand_conf = verify_object_existence(cand_b64, unclear_entity, model, qwen_processor, model.device)
                    if cand_conf > best_conf:
                        best_conf = cand_conf
                        best_item = item
                
                added_new_img = False
                if best_item:
                    try:
                        b64 = best_item['b64']
                        if "base64," in b64: b64 = b64.split("base64,")[1]
                        pil_crop = Image.open(io.BytesIO(base64.b64decode(b64))).convert("RGB")
                        
                        # 🔥 关键：不直接写字，只存 label 字符串
                        new_component = {
                            'pil_img': pil_crop,
                            'bbox': best_item['pixel_bbox'],
                            'label': unclear_entity 
                        }
                        
                        all_focus_components.append(new_component)
                        refined_components.append(new_component)
                        
                        current_image_set = build_current_image_set()
                        
                        if best_conf > 0.0: print(f"   ✅ Refined View Added (Conf: {best_conf:.2f})")
                        else: print(f"   ⚠️ Best Available Added (Low Conf: {best_conf:.2f})")
                            
                        added_new_img = True
                    except Exception as e:
                        print(f"   ❌ Merge Error: {e}")
                else:
                    print(f"   ❌ No Candidates.")
                
                if added_new_img:
                    save_images_to_folder(save_root_dir, sample_id, current_image_set, step_prefix=f"S{s}_T{t}_Step{loop_idx+1}_Refined")
                
                loop_idx += 1
            
            save_images_to_folder(save_root_dir, sample_id, current_image_set, step_prefix=f"S{s}_T{t}_Final")
            
            serializable_comps = []
            for c in all_focus_components:
                serializable_comps.append({'b64': pil_to_base64(c['pil_img']), 'bbox': c['bbox']})
            data[3] = serializable_comps
            
            if not str(s) in outputs: outputs[str(s)] = {}
            
            # 🔥 [修改] 将 answer_history 添加到返回列表中（index 10）
            outputs[str(s)][str(t)] = [
                [answer_out], 
                final_output_text, 
                data[1], 
                current_image_set, 
                final_messages, 
                data[2], 
                data[0], 
                data[4], 
                final_confidence,
                confidence_history,
                answer_history  # <--- 新增这一项
            ]

    return outputs

# --- 主循环 ---
def cycle_epoch_infer(gpu_id, rank, dataset_part, savedir, max_pixels, sig, thre):
    device = f"cuda:{gpu_id}"
    print(rank, len(dataset_part), device)

    model_path = r"/data2/shaos/data/Qwen/Qwen2.5-VL-7B-Instruct"
    model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
        model_path, torch_dtype=torch.bfloat16, attn_implementation="flash_attention_2", device_map=device
    )
    qwen_processor = AutoProcessor.from_pretrained(model_path, use_fast=True, min_pixels=256*28*28, max_pixels=max_pixels*28*28)

    save_root_dir = os.path.dirname(savedir) if os.path.dirname(savedir) else "."
    save_root_dir = os.path.join(save_root_dir, "saved_inference_process")
    if not os.path.exists(save_root_dir):
        os.makedirs(save_root_dir, exist_ok=True)

    for sample in tqdm(dataset_part):
        results = sample
        img_url = [sample["image"]]
        ori_img_url = list(img_url)
        messages = [{"role": "user", "content": []}]
        for img in img_url:
            messages[-1]["content"].append({"type": "image", "image": img})

        ques = sample["Text"]
        messages[-1]["content"].append({"type": "text", "text": ques+"\nAnswer with the option's letter from the given choices directly."})
        text,image_inputs,video_inputs,inputs,video_kwargs = get_inputs(messages,qwen_processor,model)
        output_text, end_ques, _ = messages2out(model,qwen_processor,inputs)
        
        results["answer"] = {}
        results["answer"]["ori"] = output_text[0]
        results["bounding_box"] = {}
        results["prompt_text"] = {}
        results["confidence_history"] = {} 
        results["step_answers"] = {}  # 🔥 [新增] 初始化步骤回答字段
        
        torch.cuda.empty_cache()
        
        outputs = once_infer(model, qwen_processor, sample, messages, img_url, ori_img_url, ques, sig, thre, save_root_dir, sample["id"])
        
        for s in sig:
            for t in thre:
                data = outputs[str(s)][str(t)]
                results["answer"][f"HiDe_s{s}_t{t}"] = data[1][0]
                results["prompt_text"][f"HiDe"] = data[0][0]
                results["bounding_box"][f"HiDe_s{s}_t{t}"] = data[7]
                
                # 兼容旧版本：检查索引是否存在
                if len(data) > 9:
                    results["confidence_history"][f"HiDe_s{s}_t{t}"] = data[9]
                
                # 🔥 [新增] 保存步骤回答
                if len(data) > 10:
                    results["step_answers"][f"HiDe_s{s}_t{t}"] = data[10]
        
        serialize_dict(results,savedir)
        torch.cuda.empty_cache()
        print(f"Results saved to {savedir}")
    
    del model
    torch.cuda.empty_cache()