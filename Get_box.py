import os
from transformers import AutoProcessor
from qwen_vl_utils import process_vision_info
import torch
import torch.nn.functional as F
import numpy as np
from skimage.measure import label, regionprops
import cv2
from PIL import Image, ImageDraw, ImageFont
import io
import base64
from utiles import *

def get_candidates_from_raw_att(att_map, img_url, img_idx, n_candidates=50):
    if img_idx >= len(img_url): return []
    candidates_info = get_focus_region_candidates(att_map, top_n=n_candidates)
    results = []
    try:
        current_img_str = img_url[img_idx]
        if os.path.exists(current_img_str): pil_img = Image.open(current_img_str).convert("RGB")
        elif "base64," in current_img_str: pil_img = Image.open(io.BytesIO(base64.b64decode(current_img_str.split("base64,")[1]))).convert("RGB")
        else: pil_img = Image.open(io.BytesIO(base64.b64decode(current_img_str))).convert("RGB")
    except: return []
    W, H = pil_img.size
    for item in candidates_info:
        nx1, ny1, nx2, ny2 = item['bbox']
        x1, y1 = max(0, int(nx1 * W)), max(0, int(ny1 * H))
        x2, y2 = min(W, int(nx2 * W)), min(H, int(ny2 * H))
        if x2 <= x1 or y2 <= y1: continue
        crop = pil_img.crop((x1, y1, x2, y2))
        results.append({'b64': pil_to_base64(crop), 'pixel_bbox': [x1, y1, x2, y2], 'score': item['score']})
    return results
    
# ==========================================
# 核心算法：基于原始坐标的空间拼接 + 动态消隐
# ==========================================
def stitch_components_spatially(components, bg_color=(255, 255, 255), gap=10):
    """
    components: list of dict {'pil_img': PIL, 'bbox': [x1,y1,x2,y2], 'label': str/None}
    """
    if not components: return None
    
    # 1. 提取所有坐标，构建压缩映射
    xs = sorted(list(set([c['bbox'][0] for c in components] + [c['bbox'][2] for c in components])))
    ys = sorted(list(set([c['bbox'][1] for c in components] + [c['bbox'][3] for c in components])))
    
    x_map = {}
    current_x = gap
    for i in range(len(xs)):
        x_map[xs[i]] = current_x
        if i < len(xs) - 1:
            dist = xs[i+1] - xs[i]
            has_content = False
            for c in components:
                if c['bbox'][0] <= xs[i] and c['bbox'][2] >= xs[i+1]:
                    has_content = True; break
            if has_content: current_x += dist
            else: current_x += min(dist, gap)
            
    y_map = {}
    current_y = gap
    for i in range(len(ys)):
        y_map[ys[i]] = current_y
        if i < len(ys) - 1:
            dist = ys[i+1] - ys[i]
            has_content = False
            for c in components:
                if c['bbox'][1] <= ys[i] and c['bbox'][3] >= ys[i+1]:
                    has_content = True; break
            if has_content: current_y += dist
            else: current_y += min(dist, gap)

    # 2. 计算每个组件在画布上的放置位置 (rect)
    placements = []
    min_draw_x, min_draw_y = float('inf'), float('inf')
    max_draw_x, max_draw_y = 0, 0
    
    for i, c in enumerate(components):
        x1, y1, x2, y2 = c['bbox']
        img = c['pil_img']
        
        # 映射后的左上角
        paste_x = x_map[x1]
        paste_y = y_map[y1]
        
        w, h = img.size
        
        # 记录放置信息
        placements.append({
            'index': i,
            'img': img,
            'label': c.get('label', None), # 获取原始标签
            'rect': [paste_x, paste_y, paste_x + w, paste_y + h], # 绝对坐标 [x1, y1, x2, y2]
            'draw_label': True # 默认为True，碰撞检测后可能设为False
        })
        
        min_draw_x = min(min_draw_x, paste_x)
        min_draw_y = min(min_draw_y, paste_y)
        max_draw_x = max(max_draw_x, paste_x + w)
        max_draw_y = max(max_draw_y, paste_y + h)

    # 3. 碰撞检测：如果两个图重叠，则去掉它们的标签
    n = len(placements)
    for i in range(n):
        for j in range(i + 1, n):
            rect_a = placements[i]['rect']
            rect_b = placements[j]['rect']
            
            # 计算重叠
            x_left = max(rect_a[0], rect_b[0])
            y_top = max(rect_a[1], rect_b[1])
            x_right = min(rect_a[2], rect_b[2])
            y_bottom = min(rect_a[3], rect_b[3])
            
            # 只要有重叠区域
            if x_right > x_left and y_bottom > y_top:
                placements[i]['draw_label'] = False
                placements[j]['draw_label'] = False

    # 4. 创建画布
    canvas_w = max(100, int(max_draw_x - min_draw_x + gap * 2))
    canvas_h = max(100, int(max_draw_y - min_draw_y + gap * 2))
    canvas = Image.new("RGB", (canvas_w, canvas_h), bg_color)
    
    offset_x = gap - min_draw_x
    offset_y = gap - min_draw_y
    
    # 5. 绘制
    for p in placements:
        final_img = p['img']
        label_text = p['label']
        
        # 只有当允许绘制且有标签时，才处理文字
        if p['draw_label'] and label_text:
            final_img = annotate_image_with_text(final_img, label_text)
            
        canvas.paste(final_img, (int(p['rect'][0] + offset_x), int(p['rect'][1] + offset_y)))
        
    return pil_to_base64(canvas)

# ==========================================
# 图像标注 (PIL Logic)
# ==========================================
def draw_wrapped_text(draw, text, font, pos, max_width, bg_color, text_color, measure_only=False):
    x, y = pos
    lines = []
    words = text.split()
    current_line = []
    
    for word in words:
        test_line = ' '.join(current_line + [word])
        bbox = draw.textbbox((0, 0), test_line, font=font)
        w = bbox[2] - bbox[0]
        if w <= max_width: current_line.append(word)
        else:
            if current_line: lines.append(' '.join(current_line)); current_line = [word]
            else: lines.append(word); current_line = []
    if current_line: lines.append(' '.join(current_line))
    
    line_heights = []
    max_line_w = 0
    for line in lines:
        bbox = draw.textbbox((0, 0), line, font=font)
        max_line_w = max(max_line_w, bbox[2] - bbox[0])
        line_heights.append(bbox[3] - bbox[1])
    
    line_spacing = 4; pad = 4 
    if not lines: total_h, bg_w, bg_h = 0, 0, 0
    else:
        total_h = sum(line_heights) + (len(lines) - 1) * line_spacing
        bg_w = max_line_w + 2 * pad
        bg_h = total_h + 2 * pad
    
    if measure_only: return bg_w, bg_h

    draw.rectangle((x, y, x + bg_w, y + bg_h), fill=bg_color)
    curr_y = y + pad
    for i, line in enumerate(lines):
        draw.text((x + pad, curr_y), line, fill=text_color, font=font)
        curr_y += line_heights[i] + line_spacing
    return bg_w, bg_h

def annotate_image_with_text(pil_img, text, bg_color=(105, 105, 105, 200), text_color=(255, 255, 255, 255)):
    # 如果没有文本，直接返回原图
    if not text: return pil_img
    
    img = pil_img.convert("RGBA")
    draw = ImageDraw.Draw(img, "RGBA")
    item_w, item_h = img.size
    
    ref_size = min(item_w, item_h)
    font_size = int(np.clip(ref_size * 0.2, 10, 24))
    try: font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except: font = ImageFont.load_default()

    max_text_width = min(item_w, max(80, int(item_w * 1.5)))
    text_w, text_h = draw_wrapped_text(draw, text, font, (0, 0), max_text_width, bg_color, text_color, measure_only=True)

    pad = 2
    draw_x = 0
    if draw_x + text_w > item_w: draw_x = max(0, item_w - text_w)
    draw_y = 0 + pad 

    draw_wrapped_text(draw, text, font, (draw_x, draw_y), max_text_width, bg_color, text_color)
    return img.convert("RGB")

def pil_to_base64(pil_img, format="PNG"):
    buffered = BytesIO()
    img_format = pil_img.format if pil_img.format else format
    pil_img.save(buffered, format=img_format)
    return f"data:image;base64,{base64.b64encode(buffered.getvalue()).decode('utf-8')}"

# ==========================================
# 辅助函数
# ==========================================
def extract_anchor_points(rel_map, k=50, min_distance=2):
    H, W = rel_map.shape
    flat_indices = np.argsort(rel_map.flatten())[::-1]
    anchor_points = []
    for idx in flat_indices:
        if len(anchor_points) >= k: break
        y = idx // W; x = idx % W
        score = rel_map[y, x]
        too_close = False
        for (ax, ay, _) in anchor_points:
            dist = np.sqrt((x - ax)**2 + (y - ay)**2)
            if dist < min_distance: too_close = True; break
        if not too_close and score > 0: anchor_points.append((x, y, score))
    return anchor_points

def generate_stepwise_adaptive_roi(rel_map, anchor, min_side=2, max_side=24, step_size=2, expansion_threshold=0.2):
    H, W = rel_map.shape
    x_center, y_center, anchor_score = anchor
    half_min = max(1, min_side // 2)
    left = max(0, x_center - half_min); right = min(W-1, x_center + half_min)
    top = max(0, y_center - half_min); bottom = min(H-1, y_center + half_min)
    current_step = step_size
    for _ in range(20):
        if (right - left + 1) >= max_side and (bottom - top + 1) >= max_side: break
        direction_scores = {}
        if left >= current_step:
            reg = rel_map[top:bottom+1, max(0, left - current_step):left]
            direction_scores['left'] = reg.mean() if reg.size > 0 else 0
        if right + current_step < W:
            reg = rel_map[top:bottom+1, right+1:min(W-1, right + current_step)+1]
            direction_scores['right'] = reg.mean() if reg.size > 0 else 0
        if top >= current_step:
            reg = rel_map[max(0, top - current_step):top, left:right+1]
            direction_scores['up'] = reg.mean() if reg.size > 0 else 0
        if bottom + current_step < H:
            reg = rel_map[bottom+1:min(H-1, bottom + current_step)+1, left:right+1]
            direction_scores['down'] = reg.mean() if reg.size > 0 else 0
        if not direction_scores: break
        expanded = False
        threshold_val = anchor_score * expansion_threshold
        for direction, score in direction_scores.items():
            if score >= threshold_val:
                expanded = True
                if direction == 'left': left = max(0, left - current_step)
                elif direction == 'right': right = min(W-1, right + current_step)
                elif direction == 'up': top = max(0, top - current_step)
                elif direction == 'down': bottom = min(H-1, bottom + current_step)
        if not expanded:
            if current_step > 1: current_step = max(1, current_step // 2)
            else: break
    return (left, top, right, bottom)

def non_maximum_suppression(rois, scores, iou_threshold=0.3):
    if not rois: return []
    boxes = np.array(rois); scores = np.array(scores)
    x1 = boxes[:, 0]; y1 = boxes[:, 1]; x2 = boxes[:, 2]; y2 = boxes[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]; keep = []
    while order.size > 0:
        i = order[0]; keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]]); yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]]); yy2 = np.minimum(y2[i], y2[order[1:]])
        w = np.maximum(0.0, xx2 - xx1 + 1); h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h; union = areas[i] + areas[order[1:]] - inter
        iou = inter / (union + 1e-6)
        inds = np.where(iou <= iou_threshold)[0]
        order = order[inds + 1]
    return keep

def get_focus_region_candidates(att_map, top_n=50):
    att_map = att_map ** 2 
    if att_map.max() > 0: att_map = att_map / att_map.max()
    H, W = att_map.shape
    anchors = extract_anchor_points(att_map, k=30, min_distance=2)
    if not anchors: return []
    all_rois = []; all_scores = []
    for anchor in anchors:
        try:
            roi = generate_stepwise_adaptive_roi(att_map, anchor, min_side=3, max_side=18, step_size=1, expansion_threshold=0.2)
            x1, y1, x2, y2 = roi
            x1, y1 = int(max(0, x1)), int(max(0, y1)); x2, y2 = int(min(W-1, x2)), int(min(H-1, y2))
            all_rois.append([x1, y1, x2, y2])
            all_scores.append(att_map[y1:y2+1, x1:x2+1].mean())
        except: continue
    if not all_rois: return []
    keep_indices = non_maximum_suppression(all_rois, all_scores, iou_threshold=0.3)
    candidates = []
    for idx in keep_indices[:top_n]:
        x1, y1, x2, y2 = all_rois[idx]
        norm_bbox_corrected = (x1/W, y1/H, (x2+1)/W, (y2+1)/H)
        candidates.append({'bbox': norm_bbox_corrected, 'score': all_scores[idx]})
    return candidates

def verify_object_existence(subimage, object_description, model, processor, device):
    try:
        question = f"Is the {object_description} clearly visible in this image? Answer Yes or No."
        if isinstance(subimage, str):
            if "base64," in subimage: subimage = subimage.split("base64,")[1]
            image_data = base64.b64decode(subimage)
            pil_img = Image.open(io.BytesIO(image_data)).convert("RGB")
        else: pil_img = subimage
        messages = [{"role": "user", "content": [{"type": "image", "image": pil_img}, {"type": "text", "text": question}]}]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs, video_kwargs = process_vision_info(messages, return_video_kwargs=True)
        inputs = processor(text=[text], images=image_inputs, padding=True, return_tensors="pt", **video_kwargs).to(device)
        tokenizer = processor.tokenizer
        yes_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in ["Yes", "yes", "YES"] if len(tokenizer.encode(w, add_special_tokens=False))==1]
        no_ids = [tokenizer.encode(w, add_special_tokens=False)[0] for w in ["No", "no", "NO"] if len(tokenizer.encode(w, add_special_tokens=False))==1]
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[0, -1, :]
            l_yes = max([logits[i].item() for i in set(yes_ids)]) if yes_ids else -100
            l_no = max([logits[i].item() for i in set(no_ids)]) if no_ids else -100
            probs = torch.softmax(torch.tensor([l_yes, l_no]), dim=0)
            return 2 * (probs[0].item() - 0.5)
    except: return -1.0

# Get_box.py

# Get_box.py

def visualize_attention_overlay(img_path, att_map, save_path, alpha=0.6):
    # --- 1. 读取背景图 ---
    if os.path.exists(img_path): 
        img = cv2.imread(img_path)
        if img is None:
            print(f"❌ [Viz Error] cv2.imread 读取失败: {img_path}")
            return
    elif "base64," in img_path: 
        try:
            img = cv2.cvtColor(np.array(Image.open(io.BytesIO(base64.b64decode(img_path.split("base64,")[1]))).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception as e:
            print(f"❌ [Viz Error] Base64 解码失败: {e}")
            return
    else: 
        print(f"❌ [Viz Error] 图片路径不存在: {img_path}")
        return
    
    try:
        H, W = img.shape[:2]
        
        # --- 2. 数据清洗 (终极加固) ---
        # 确保 att_map 是 numpy 数组
        att_map = np.array(att_map, dtype=np.float32)
        
        # 处理 NaN 和 Inf (防止除零或溢出错误)
        att_map = np.nan_to_num(att_map, nan=0.0, posinf=1.0, neginf=0.0)
        
        # 压缩维度：确保它是 (h, w)，而不是 (h, w, 1) 或 (1, h, w)
        if att_map.ndim == 3:
            att_map = att_map.squeeze()
            
        # 再次检查维度，如果还是不对（比如是个向量），则跳过
        if att_map.ndim != 2:
            print(f"⚠️ [Viz Warning] Skip: att_map 维度异常 {att_map.shape}")
            return

        # --- 3. 归一化与缩放 ---
        min_val, max_val = att_map.min(), att_map.max()
        if max_val - min_val > 1e-8:
            att_map = (att_map - min_val) / (max_val - min_val)
        else:
            att_map = np.zeros_like(att_map) # 如果最大最小一样，说明是纯色/全零

        # --- 4. 转换数据类型 ---
        # Resize 到目标尺寸
        att_resized = cv2.resize(att_map, (W, H), interpolation=cv2.INTER_LINEAR)
        
        # 放大到 0-255 并转为 uint8
        heatmap_uint8 = (att_resized * 255).astype(np.uint8)
        
        # --- 5. OpenCV 色彩映射 ---
        # 这里的输入必须是 uint8 且单通道
        heatmap = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
        
        # --- 6. 叠加与保存 ---
        result = cv2.addWeighted(img, 1 - alpha, heatmap, alpha, 0)
        
        success = cv2.imwrite(save_path, result)
        if not success:
            print(f"❌ [Viz Error] cv2.imwrite 保存失败: {save_path}")
            
    except Exception as e:
        print(f"❌ [Viz Error] 发生未知错误: {e}")
        # 打印调试信息帮助定位
        try:
            print(f"   Debug: att_map shape={att_map.shape}, dtype={att_map.dtype}")
        except: pass

def draw_all_candidates(img_path, candidates_info, save_path):
    try:
        if os.path.exists(img_path): img = cv2.imread(img_path)
        elif "base64," in img_path: img = cv2.cvtColor(np.array(Image.open(io.BytesIO(base64.b64decode(img_path.split("base64,")[1]))).convert("RGB")), cv2.COLOR_RGB2BGR)
        else: return
        for i, item in enumerate(candidates_info):
            x1, y1, x2, y2 = item['pixel_bbox']
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 2)
            cv2.putText(img, f"{item['score']:.2f}", (x1, max(y1-5, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
        cv2.imwrite(save_path, img)
    except: pass

# --- 保留旧函数以兼容 ---
def get_inputs(messages,processor,model):
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    image_inputs, video_inputs, video_kwargs = process_vision_info(messages,return_video_kwargs=True)
    inputs = processor(text=[text], images=image_inputs, videos=video_inputs, padding=True, return_tensors="pt", **video_kwargs)
    inputs = inputs.to(model.device)
    return text,image_inputs,video_inputs,inputs,video_kwargs

def messages2out(model, processor, inputs):
    inputs = inputs.to(model.device)
    end_ques = len(inputs['input_ids'][0])
    with torch.no_grad():
        outputs = model.generate(**inputs, use_cache=True, max_new_tokens=1024, do_sample=False, return_dict_in_generate=True, output_scores=True)
    generated_ids = outputs.sequences
    generated_ids_trimmed = [out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
    output_text = processor.batch_decode(generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    confidence = 0.0
    if outputs.scores:
        probs = torch.nn.functional.softmax(outputs.scores[0], dim=-1)
        top_prob, _ = torch.max(probs, dim=-1)
        confidence = top_prob.item()
    del inputs, generated_ids, outputs
    torch.cuda.empty_cache()
    return output_text, end_ques, confidence

def messages2att(model,processor,inputs):
    inputs = inputs.to(model.device)
    end_ques = len(inputs['input_ids'][0])
    img_start, img_end = [], []
    idx2word_dicts = {}
    need_2_att_w = []
    for i in range(len(inputs['input_ids'][0])):
        words = processor.post_process_image_text_to_text(torch.tensor([inputs['input_ids'][0][i]]), skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        idx2word_dicts[inputs['input_ids'][0][i].cpu().item()] = words
        if inputs['input_ids'][0][i].cpu().item() == 151652: img_start.append(i+1)
        if inputs['input_ids'][0][i].cpu().item() == 151653: img_end.append(i)
    for i in range(len(inputs['input_ids'][0])):
        if i>img_end[-1]: need_2_att_w.append(i)
    with torch.no_grad(): out = model(**inputs, output_attentions=True,target_indices=torch.tensor(need_2_att_w)) 
    attention = [att for att in out['attentions'] if att is not None]
    del inputs,out
    torch.cuda.empty_cache()
    return attention,idx2word_dicts,img_start,img_end

def find_top_n_attended_regions(norm_att, n, threshold=0.5):
    att_map = np.array(norm_att)
    binarized_map = (att_map >= threshold)
    if not np.any(binarized_map): return [],0
    labeled_map = label(binarized_map, connectivity=2)
    regions = regionprops(labeled_map)
    scored_regions = []
    for region in regions:
        mask = (labeled_map == region.label)
        score = np.sum(att_map[mask])
        scored_regions.append({'score': score, 'bbox': region.bbox})
    sorted_regions = sorted(scored_regions, key=lambda r: r['score'], reverse=True)
    final_boxes = []
    for region in sorted_regions:
        y0, x0, y1, x1 = region['bbox']
        final_boxes.append([x0, y0, x1, y1])
    return final_boxes, len(final_boxes)

def swap_and_rebuild_dict(imgs_words_att_box):
    img_merged_boxes = {}
    for img_idx in imgs_words_att_box:
        words_att = imgs_words_att_box[img_idx]
        for word in words_att:
            if word not in img_merged_boxes: img_merged_boxes[word] = {}
            img_merged_boxes[word][img_idx] = words_att[word]
    return img_merged_boxes

def compact_and_center_with_relative_pos(imgidx, img_nums, image, normalized_bboxes, n=1):
    return None, []

def calculate_iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    interArea = max(0, x2 - x1) * max(0, y2 - y1)
    box1Area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2Area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return interArea / float(box1Area + box2Area - interArea + 1e-6)

def merge_overlapping_bboxes(bboxes, iou_threshold=0.3):
    if not bboxes: return []
    bboxes = [list(b) for b in bboxes] 
    while True:
        merged_one = False
        i = 0
        while i < len(bboxes):
            j = i + 1
            while j < len(bboxes):
                box1 = bboxes[i]; box2 = bboxes[j]
                has_intersection = not (box1[2] < box2[0] or box1[0] > box2[2] or box1[3] < box2[1] or box1[1] > box2[3])
                should_merge = False
                if has_intersection:
                    if calculate_iou(box1, box2) > iou_threshold: should_merge = True
                if should_merge:
                    new_x0 = min(box1[0], box2[0]); new_y0 = min(box1[1], box2[1])
                    new_x1 = max(box1[2], box2[2]); new_y1 = max(box1[3], box2[3])
                    bboxes[i] = [new_x0, new_y0, new_x1, new_y1]
                    bboxes.pop(j); merged_one = True; break 
                else: j += 1
            if merged_one: break 
            else: i += 1
        if not merged_one: break
    return bboxes

def from_img_and_att_get_cropbox(inputs, attention, dicts, img_url, img_start, img_end, sig, thre, target_entities=None, debug_dir=None):
    if not img_end: return {str(s): {str(t): [{}, {}, {}, [], {}, {}] for t in thre} for s in sig}
    tmp_att = [att for att in attention if att is not None]
    start_k = img_end[-1]+1; end_k = len(inputs['input_ids'][0])
    results = {}
    
    for s in sig:
        for t in thre:
            # 请确保 process 函数逻辑正确，此处假设已引入或包含
            accept_att = process(dicts, start_k, end_k, tmp_att, inputs, img_start, img_end, s)
            
            imgs_words_att_box = {}
            for img_idx in accept_att:
                accept_word_att = accept_att[img_idx]
                words_att_box = {}
                for word in accept_word_att:
                    att_map = accept_word_att[word][0]
                    boxs, _ = find_top_n_attended_regions(att_map, 100, t)
                    if boxs:
                        H, W = att_map.shape
                        words_att_box[word] = []
                        for box in boxs:
                            x0,y0,x1,y1 = box
                            words_att_box[word].append((x0/W, y0/H, (x1)/W, (y1)/H))
                imgs_words_att_box[img_idx] = words_att_box

            img_merged_boxes = swap_and_rebuild_dict(imgs_words_att_box)
            
            words_lines = {}
            get_words = ""
            for i in range(start_k, end_k):
                token_idx = inputs['input_ids'][0][i].item()
                if token_idx < 151643: get_words += dicts[token_idx]
                for word in img_merged_boxes:
                    if i == word+1: words_lines[word] = get_words; get_words = ''
            for word in img_merged_boxes:
                if i == word: words_lines[word] = get_words; get_words = ''
            words_lines[-1] = get_words

            crop_list = {}
            bounding_boxes = {}
            highlight_components = [] 
            
            for word in img_merged_boxes:
                if not word in crop_list: crop_list[word] = {}
                for imgidx in img_merged_boxes[word]:
                    if not imgidx in bounding_boxes: bounding_boxes[imgidx] = []
                    for boxid in range(len(img_merged_boxes[word][imgidx])):
                        bounding_boxes[imgidx].append(img_merged_boxes[word][imgidx][boxid])
            
            for imgidx in bounding_boxes:
                if imgidx < len(img_url):
                    curr_url = img_url[imgidx]
                    try:
                        if os.path.exists(curr_url): pil_ori = Image.open(curr_url).convert("RGB")
                        elif "base64," in curr_url: pil_ori = Image.open(io.BytesIO(base64.b64decode(curr_url.split(",")[1]))).convert("RGB")
                        else: pil_ori = Image.open(io.BytesIO(base64.b64decode(curr_url))).convert("RGB")
                        W, H = pil_ori.size
                        
                        pixel_boxes = []
                        for b in bounding_boxes[imgidx]:
                            x1, y1, x2, y2 = b
                            pixel_boxes.append([int(x1*W), int(y1*H), int(x2*W), int(y2*H)])
                        
                        distinct_boxes = non_maximum_suppression(pixel_boxes, [1.0]*len(pixel_boxes), iou_threshold=0.3)
                        
                        for idx_k, box_idx in enumerate(distinct_boxes):
                            x1, y1, x2, y2 = pixel_boxes[box_idx]
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(W, x2), min(H, y2)
                            if x2 <= x1 or y2 <= y1: continue

                            crop = pil_ori.crop((x1, y1, x2, y2))
                            # Initial Region label=None，避免文字干扰
                            highlight_components.append({
                                'pil_img': crop,
                                'bbox': [x1, y1, x2, y2],
                                'label': None 
                            })
                    except Exception as e:
                        print(f"Error extract initial: {e}")

            clean_raw_maps = {}
            if target_entities:
                all_tokens = []
                for k in range(start_k, end_k):
                    token_id = inputs['input_ids'][0][k].item()
                    token_text = dicts.get(token_id, "").lower().strip()
                    if token_text: all_tokens.append({'id': k, 'text': token_text})
                
                for entity in target_entities:
                    entity_clean = entity.lower().strip()
                    matched_maps = []
                    target_img_idx = 0 
                    for token_info in all_tokens:
                        t_text = token_info['text']
                        t_id = token_info['id']
                        if t_text in entity_clean and len(t_text) > 1:
                            for img_idx in accept_att:
                                if t_id in accept_att[img_idx]:
                                    raw_map = accept_att[img_idx][t_id][0]
                                    matched_maps.append(raw_map)
                                    target_img_idx = img_idx
                    if matched_maps:
                        combined_map = np.max(np.array(matched_maps), axis=0)
                        clean_raw_maps[entity_clean] = (target_img_idx, combined_map)
            else:
                for word_idx, entity_text in words_lines.items():
                    if word_idx == -1: continue
                    clean_text = entity_text.strip().lower()
                    if not clean_text: continue
                    for img_idx in accept_att:
                        if word_idx in accept_att[img_idx]:
                            raw_map = accept_att[img_idx][word_idx][0]
                            clean_raw_maps[clean_text] = (img_idx, raw_map)
                            break

            if not str(s) in results: results[str(s)] = {}
            results[str(s)][str(t)] = [
                img_merged_boxes, 
                crop_list, 
                words_lines, 
                highlight_components, 
                bounding_boxes, 
                clean_raw_maps,
                accept_att  # <--- 🔥 新增这一项 (索引为 6)
            ]
            
    return results