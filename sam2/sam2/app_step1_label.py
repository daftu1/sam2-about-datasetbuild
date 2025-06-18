import streamlit as st
import cv2
import os
import numpy as np
from PIL import Image
from ultralytics import SAM
import torch
import uuid

# 初始化SAM2模型（第一次加载可能较慢）
@st.cache_resource
def load_sam_model():
    return SAM("sam2.1_b.pt")  # 或你自己的微调sam模型路径

sam_model = load_sam_model()

# 设置缓存目录
FRAME_DIR = "sample_frames"
MASK_DIR = "saved_masks"
os.makedirs(FRAME_DIR, exist_ok=True)
os.makedirs(MASK_DIR, exist_ok=True)

st.title("🎯 Step 1：采样帧 + SAM2 掩膜打点标注")

uploaded_video = st.file_uploader("上传视频", type=["mp4"])
sampling_interval = st.number_input("每隔N帧抽取1帧（建议为10~30）", value=15, min_value=1, max_value=100)

if uploaded_video:
    video_path = f"temp_video.mp4"
    with open(video_path, "wb") as f:
        f.write(uploaded_video.read())

    # 抽帧
    st.info("正在抽取帧...")
    cap = cv2.VideoCapture(video_path)
    frame_idx, saved_idx = 0, 0
    frames = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % sampling_interval == 0:
            img_path = os.path.join(FRAME_DIR, f"frame_{saved_idx:04d}.jpg")
            cv2.imwrite(img_path, frame)
            frames.append(img_path)
            saved_idx += 1
        frame_idx += 1

    cap.release()
    st.success(f"共提取 {len(frames)} 帧，准备进入标注流程。")

    frame_idx = st.slider("选择帧编号", 0, len(frames) - 1)
    selected_frame = cv2.imread(frames[frame_idx])
    rgb_img = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)
    st.image(rgb_img, caption=f"Frame {frame_idx}", channels="RGB")

    st.subheader("📌 掩膜打点")
    x_click = st.number_input("点击点 X 坐标", min_value=0, max_value=selected_frame.shape[1] - 1)
    y_click = st.number_input("点击点 Y 坐标", min_value=0, max_value=selected_frame.shape[0] - 1)
    label = st.text_input("请输入该掩膜的类别标签（如：锅、辣椒）")

    if st.button("生成掩膜"):
        if label.strip() == "":
            st.warning("请填写类别标签")
        else:
            prompts = {"points": [[x_click, y_click]], "labels": [1]}  # 正样本点
            result = sam_model(selected_frame, **prompts)[0]
            mask = result.masks.data[0].cpu().numpy().astype(np.uint8) * 255

            # 可视化
            color_mask = cv2.applyColorMap(mask, cv2.COLORMAP_JET)
            combined = cv2.addWeighted(rgb_img, 0.6, color_mask, 0.4, 0)
            st.image(combined, caption=f"掩膜预览：{label}", channels="RGB")

            # 保存
            uid = uuid.uuid4().hex[:8]
            cv2.imwrite(os.path.join(MASK_DIR, f"{label}_{frame_idx}_{uid}.png"), mask)
            with open(os.path.join(MASK_DIR, f"{label}_{frame_idx}_{uid}.txt"), "w") as f:
                f.write(f"{label},{x_click},{y_click},{frames[frame_idx]}")
            st.success("已保存该掩膜与标注信息！")
