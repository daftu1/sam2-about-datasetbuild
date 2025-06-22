import os
os.environ["TORCHINDUCTOR_DISABLE"] = "1"
os.environ["TORCH_COMPILE_DISABLE"] = "1"

import streamlit as st
st.set_page_config(layout="wide")

import os, uuid, cv2, shutil, numpy as np, torch
from PIL import Image
from moviepy import VideoFileClip
from streamlit_image_coordinates import streamlit_image_coordinates
from torchvision.ops import masks_to_boxes
from sam2.build_sam import build_sam2_video_predictor
import contextlib

@st.cache_resource
def load_sam2_model():
    checkpoint = os.path.join(os.path.expanduser("~"), "sam2", "checkpoints", "sam2.1_hiera_base_plus.pt")
    model_cfg = os.path.join("configs", "sam2.1", "sam2.1_hiera_b+.yaml")
    # 若可用则使用GPU，否则退回CPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return build_sam2_video_predictor(model_cfg, checkpoint, device=device, vos_optimized=True)

sam2_model = load_sam2_model()

st.title("🎬 Pre：交互式视频标注 & 自动掩码传播（SAM2集成）")

VIDEO_DIR = "video_segments"
os.makedirs(VIDEO_DIR, exist_ok=True)

# 上传视频
uploaded_video = st.file_uploader("📁 上传完整原始视频", type=["mp4"])
if uploaded_video:
    original_path = "temp_uploaded.mp4"
    with open(original_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(original_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = frame_count // fps
    st.video(original_path)

    st.subheader("✂️ 视频裁剪")
    start_time = st.slider("起始时间（秒）", 0, duration - 1, 0)
    end_time = st.slider("结束时间（秒）", start_time + 1, duration, start_time + 5)

    if st.button("裁剪并保存片段"):
        clip = VideoFileClip(original_path).subclipped(start_time, end_time)
        session_id = uuid.uuid4().hex[:8]
        segment_path = os.path.join(VIDEO_DIR, f"{session_id}.mp4")
        clip.write_videofile(segment_path, codec="libx264")
        st.success(f"✅ 视频裁剪完成: {segment_path}")
        st.session_state["segment_path"] = segment_path
        st.session_state["session_id"] = session_id
        cap.release()
        shutil.move(original_path, f"{original_path}.bak")

# 视频拆帧 + 加载状态
session_id = st.session_state.get("session_id", None)
segment_path = st.session_state.get("segment_path", None)

if session_id and segment_path:
    FRAME_DIR = f"frame_cache_{session_id}"
    os.makedirs(FRAME_DIR, exist_ok=True)

    if not os.listdir(FRAME_DIR):
        cap = cv2.VideoCapture(segment_path)
        frame_idx = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_path = os.path.join(FRAME_DIR, f"{frame_idx:04d}.jpg")
            cv2.imwrite(frame_path, frame)
            frame_idx += 1
        cap.release()
        st.success(f"✅ 共提取 {frame_idx} 帧至缓存目录 {FRAME_DIR}")

    frame_files = sorted(os.listdir(FRAME_DIR))
    frame_index = st.session_state.get("frame_index", 0)
    current_frame_path = os.path.join(FRAME_DIR, frame_files[frame_index])
    current_img = Image.open(current_frame_path)

    frame_np = np.array(current_img.convert("RGB"))
    if "overlay_map" in st.session_state and frame_index in st.session_state["overlay_map"]:
        preview_img = st.session_state["overlay_map"][frame_index]
    else:
        preview_img = frame_np.copy()

    points = st.session_state.get("points", {}).get(frame_index, [])
    for x, y, l in points:
        color = (0, 255, 0) if l == 1 else (0, 0, 255)
        cv2.circle(preview_img, (int(x), int(y)), 5, color, -1)

    click_mode = st.radio("点击模式", ["保留点", "去除点"], index=0, horizontal=True)
    labelP = 1 if click_mode == "保留点" else 0

    # 左图右控
    col1, col2 = st.columns([3, 1])

    with col1:
        click = streamlit_image_coordinates(preview_img, key=f"frame_{frame_index}")

    with col2:
        st.markdown("### 🎞️ 当前帧控制")
        frame_index = st.slider("帧位置", 0, len(frame_files) - 1, value=frame_index, key="frame_index")
        st.write(f"当前帧编号：**{frame_index}**")

        # 标签管理
        st.subheader("🏷️ 标签输入与确认")
        if "label_history" not in st.session_state:
            st.session_state["label_history"] = []
        label_input = st.text_input("✏️ 输入标签名", value="", placeholder="e.g. tomato")
        if label_input:
            suggestions = [l for l in st.session_state["label_history"] if l.startswith(label_input.lower())]
            if suggestions:
                st.markdown("🔍 自动补全建议：" + ", ".join(suggestions[:5]))

        if st.button("✅ 确定标签"):
            label = label_input.strip().lower()
            if label:
                if label not in st.session_state["label_history"]:
                    st.session_state["label_history"].append(label)
                st.session_state["current_label"] = label
                st.success(f"✅ 当前使用标签：`{label}`")
            else:
                st.warning("⚠️ 标签不能为空")

        label = st.session_state.get("current_label", None)

        if click:
            if "points" not in st.session_state:
                st.session_state["points"] = {}
            if frame_index not in st.session_state["points"]:
                st.session_state["points"][frame_index] = []
            st.session_state["points"][frame_index].append((click["x"], click["y"], labelP))
            st.rerun()

        if st.button("🧹 清除当前帧所有点"):
            if "points" in st.session_state and frame_index in st.session_state["points"]:
                st.session_state["points"][frame_index] = []
                if "overlay_map" in st.session_state and frame_index in st.session_state["overlay_map"]:
                    del st.session_state["overlay_map"][frame_index]
                st.success("✅ 当前帧点清除完毕")
                st.experimental_rerun()

        if st.button("🧼 清除所有帧的点"):
            st.session_state["points"] = {}
            if "overlay_map" in st.session_state:
                st.session_state["overlay_map"] = {}
            st.success("✅ 所有帧点清除完毕")
            st.experimental_rerun()

        if st.button("👁️ 预览当前帧标注"):
            if not points:
                st.warning("⚠️ 当前帧无点击点")
            else:
                pts = []
                lbls = []
                for x, y, l in points:
                    pts.append([x, y])
                    lbls.append(l)
                with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    inference_state = sam2_model.init_state(segment_path)
                    frame_idx, obj_ids, masks = sam2_model.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_index,
                        obj_id=0,
                        points=pts,
                        labels=lbls,
                        clear_old_points=True,
                        normalize_coords=False
                    )
                    mask_tensor = (masks[0] > 0.0)
                    if mask_tensor.any():
                        box = masks_to_boxes(mask_tensor)[0].int().cpu().tolist()
                    else:
                        st.warning("⚠️ 生成的掩码为空，无法计算外接框")
                        box = [0, 0, 0, 0]
                    mask = mask_tensor[0].byte().cpu().numpy()
                    x1, y1, x2, y2 = box
                    overlay = frame_np.copy()
                    overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([128, 128, 255]) * 0.5).astype(np.uint8)
                    for x, y, l in points:
                        cv2.circle(overlay, (int(x), int(y)), 5, (0,255,0) if l==1 else (0,0,255), -1)
                    cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
                    st.image(overlay, caption=f"标签: {label} | 点数: {len(points)} | BBox: {box}")
                    if label is not None and label in st.session_state["label_history"]:
                        label_id = st.session_state["label_history"].index(label)
                        save_dir = f"yolo_labels_{session_id}"
                        os.makedirs(save_dir, exist_ok=True)
                        h, w = mask.shape
                        yolo_line = f"{label_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
                        label_file = os.path.join(save_dir, frame_files[frame_index].replace(".jpg", ".txt"))
                        with open(label_file, "w") as f:
                            f.write(yolo_line)
                        st.success(f"✅ 单帧标签保存成功: {label_file}")

        if st.button("⚡ 自动掩码传播"):
            ref_points = st.session_state["points"].get(frame_index, [])
            if not ref_points or not label:
                st.warning("⚠️ 首帧未打点或标签未设置")
            else:
                save_dir = f"yolo_labels_{session_id}"
                os.makedirs(save_dir, exist_ok=True)
                label_id = st.session_state["label_history"].index(label)
                st.session_state["overlay_map"] = {}
                # 根据环境自动选择 CUDA 或 CPU
                with torch.autocast("cuda" if torch.cuda.is_available() else "cpu"):
                    inference_state = sam2_model.init_state(segment_path)
                    pts = [[p[0], p[1]] for p in ref_points]
                    lbls = [p[2] for p in ref_points]
                    sam2_model.add_new_points_or_box(
                        inference_state=inference_state,
                        frame_idx=frame_index,
                        obj_id=0,
                        points=pts,
                        labels=lbls,
                        clear_old_points=True,
                        normalize_coords=False
                    )
                    # 批量传播
                    video_segments = {}
                    for out_frame_idx, out_obj_ids, out_mask_logits in sam2_model.propagate_in_video(inference_state):
                        video_segments[out_frame_idx] = {
                            out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                            for i, out_obj_id in enumerate(out_obj_ids)
                        }
                    # 遍历每帧，保存YOLO标签和预览
                    for i, frame_file in enumerate(frame_files):
                        mask = video_segments.get(i, {}).get(0, None)
                        if mask is None:
                            continue
                        device = 'cuda' if torch.cuda.is_available() else 'cpu'
                        mask_tensor = torch.from_numpy(mask[None]).to(device)
                        if mask_tensor.any():
                            box = masks_to_boxes(mask_tensor)[0].int().cpu().tolist()
                        else:
                            st.warning(f"⚠️ 第{i}帧掩码为空，跳过外接框计算")
                            continue
                        x1, y1, x2, y2 = box
                        h, w = mask.shape
                        yolo_line = f"{label_id} {(x1+x2)/2/w:.6f} {(y1+y2)/2/h:.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}\n"
                        label_file = os.path.join(save_dir, frame_file.replace(".jpg", ".txt"))
                        with open(label_file, "w") as f:
                            f.write(yolo_line)
                        img_path = os.path.join(FRAME_DIR, frame_file)
                        img = np.array(Image.open(img_path).convert("RGB"))
                        overlay = img.copy()
                        overlay[mask == 1] = (overlay[mask == 1] * 0.5 + np.array([128,128,255]) * 0.5).astype(np.uint8)
                        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0,255,0), 2)
                        st.session_state["overlay_map"][i] = overlay
                st.success("✅ 所有帧标签与图像已自动生成，可逐帧预览") 