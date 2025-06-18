
import streamlit as st
st.set_page_config(layout="wide")  # 必须为首行

import os, uuid, cv2, shutil, numpy as np, torch
from PIL import Image
from moviepy import VideoFileClip
from ultralytics import SAM
from streamlit_image_coordinates import streamlit_image_coordinates
from sam2.build_sam import build_sam2_video_predictor
from torchvision.ops import masks_to_boxes

device = "cuda" if torch.cuda.is_available() else "cpu"

sam2_checkpoint = "../checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"


predictor = build_sam2_video_predictor(model_cfg, sam2_checkpoint, device=device)

inference_state = None

st.title("🎬 Step 2：视频打点 & 多帧掩码生成")

VIDEO_DIR = "video_segments"
os.makedirs(VIDEO_DIR, exist_ok=True)

# 上传原始视频
uploaded_video = st.file_uploader("📁 上传完整视频", type=["mp4"])
if uploaded_video:
    original_path = "temp_uploaded.mp4"
    with open(original_path, "wb") as f:
        f.write(uploaded_video.read())

    cap = cv2.VideoCapture(original_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    duration = frame_count // fps
    st.video(original_path)

    start_time = st.slider("起始时间 (秒)", 0, duration - 1, 0)
    end_time = st.slider("结束时间 (秒)", start_time + 1, duration, start_time + 5)

    if st.button("✂️ 裁剪视频段"):
        clip = VideoFileClip(original_path).subclipped(start_time, end_time)
        session_id = uuid.uuid4().hex[:8]
        segment_path = os.path.join(VIDEO_DIR, f"{session_id}.mp4")
        clip.write_videofile(segment_path, codec="libx264")
        st.success(f"✅ 裁剪完成: {segment_path}")
        st.session_state["segment_path"] = segment_path
        st.session_state["session_id"] = session_id
        shutil.move(original_path, f"{original_path}.bak")

# 视频掩码标注
if "segment_path" in st.session_state:
    segment_path = st.session_state["segment_path"]
    cap = cv2.VideoCapture(segment_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    st.video(segment_path)

    st.markdown("## 🧠 打点与掩码传播")

    label = st.text_input("目标标签名", value="pot")
    frame_idx = st.slider("选择标注帧", 0, total_frames - 1, 0)

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, selected_frame = cap.read()
    if not ret:
        st.error("无法读取帧")
        st.stop()

    rgb_frame = cv2.cvtColor(selected_frame, cv2.COLOR_BGR2RGB)

    if "click_points" not in st.session_state:
        st.session_state.click_points = []
        st.session_state.click_labels = []
        st.session_state.ann_obj_id = 1

    mode = st.radio("点击模式", ["添加点（正）", "去除点（负）"])
    is_positive = 1 if "正" in mode else 0

    def draw_points(image, points, labels):
        vis = image.copy()
        for (x, y), label in zip(points, labels):
            color = (0, 255, 0) if label == 1 else (255, 0, 0)
            cv2.circle(vis, (x, y), 6, color, -1)
        return vis

    annotated = draw_points(rgb_frame, st.session_state.click_points, st.session_state.click_labels)
    result = streamlit_image_coordinates(Image.fromarray(annotated), key=f"click_{frame_idx}")

    if result:
        st.session_state.click_points.append((result["x"], result["y"]))
        st.session_state.click_labels.append(is_positive)
        st.rerun()


    col1, col2 = st.columns(2)
    if col1.button("🧹 清空点击点"):
        st.session_state.click_points = []
        st.session_state.click_labels = []
        st.experimental_rerun()

    if col2.button("🚀 运行掩码传播"):
        if not st.session_state.click_points:
            st.warning("请先打点")
        else:
            points = np.array(st.session_state.click_points, dtype=np.float32)
            labels_arr = np.array(st.session_state.click_labels, dtype=np.int32)
            ann_obj_id = st.session_state.ann_obj_id

            if inference_state is None:
                inference_state = predictor.get_inference_state(video_path=segment_path)

            _, out_obj_ids, _ = predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=frame_idx,
                obj_id=ann_obj_id,
                points=points,
                labels=labels_arr,
            )

            video_segments = {}
            for idx, out_ids, out_logits in predictor.propagate_in_video(inference_state):
                video_segments[idx] = {
                    out_ids[i]: (out_logits[i] > 0).cpu().numpy()
                    for i in range(len(out_ids))
                }

            st.success("✅ 掩码传播完成")

            export_dir = f"mask_propagation_output/{st.session_state['session_id']}_{label}"
            os.makedirs(export_dir, exist_ok=True)
            cap = cv2.VideoCapture(segment_path)
            for idx in sorted(video_segments.keys()):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, f = cap.read()
                if not ret:
                    continue
                cv2.imwrite(f"{export_dir}/img_{idx:04d}.jpg", f)

                for obj_id, mask in video_segments[idx].items():
                    mask_path = f"{export_dir}/mask_{idx:04d}_{obj_id}.png"
                    cv2.imwrite(mask_path, mask * 255)

                    boxes = masks_to_boxes(torch.tensor(mask[None])).numpy()
                    with open(f"{export_dir}/img_{idx:04d}.txt", "w") as ftxt:
                        for box in boxes:
                            cx = (box[0] + box[2]) / 2 / mask.shape[1]
                            cy = (box[1] + box[3]) / 2 / mask.shape[0]
                            w = (box[2] - box[0]) / mask.shape[1]
                            h = (box[3] - box[1]) / mask.shape[0]
                            ftxt.write(f"0 {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")

            st.success(f"所有掩码与标签已导出至：{export_dir}")
