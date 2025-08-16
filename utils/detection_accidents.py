import streamlit as st
import cv2
import numpy as np
import tempfile
import os
import time
from PIL import Image, ImageEnhance, ImageFilter
from typing import Optional, Dict, Any

st.set_page_config(
    page_title="Détection d'Accident",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ---------- CSS ----------
st.markdown(
    """
<style>
    .main-header {
        background: linear-gradient(90deg, #1f4e79, #2d5a87);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        color: white;
    }
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border: 2px dashed #dee2e6;
        margin-bottom: 2rem;
    }
    .controls-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .results-section {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    .stButton > button {
        background: linear-gradient(90deg, #1f4e79, #2d5a87);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 1rem;
    }
    .stButton > button:hover {
        background: linear-gradient(90deg, #2d5a87, #1f4e79);
    }
</style>
""",
    unsafe_allow_html=True,
)

# ---------- Header ----------
st.markdown(
    """
<div class="main-header">
    <h1>🚨 Détection d'Accidents</h1>
    <p>Analysez des images et des vidéos avec un modèle YOLOv12 finetuné pour détecter les accidents.</p>
</div>
""",
    unsafe_allow_html=True,
)

# ---------- Helpers ----------
def resize_video_for_analysis(video_path: str, target_width: int = 640) -> str:
    """Redimensionne la vidéo pour l'analyse et renvoie le chemin de sortie."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError("Impossible d'ouvrir la vidéo pour redimensionnement.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    aspect_ratio = height / width if width != 0 else 9/16
    new_h = int(target_width * aspect_ratio)
    out_path = tempfile.mktemp(suffix=f"_resized_{target_width}.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (target_width, new_h))
    frame_count = 0
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (target_width, new_h))
            out.write(resized)
            frame_count += 1
            if frame_count % 200 == 0:
                st.write(f"Redimensionnement : {frame_count} frames traitées")
    finally:
        cap.release()
        out.release()
    st.success(f"✅ Vidéo redimensionnée: {width}x{height} → {target_width}x{new_h}")
    return out_path

def safe_extract_box(box) -> Optional[Dict[str, Any]]:
    """
    Récupère x1,y1,x2,y2,cls,conf depuis un objet 'box' d'ultralytics
    Retourne None si extraction échoue.
    """
    try:
        # box.xyxy peut être un tensor ou un numpy-like
        coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy[0])
        x1, y1, x2, y2 = map(int, coords[:4])
        # confidence
        conf = float(box.conf) if hasattr(box, "conf") else float(box[4])
        cls_idx = int(box.cls) if hasattr(box, "cls") else int(box[5])
        return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "cls": cls_idx}
    except Exception:
        return None

def process_video_with_annotations(video_path: str, model, conf_threshold: float, max_detections: int, skip_frames: int = 1):
    """
    Parcours la vidéo, effectue la détection et écrit une vidéo annotée.
    Retourne un dict avec métadonnées et le chemin du fichier de sortie.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        st.error("Impossible d'ouvrir la vidéo.")
        return None

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0

    out_path = tempfile.mktemp(suffix="_annotated.mp4")
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(out_path, fourcc, fps, (width, height))

    total_detections = 0
    detection_classes = {}
    frame_detections = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    frame_count = 0
    processed_frames = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % skip_frames == 0:
                # model.predict : renvoie Results
                results = model.predict(frame, conf=conf_threshold, max_det=max_detections, verbose=False)
                annotated_frame = frame.copy()
                frame_info = {"frame_number": frame_count, "detections": []}

                if len(results) > 0 and getattr(results[0], "boxes", None) is not None:
                    for box in results[0].boxes:
                        info = safe_extract_box(box)
                        if info is None:
                            continue
                        cls_idx = info["cls"]
                        class_name = model.names.get(cls_idx, str(cls_idx)) if isinstance(model.names, dict) else model.names[cls_idx]
                        conf = info["conf"]
                        total_detections += 1
                        detection_classes[class_name] = detection_classes.get(class_name, 0) + 1
                        frame_info["detections"].append({"class": class_name, "confidence": conf})

                        x1, y1, x2, y2 = info["x1"], info["y1"], info["x2"], info["y2"]

                        # Color / label
                        if class_name.lower() == "accident":
                            color = (0, 0, 255)
                            label = f"ACCIDENT: {conf:.2f}"
                        else:
                            color = (0, 255, 0)
                            label = f"{class_name}: {conf:.2f}"

                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                        label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
                        y_text = max(y1 - 10, label_size[1] + 5)
                        cv2.rectangle(annotated_frame, (x1, y_text - label_size[1] - 5), (x1 + label_size[0] + 6, y_text), color, -1)
                        cv2.putText(annotated_frame, label, (x1 + 3, y_text - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                else:
                    cv2.putText(annotated_frame, "Aucune detection", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

                frame_detections.append(frame_info)
                out.write(annotated_frame)
                processed_frames += 1
            else:
                out.write(frame)

            frame_count += 1
            if total_frames > 0:
                progress_bar.progress(min(frame_count / total_frames, 1.0))
            status_text.text(f"Traitement: Frame {frame_count} / {total_frames} ({processed_frames} traitées)")
    finally:
        cap.release()
        out.release()
        progress_bar.empty()
        status_text.empty()

    return {
        "output_path": out_path,
        "total_detections": total_detections,
        "detection_classes": detection_classes,
        "frame_detections": frame_detections,
        "total_frames": total_frames,
        "processed_frames": processed_frames,
    }

# ---------- Lazy model loader ----------
@st.cache_resource
def load_model(model_path: str = "models/detection_accident.pt"):
    """
    Charge le modèle de façon lazy. L'import d'ultralytics est fait ici
    pour éviter d'importer les binaires natifs au startup de Streamlit.
    """
    try:
        from ultralytics import YOLO  # import local
    except Exception as e:
        st.error(f"Impossible d'importer ultralytics: {e}")
        return None

    try:
        model = YOLO(model_path)
        return model
    except Exception as e:
        st.error(f"Erreur lors du chargement du modèle: {e}")
        return None

# ---------- UI & logique ----------
# Upload
st.markdown('<div class="upload-section"><h3>📁 Upload de Fichiers</h3><p>Uploadez image ou vidéo</p></div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Choisissez un fichier (image/vidéo)", type=["jpg", "jpeg", "png", "bmp", "mp4", "avi", "mov", "mkv"])

# Controls
col1, col2 = st.columns([1, 1])
with col1:
    st.subheader("🎛️ Contrôles")
    confidence = st.slider("Seuil de confiance", 0.1, 1.0, 0.5, 0.05)
    max_detections = st.number_input("Nombre max de détections", 1, 200, 50)
    show_labels = st.checkbox("Afficher les étiquettes", value=True)
    show_scores = st.checkbox("Afficher les scores", value=True)
    show_fps = st.checkbox("Afficher FPS", value=True)
    show_frame_details = st.checkbox("Afficher détails frame par frame (vidéos)", value=False)

    # Options vidéo
    skip_frames = 1
    optimize_resolution = False
    target_width = 640
    confidence_video = confidence
    if uploaded_file and uploaded_file.type.startswith("video"):
        skip_frames = st.slider("Traiter 1 frame sur N", 1, 10, 3)
        optimize_resolution = st.checkbox("Réduire la résolution pour l'analyse", value=True)
        target_width = st.selectbox("Résolution cible", [640, 720, 1280], index=0)
        confidence_video = st.slider("Seuil de confiance (vidéo)", 0.1, 1.0, 0.4, 0.05)

    detect_button = st.button("🚀 Lancer la Détection") if uploaded_file else False

with col2:
    st.subheader("📊 Résultats")

    if uploaded_file is None:
        st.info("👆 Uploadez un fichier pour commencer l'analyse")

    elif uploaded_file.type.startswith("image"):
        original_image = Image.open(uploaded_file).convert("RGB")
        st.image(original_image, caption="Image uploadée", use_container_width=True)

        # Augmentations
        brightness = st.slider("Luminosité", 0.5, 2.0, 1.0, 0.1)
        contrast = st.slider("Contraste", 0.5, 2.0, 1.0, 0.1)
        saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.1)
        sharpness = st.slider("Netteté", 0.0, 2.0, 1.0, 0.1)
        blur = st.slider("Flou", 0, 10, 0, 1)
        noise = st.slider("Bruit", 0, 50, 0, 5)

        augmented = original_image
        if detect_button:
            # Apply augmentations then detect
            try:
                # conversions
                enhancer = ImageEnhance.Brightness(augmented)
                augmented = enhancer.enhance(brightness)
                enhancer = ImageEnhance.Contrast(augmented)
                augmented = enhancer.enhance(contrast)
                enhancer = ImageEnhance.Color(augmented)
                augmented = enhancer.enhance(saturation)
                enhancer = ImageEnhance.Sharpness(augmented)
                augmented = enhancer.enhance(sharpness)
                if blur > 0:
                    augmented = augmented.filter(ImageFilter.GaussianBlur(blur))
                if noise > 0:
                    arr = np.array(augmented).astype(np.float32)
                    arr += np.random.normal(0, noise, arr.shape)
                    arr = np.clip(arr, 0, 255).astype(np.uint8)
                    augmented = Image.fromarray(arr)

                model = load_model()
                if model is None:
                    st.error("Modèle non chargé. Vérifie l'installation.")
                else:
                    st.info("Détection en cours...")
                    start_time = time.time()
                    img_arr = np.array(augmented)
                    results = model.predict(img_arr, conf=confidence, max_det=max_detections, verbose=False)
                    duration = time.time() - start_time
                    fps = 1.0 / duration if duration > 0 else 0.0
                    # affichage
                    if show_fps:
                        st.metric("FPS", f"{fps:.2f}")
                    if len(results) and getattr(results[0], "boxes", None) is not None and len(results[0].boxes) > 0:
                        st.success("Détections trouvées")
                        annotated = results[0].plot()
                        st.image(annotated, caption="Image annotée", use_container_width=True)
                        # listing des classes
                        total = 0
                        classes_count = {}
                        for box in results[0].boxes:
                            info = safe_extract_box(box)
                            if info is None:
                                continue
                            cls_name = model.names.get(info["cls"], str(info["cls"])) if isinstance(model.names, dict) else model.names[info["cls"]]
                            total += 1
                            classes_count[cls_name] = classes_count.get(cls_name, 0) + 1
                            if show_scores:
                                st.write(f"{cls_name} — conf: {info['conf']:.3f}")
                            else:
                                st.write(cls_name)
                        st.metric("Total détections", total)
                        for k, v in classes_count.items():
                            st.write(f"- {k}: {v}")
                    else:
                        st.warning("Aucune détection trouvée")
            except Exception as e:
                st.error(f"Erreur pendant la détection: {e}")

    elif uploaded_file.type.startswith("video"):
        st.video(uploaded_file)
        if detect_button:
            tmp_video = None
            resized_video = None
            try:
                # save uploaded file to temp
                with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as f:
                    f.write(uploaded_file.read())
                    tmp_video = f.name

                video_for_analysis = tmp_video
                if optimize_resolution:
                    st.info("🔄 Redimensionnement...")
                    resized_video = resize_video_for_analysis(tmp_video, target_width=int(target_width))
                    video_for_analysis = resized_video

                conf_threshold = confidence_video
                model = load_model()
                if model is None:
                    st.error("Modèle non chargé.")
                else:
                    st.info("🎬 Génération vidéo annotée...")
                    result = process_video_with_annotations(video_for_analysis, model, conf_threshold, max_detections, skip_frames)
                    if result:
                        st.success("Traitement terminé ✅")
                        # lecture et affichage
                        with open(result["output_path"], "rb") as vf:
                            video_bytes = vf.read()
                        st.video(video_bytes, format="video/mp4", start_time=0)
                        st.download_button("⬇️ Télécharger la vidéo annotée", data=video_bytes, file_name=f"annotated_{os.path.basename(uploaded_file.name)}", mime="video/mp4")
                        # Stats
                        st.metric("Total détections", result["total_detections"])
                        st.metric("Frames traitées", result["processed_frames"])
                        # show class split
                        for cls, cnt in result["detection_classes"].items():
                            emoji = "🚨" if cls.lower()=="accident" else "✅"
                            st.write(f"{emoji} {cls}: {cnt}")
                    else:
                        st.error("Erreur pendant le traitement vidéo.")
            except Exception as e:
                st.error(f"Erreur lors de l'analyse: {e}")
            finally:
                # cleanup temps
                for p in [tmp_video, resized_video]:
                    try:
                        if p and os.path.exists(p):
                            os.unlink(p)
                    except Exception:
                        pass

# Footer
st.markdown("---")
st.markdown('<div style="text-align:center;color:#666;">🚨 Application de Détection d\'Accidents</div>', unsafe_allow_html=True)
