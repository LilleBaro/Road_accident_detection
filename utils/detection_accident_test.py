import tempfile
import os
import time
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
import streamlit as st

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

