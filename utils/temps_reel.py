# utils/temps_reel.py
import streamlit as st
import time
import io
import numpy as np
from PIL import Image

# Note : on évite d'importer streamlit_webrtc, av, ultralytics au module level
# pour réduire la surface d'import qui pourrait déclencher des libs natives incompatibles.

def run_temps_reel():
    st.markdown(
        """
        <div style="background:#191970; padding:18px; border-radius:10px; color:white;">
            <h1 style="margin:0">📷 Détection d'objets en temps réel</h1>
            <p style="margin:0.25rem 0 0 0; color:#F0F2D6;">
                Active ta caméra et observe les détections YOLOv12 (modèle COCO). Ajuste le seuil et capture des snapshots.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.write("")

    # Layout : left = controls, right = video + info
    col_control, col_video = st.columns([1, 2])

    with col_control:
        st.subheader("Contrôles")
        model_path = st.text_input("Chemin du modèle YOLO (general)", "models/yolo12s.pt")
        conf = st.slider("Seuil de confiance (conf)", 0.05, 0.99, 0.30, 0.01)
        max_boxes = st.number_input("Nombre max de boxes affichées", min_value=1, max_value=200, value=100, step=1)
        show_labels = st.checkbox("Afficher étiquettes & scores", value=True)
        show_fps = st.checkbox("Afficher FPS en direct", value=True)
        btn_reload = st.button("🔄 Recharger le modèle (lazy)")
        activate_cam = st.checkbox("Activer la caméra", value=True)

    with col_video:
        st.subheader("Flux vidéo")
        info_cols = st.columns([1, 1, 1])

        fps_text = info_cols[0].empty()
        snapshot_col = info_cols[1]
        status_col = info_cols[2]

        # Si la caméra n'est pas activée, on arrête là.
        if not activate_cam:
            st.warning("Caméra désactivée — coche 'Activer la caméra' pour lancer le flux.")
            return

        # Lazy imports — on importe ces modules seulement quand on veut vraiment la caméra
        try:
            import av
            from streamlit_webrtc import webrtc_streamer, VideoTransformerBase, WebRtcMode
            from ultralytics import YOLO
        except Exception as e:
            st.error(f"Impossible d'importer une dépendance nécessaire (av / streamlit_webrtc / ultralytics). Erreur : {e}")
            st.caption("Vérifie que tu as bien installé : streamlit-webrtc, av, ultralytics.")
            return

        # Gestion du chargement du modèle dans st.session_state (lazy)
        cache_key = "yolo_cached_model_path"
        if cache_key not in st.session_state:
            st.session_state[cache_key] = None

        need_reload = False
        if st.session_state[cache_key] != model_path:
            need_reload = True
            st.session_state[cache_key] = model_path
        if btn_reload:
            need_reload = True

        if need_reload or ("yolo_model_obj" not in st.session_state):
            with st.spinner("Chargement du modèle YOLO..."):
                try:
                    st.session_state["yolo_model_obj"] = YOLO(model_path)
                except Exception as e:
                    st.error(f"Erreur lors du chargement du modèle : {e}")
                    return

        model = st.session_state.get("yolo_model_obj", None)
        if model is None:
            st.error("Modèle introuvable ou non chargé.")
            return

        # Définit la classe transformer en utilisant recv() (nouvelle API)
        class ObjectDetection(VideoTransformerBase):
            def __init__(self):
                super().__init__()
                self.model = model
                self.conf = conf
                self.max_boxes = max_boxes
                self.show_labels = show_labels
                self.last_frame = None
                self.prev_time = None
                self.fps = 0.0

            def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
                # transform similaire à l'ancien transform()
                img = frame.to_ndarray(format="bgr24")

                try:
                    results = self.model.predict(img, conf=self.conf, verbose=False)
                    plotted = results[0].plot()
                except Exception:
                    # en cas d'erreur, on renvoie l'image d'origine
                    plotted = img

                # Normaliser le format pour av.VideoFrame (BGR attendu)
                if isinstance(plotted, Image.Image):
                    plotted = np.array(plotted)[:, :, ::-1]  # RGB -> BGR
                else:
                    # si numpy array, ultralytics retourne souvent RGB
                    if plotted.shape[2] == 3:
                        plotted = plotted[:, :, ::-1]  # RGB -> BGR

                self.last_frame = plotted

                # calcul FPS
                now = time.time()
                if self.prev_time is not None:
                    dt = now - self.prev_time
                    if dt > 0:
                        fps_now = 1.0 / dt
                        self.fps = 0.9 * self.fps + 0.1 * fps_now if self.fps else fps_now
                self.prev_time = now

                return av.VideoFrame.from_ndarray(plotted, format="bgr24")

        # Démarrage du streamer
        ctx = webrtc_streamer(
            key="realtime-object-detection",
            mode=WebRtcMode.SENDRECV,
            video_transformer_factory=ObjectDetection,
            media_stream_constraints={"video": True, "audio": False},
            async_transform=True,
            desired_playing_state=True,
        )

        # Statut & FPS
        if ctx and ctx.state.playing:
            status_col.info("Statut : ✅ En ligne")
        else:
            status_col.warning("Statut : ⛔ Caméra non active")

        if ctx and hasattr(ctx, "video_transformer") and ctx.video_transformer is not None and show_fps:
            fps_val = f"{ctx.video_transformer.fps:.1f} fps" if getattr(ctx.video_transformer, "fps", 0) else "—"
            fps_text.metric("FPS", fps_val)
        else:
            fps_text.metric("FPS", "—")

        # Snapshot
        if snapshot_col.button("📸 Capturer un snapshot"):
            if ctx and hasattr(ctx, "video_transformer") and ctx.video_transformer is not None:
                last = ctx.video_transformer.last_frame
                if last is None:
                    st.warning("Aucune frame disponible (attends le flux).")
                else:
                    img_rgb = last[:, :, ::-1]
                    pil_img = Image.fromarray(img_rgb)
                    buf = io.BytesIO()
                    pil_img.save(buf, format="PNG")
                    st.image(pil_img, caption="Snapshot", use_container_width=True)
                    st.download_button("⬇️ Télécharger le snapshot (PNG)", data=buf.getvalue(), file_name="snapshot.png", mime="image/png")
            else:
                st.warning("Flux non disponible — attends quelques secondes.")

        st.caption("Remarque : l'inférence en temps réel utilise des ressources CPU. Pour du vrai temps réel, exécute avec GPU.")
