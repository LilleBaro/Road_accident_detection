import streamlit as st
from streamlit_option_menu import option_menu
from utils.temps_reel import run_temps_reel
from utils.detection_accident_test import resize_video_for_analysis


st.set_page_config(page_title="My Streamlit App", layout="wide")

st.markdown("""
    <style>
        [data-testid="stSidebar"] {display: none;}
    </style>
""", unsafe_allow_html=True)

selected = option_menu(
    menu_title="M E N U",
    options=["Home","A propos", "Detection d'accidents", "Detection en temps réel"],
    icons=["house", "info-circle", "play-circle","camera"],
    menu_icon="cast",
    default_index=0,
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#F0F2D6"},  
        "icon": {"color": "#191970", "font-size": "20px"},  
        "nav-link": {"font-size": "16px", "text-align": "center", "margin": "0px", "--hover-color": "#E1AD01"},
        "nav-link-selected": {"background-color": "#E1AD01"},              
    })

if selected == "Home":
    st.markdown(
        """
        <div style="background-color:#E1AD01; padding:20px; border-radius:10px; text-align:center;">
            <h1 style="color:white;">🚦 Application de Détection d'Accidents & Objets en Temps Réel</h1>
            <p style="color:white; font-size:18px;">Analysez vos vidéos ou utilisez votre caméra pour détecter en direct !</p>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.write("")  

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
            <div style="background-color:#F0F2D6; padding:20px; border-radius:10px;">
                <h3>🛑 Détection d'accidents</h3>
                <p>Uploadez une vidéo pour analyser et détecter automatiquement les accidents routiers grâce à un modèle YOLOv12 fine-tuné.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        st.markdown(
            """
            <div style="background-color:#F0F2D6; padding:20px; border-radius:10px;">
                <h3>📷 Détection en temps réel</h3>
                <p>Activez votre caméra et laissez notre modèle YOLOv12 repérer instantanément les objets autour de vous.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.write("")  

    st.subheader("📌 Comment utiliser l'application ?")
    st.markdown("""
    1. **Détection d'accidents** : Cliquez sur l'onglet *"Détection d'accidents"* pour uploader une vidéo.
    2. **Détection en temps réel** : Cliquez sur l'onglet *"Détection en temps réel"* pour activer votre caméra.
    3. **À propos** : Découvrez notre équipe et le fonctionnement de l'application.
    """)

    st.info("⚡ Conseil : Utilisez un écran large pour une meilleure expérience.")
    st.write("👨‍💻 Développé par **MBENGUE Mohamed Lamine**, **DIALLO Alpha Oumar** et **MANTSOUAKA MPINDOU Franck Arthur**")


elif selected == "A propos":
    st.title("A propos de notre application")

    st.title("À propos de l'application")
    st.markdown("""
    Cette application est un outil de **Computer Vision** développé avec **Streamlit** et basé sur les modèles **YOLOv12**.

    ### Fonctionnalités principales :
    - **Détection en temps réel d'objets** : à l'aide d'un modèle YOLOv12 généraliste, entraîné sur le dataset **COCO**, permettant d'identifier divers objets via la caméra de l'utilisateur.
    - **Détection d'accidents** : grâce à un modèle YOLOv12 fine-tuné spécifiquement sur un dataset d'accidents, l'utilisateur peut analyser des **images** ou **vidéos** pour détecter des situations à risque.

    ### Objectif :
    Fournir une application interactive et intuitive permettant de tester rapidement les capacités des modèles de vision par ordinateur dans des scénarios réels (surveillance, sécurité routière, détection en direct).

    ---
    """)



elif selected == "Detection d'accidents":
    # st.title("Detection d'accidents")
    # st.write("Cette fonctionnalité vous permet de détecter des accidents routiers à partir d'images ou de vidéos.")
    # st.write("Veuillez uploader une vidéo pour commencer la détection.")

# ---------------Code pour la détection d'accidents----------------
    def run_detection_accidents():
        import tempfile
        import os
        import time
        import cv2
        import numpy as np
        from PIL import Image, ImageEnhance, ImageFilter

        st.markdown("""
        <div style="background:#1f4e79; padding:12px; border-radius:8px; color:white;">
            <h2 style="margin:0">🚨 Détection d'Accidents</h2>
            <p style="margin:0.25rem 0 0 0; color:#F0F2D6;">Uploadez une image ou une vidéo pour détecter les accidents.</p>
        </div>
        """, unsafe_allow_html=True)

        # Helper utilities (nested to avoid module-level heavy imports)
        def safe_extract_box(box):
            try:
                coords = box.xyxy[0].cpu().numpy() if hasattr(box.xyxy, "cpu") else np.array(box.xyxy[0])
                x1, y1, x2, y2 = map(int, coords[:4])
                conf = float(box.conf) if hasattr(box, "conf") else float(box[4])
                cls_idx = int(box.cls) if hasattr(box, "cls") else int(box[5])
                return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "conf": conf, "cls": cls_idx}
            except Exception:
                return None

        def process_video_with_annotations(video_path: str, model, conf_threshold: float, max_detections: int, skip_frames: int = 1):
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

        # UI upload
        uploaded_file = st.file_uploader("Choisissez un fichier image ou vidéo", type=["jpg","jpeg","png","bmp","mp4","avi","mov","mkv"]) 
        if uploaded_file is None:
            st.info("👆 Uploadez un fichier pour commencer l'analyse")
            return

        # Controls
        col1, col2 = st.columns([1,1])
        with col1:
            confidence = st.slider("Seuil de confiance", 0.1, 1.0, 0.5, 0.05)
            max_detections = st.number_input("Nombre max de détections", 1, 200, 50)
            show_scores = st.checkbox("Afficher les scores", value=True)
            skip_frames = 1
            optimize_resolution = False
            target_width = 640
            confidence_video = confidence
            if uploaded_file.type.startswith("video"):
                skip_frames = st.slider("Traiter 1 frame sur N", 1, 10, 3)
                optimize_resolution = st.checkbox("Réduire la résolution pour l'analyse", value=True)
                target_width = st.selectbox("Résolution cible", [640,720,1280], index=0)
                confidence_video = st.slider("Seuil de confiance (vidéo)", 0.1, 1.0, 0.4, 0.05)
            detect_button = st.button("🚀 Lancer la Détection")

        with col2:
            st.subheader("📊 Résultats")
            # Lazy model loader using st.cache_resource
            @st.cache_resource
            def _load_model():
                try:
                    from ultralytics import YOLO
                except Exception as e:
                    st.error(f"Impossible d'importer ultralytics: {e}")
                    return None
                try:
                    return YOLO('models/detection_accident.pt')
                except Exception as e:
                    st.error(f"Erreur lors du chargement du modèle: {e}")
                    return None

            model = _load_model()

            if uploaded_file.type.startswith('image'):
                original_image = Image.open(uploaded_file).convert('RGB')
                st.image(original_image, caption='Image uploadée', use_container_width=True)
                if detect_button and model is not None:
                    img_arr = np.array(original_image)
                    with st.spinner("Détection en cours..."):
                        start = time.time()
                        results = model.predict(img_arr, conf=confidence, max_det=max_detections, verbose=False)
                        duration = time.time() - start
                        fps = 1.0 / duration if duration>0 else 0.0
                    if len(results) and getattr(results[0], 'boxes', None) and len(results[0].boxes)>0:
                        st.success('Détections trouvées')
                        annotated = results[0].plot()
                        st.image(annotated, caption='Image annotée', use_container_width=True)
                        # list detections
                        total = 0
                        classes_count = {}
                        for box in results[0].boxes:
                            info = safe_extract_box(box)
                            if info is None:
                                continue
                            cls_name = model.names[info['cls']] if not isinstance(model.names, dict) else model.names.get(info['cls'], str(info['cls']))
                            total += 1
                            classes_count[cls_name] = classes_count.get(cls_name, 0) + 1
                            if show_scores:
                                st.write(f"{cls_name} — conf: {info['conf']:.3f}")
                            else:
                                st.write(cls_name)
                        st.metric('Total détections', total)
                        for k,v in classes_count.items():
                            st.write(f"- {k}: {v}")
                    else:
                        st.warning('Aucune détection trouvée')

            elif uploaded_file.type.startswith('video'):
                st.video(uploaded_file)
                if detect_button and model is not None:
                    tmp_video = None
                    resized_video = None
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as f:
                            f.write(uploaded_file.read())
                            tmp_video = f.name
                        video_for_analysis = tmp_video
                        if optimize_resolution:
                            st.info('🔄 Redimensionnement...')
                            resized_video = resize_video_for_analysis(tmp_video, target_width=int(target_width))
                            video_for_analysis = resized_video
                        conf_threshold = confidence_video
                        st.info('🎬 Génération vidéo annotée...')
                        result = process_video_with_annotations(video_for_analysis, model, conf_threshold, max_detections, skip_frames)
                        if result:
                            st.success('Traitement terminé ✅')
                            with open(result['output_path'], 'rb') as vf:
                                video_bytes = vf.read()
                            st.video(video_bytes, format='video/mp4', start_time=0)
                            st.download_button('⬇️ Télécharger la vidéo annotée', data=video_bytes, file_name=f"annotated_{os.path.basename(uploaded_file.name)}", mime='video/mp4')
                            st.metric('Total détections', result['total_detections'])
                            st.metric('Frames traitées', result['processed_frames'])
                            for cls, cnt in result['detection_classes'].items():
                                emoji = '🚨' if cls.lower()=='accident' else '✅'
                                st.write(f"{emoji} {cls}: {cnt}")
                        else:
                            st.error('Erreur pendant le traitement vidéo.')
                    finally:
                        for p in [tmp_video, resized_video]:
                            try:
                                if p and os.path.exists(p):
                                    os.unlink(p)
                            except Exception:
                                pass

    run_detection_accidents()    

elif selected == "Detection en temps réel":
    st.title("Detection en temps réel")
    st.write("Cette fonctionnalité vous permet de détecter des objets en temps réel à partir de votre caméra.")
    st.write("Veuillez autoriser l'accès à votre caméra pour commencer la détection.")
    run_temps_reel()

