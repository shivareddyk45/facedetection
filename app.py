import os
import sys
import time
import pickle
import numpy as np
import cv2
import streamlit as st
from PIL import Image

# Automatically detect if running inside a headless cloud container (Streamlit Community Cloud)
IS_CLOUD_ENV = (
    os.environ.get("STREAMLIT_SERVER_PORT") is not None
    or os.path.exists("/mount/src")
    or os.environ.get("USER") == "adminuser"
    or (not os.path.exists("/dev/video0") and not sys.platform.startswith("darwin"))
)

# Ensure project root is in sys.path
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.face_utils import (
    load_face_cascade,
    detect_faces,
    draw_faces,
    crop_face,
    save_face_sample,
    get_registered_users,
    delete_registered_user,
    sanitize_name,
    get_camera,
    release_camera,
    load_face_recognizer,
    generate_dataset_embeddings,
    save_embeddings,
    load_cached_embeddings,
    recognize_face,
    draw_recognition_results,
    SFACE_MODEL_PATH,
    EMBEDDINGS_CACHE_PATH,
    DEFAULT_RECOGNITION_THRESHOLD,
    DEFAULT_AMBIGUITY_MARGIN,
)

# -------------------------------------------------------------
# 1. Page Configuration
# -------------------------------------------------------------
st.set_page_config(
    page_title="Face Detection and Recognition",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# 2. Simple, Clean, Professional Styling (Light theme, Blue buttons, Arial)
# -------------------------------------------------------------
st.markdown(
    """
    <style>
    /* 1. Global Reset & Typography */
    html, body, .stApp {
        font-family: Arial, Helvetica, sans-serif !important;
        background-color: #f8f9fa !important;
        color: #212529 !important;
    }

    /* Target standard text elements - DO NOT TOUCH ICONS OR SVGs TO PREVENT LIGATURE OVERLAP */
    p, h1, h2, h3, h4, h5, h6, label, .stMarkdown {
        font-family: Arial, Helvetica, sans-serif !important;
        color: #212529 !important;
    }

    /* Header Typography */
    .main-title {
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #212529 !important;
        margin-bottom: 2px !important;
        margin-top: 0px !important;
    }

    .main-subtitle {
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 1rem !important;
        color: #555555 !important;
        margin-top: 0px !important;
        margin-bottom: 20px !important;
        font-weight: normal !important;
    }

    /* Section Headings */
    h3 {
        font-size: 1.15rem !important;
        font-weight: 600 !important;
        color: #212529 !important;
        margin-top: 14px !important;
        margin-bottom: 8px !important;
    }

    h4 {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #212529 !important;
        margin-top: 10px !important;
        margin-bottom: 6px !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
        border-right: 1px solid #dee2e6 !important;
    }

    section[data-testid="stSidebar"] h3 {
        font-size: 1.05rem !important;
        font-weight: 600 !important;
        margin-top: 12px !important;
        margin-bottom: 8px !important;
    }

    /* Simple, Consistent Blue Buttons */
    div.stButton > button {
        background-color: #0d6efd !important;
        color: #ffffff !important;
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 14px !important;
        font-weight: 500 !important;
        border: 1px solid #0d6efd !important;
        border-radius: 4px !important;
        padding: 6px 14px !important;
        height: 38px !important;
        min-height: 38px !important;
        max-height: 38px !important;
        width: 100% !important;
        box-shadow: none !important;
        text-shadow: none !important;
        display: flex !important;
        align-items: center !important;
        justify-content: center !important;
        text-align: center !important;
        transition: background-color 0.15s ease-in-out !important;
    }

    div.stButton > button:hover {
        background-color: #0b5ed7 !important;
        border-color: #0a58ca !important;
        color: #ffffff !important;
    }

    div.stButton > button:active {
        background-color: #0a58ca !important;
        border-color: #0a53be !important;
    }

    /* Text Input Height Alignment with Buttons */
    .stTextInput div[data-baseweb="input"] {
        height: 38px !important;
        min-height: 38px !important;
        border-radius: 4px !important;
        border: 1px solid #ced4da !important;
        background-color: #ffffff !important;
    }

    .stTextInput input {
        font-family: Arial, Helvetica, sans-serif !important;
        font-size: 14px !important;
        color: #212529 !important;
        padding: 6px 12px !important;
        height: 36px !important;
    }

    /* Metric Boxes */
    div[data-testid="stMetric"] {
        background-color: #ffffff !important;
        border: 1px solid #dee2e6 !important;
        border-radius: 4px !important;
        padding: 10px 14px !important;
        box-shadow: none !important;
    }

    div[data-testid="stMetricValue"] {
        font-family: Arial, Helvetica, sans-serif !important;
        color: #0d6efd !important;
        font-weight: 700 !important;
        font-size: 1.4rem !important;
    }

    div[data-testid="stMetricLabel"] {
        font-family: Arial, Helvetica, sans-serif !important;
        color: #6c757d !important;
        font-size: 0.85rem !important;
    }

    /* Clean Card Containers */
    .clean-card {
        background-color: #ffffff;
        border: 1px solid #dee2e6;
        border-radius: 4px;
        padding: 12px 14px;
        margin-bottom: 12px;
        color: #212529;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# -------------------------------------------------------------
# 3. Header & Subtitle
# -------------------------------------------------------------
st.markdown("<h1 class='main-title'>Face Detection and Recognition</h1>", unsafe_allow_html=True)
st.markdown("<p class='main-subtitle'>Mini project</p>", unsafe_allow_html=True)

# -------------------------------------------------------------
# 4. Sidebar Controls (Organized into three clean sections)
# -------------------------------------------------------------
# Section 1: Camera Settings
st.sidebar.markdown("### Camera Settings")
default_cam_index = 1 if IS_CLOUD_ENV else 0
camera_source = st.sidebar.radio(
    "Camera Input Method",
    options=["Live Webcam Stream (OpenCV)", "Browser Camera / Snapshot"],
    index=default_cam_index
)

if camera_source.startswith("Live Webcam"):
    camera_index = st.sidebar.number_input(
        "Camera Index",
        min_value=0,
        max_value=5,
        value=0,
        step=1
    )
else:
    camera_index = 0

st.sidebar.markdown("---")

# Section 2: Model Settings
st.sidebar.markdown("### Model Settings")
with st.sidebar.expander("Model Tuning & Thresholds", expanded=False):
    scale_factor = st.slider("Detection Scale Factor", min_value=1.05, max_value=1.5, value=1.1, step=0.05)
    min_neighbors = st.slider("Min Neighbors", min_value=3, max_value=10, value=5, step=1)
    recognition_threshold = st.slider(
        "Recognition Confidence Threshold",
        min_value=0.30,
        max_value=0.85,
        value=float(DEFAULT_RECOGNITION_THRESHOLD),
        step=0.01
    )
    ambiguity_margin = st.slider(
        "Ambiguity Rejection Margin",
        min_value=0.02,
        max_value=0.20,
        value=float(DEFAULT_AMBIGUITY_MARGIN),
        step=0.01
    )

st.sidebar.markdown("---")

# Section 3: Dataset Management
st.sidebar.markdown("### Dataset Management")
if st.sidebar.button("Sync Dataset", key="btn_sync_embeddings", use_container_width=True):
    with st.spinner("Generating embeddings..."):
        try:
            rec_net = load_face_recognizer(SFACE_MODEL_PATH)
            embs = generate_dataset_embeddings(recognizer_net=rec_net)
            save_embeddings(embs, EMBEDDINGS_CACHE_PATH)
            st.sidebar.success(f"Successfully synced {len(embs)} profile(s)!")
        except Exception as e:
            st.sidebar.error(f"Sync error: {e}")

registered_people = get_registered_users()
if registered_people:
    with st.sidebar.expander("Remove Person", expanded=False):
        person_to_remove = st.selectbox("Select Person:", options=list(registered_people.keys()), key="select_del_person")
        if st.button("Remove Person", key="btn_delete_person", use_container_width=True):
            if delete_registered_user(person_to_remove):
                st.sidebar.success(f"Removed {person_to_remove}.")
                st.rerun()

# -------------------------------------------------------------
# 5. Session State Management
# -------------------------------------------------------------
if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False
if "mode" not in st.session_state:
    st.session_state["mode"] = "Idle"
if "status_message" not in st.session_state:
    st.session_state["status_message"] = "Camera is currently stopped"
if "last_registered" not in st.session_state:
    st.session_state["last_registered"] = None

# -------------------------------------------------------------
# 6. Action Controls: Camera & Recognition Controls
# -------------------------------------------------------------
st.markdown("### Camera & Recognition Controls")
col_c1, col_c2, col_c3 = st.columns([1, 1, 1], gap="small")
with col_c1:
    btn_recognize = st.button("Start Recognition", key="btn_start_recognition", use_container_width=True)
with col_c2:
    btn_detect = st.button("Face Detection Only", key="btn_detection_only", use_container_width=True)
with col_c3:
    btn_stop = st.button("Stop Camera", key="btn_stop_camera", use_container_width=True)

# -------------------------------------------------------------
# 7. Action Controls: Registration Controls
# -------------------------------------------------------------
st.markdown("### Registration Controls")
col_reg_in, col_reg_btn = st.columns([3.5, 1], gap="small")
with col_reg_in:
    user_name_input = st.text_input(
        "Person Name",
        placeholder="Enter person's name to register (e.g. Shiva)",
        key="input_user_name",
        label_visibility="collapsed"
    )
with col_reg_btn:
    btn_register = st.button("Register Face", key="btn_register_face", use_container_width=True)

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

# Handle Button Click Actions
if btn_detect:
    st.session_state["camera_active"] = True
    st.session_state["mode"] = "Face Detection"
    st.session_state["status_message"] = "Face detection running"
    st.rerun()

elif btn_register:
    clean_name = sanitize_name(user_name_input)
    if not clean_name:
        st.warning("Please enter a person's name before clicking Register Face.")
    else:
        st.session_state["camera_active"] = True
        st.session_state["mode"] = "Registration"
        st.session_state["target_name"] = clean_name
        st.session_state["target_samples"] = 20
        st.session_state["status_message"] = f"Registering face: {clean_name}"
        st.rerun()

elif btn_recognize:
    registered_people = get_registered_users()
    if not registered_people:
        st.warning("No registered faces found! Please register a person before starting recognition.")
    else:
        st.session_state["camera_active"] = True
        st.session_state["mode"] = "Recognition"
        st.session_state["status_message"] = "Real-time recognition running"
        st.rerun()

elif btn_stop:
    st.session_state["camera_active"] = False
    st.session_state["mode"] = "Stopped"
    st.session_state["status_message"] = "Camera is currently stopped"
    st.rerun()

# -------------------------------------------------------------
# 8. Main Section: Camera Feed (Left) & Recognition Results (Right)
# -------------------------------------------------------------
col_cam, col_info = st.columns([3, 2], gap="large")

# Right Column: Recognition Results
with col_info:
    st.markdown("### Recognition Results")

    # 1. Camera Status
    status_card_placeholder = st.empty()
    cam_status_str = "Running" if st.session_state["camera_active"] else "Stopped"
    status_card_placeholder.markdown(
        f"""
        <div class="clean-card">
            <div style="font-size: 0.95rem;"><b>Camera status:</b> <span style="color: {'#0d6efd' if st.session_state['camera_active'] else '#6c757d'}; font-weight: 600;">{cam_status_str}</span></div>
            <div style="font-size: 0.9rem; color: #495057; margin-top: 2px;">Mode: <b>{st.session_state['mode']}</b></div>
        </div>
        """,
        unsafe_allow_html=True
    )

    # 2. Metrics: Faces detected & Recognized faces
    col_m1, col_m2 = st.columns(2, gap="small")
    with col_m1:
        metric_face_count = st.empty()
        metric_face_count.metric(label="Faces detected", value=0)
    with col_m2:
        metric_known_count = st.empty()
        metric_known_count.metric(label="Recognized faces", value=0)

    # 3. Current recognized person's name and confidence
    result_card_placeholder = st.empty()
    debug_card_placeholder = st.empty()
    if not st.session_state["camera_active"]:
        result_card_placeholder.markdown(
            """
            <div class="clean-card">
                <div style="font-weight: 600; margin-bottom: 4px;">Recognition Results</div>
                <div style="font-size: 0.95rem; color: #6c757d;">Camera is currently stopped.</div>
            </div>
            """,
            unsafe_allow_html=True
        )
        debug_card_placeholder.empty()

    # 4. Registered Persons list
    st.markdown("#### Registered Persons")
    reg_people_box = st.empty()
    current_registered = get_registered_users()
    if current_registered:
        profiles_list = "".join([
            f"<div style='padding: 6px 0; border-bottom: 1px solid #dee2e6; font-size: 0.95rem;'>"
            f"<b>{name}</b> — <span style='color: #6c757d;'>{count} sample(s)</span></div>"
            for name, count in current_registered.items()
        ])
        reg_people_box.markdown(
            f"""
            <div class="clean-card">
                {profiles_list}
            </div>
            """,
            unsafe_allow_html=True
        )
    else:
        reg_people_box.markdown(
            """
            <div class="clean-card" style="color: #6c757d; font-size: 0.9rem;">
                No registered persons yet. Enter a name and click <b>Register Face</b>.
            </div>
            """,
            unsafe_allow_html=True
        )

# Left Column: Large Camera Feed
with col_cam:
    st.markdown("### Camera Feed")
    video_placeholder = st.empty()

    # Success message banner when a person is registered
    if st.session_state.get("last_registered"):
        reg_info = st.session_state["last_registered"]
        st.success(f"Registered **{reg_info['name']}** ({reg_info['count']} face samples saved).")

    # Standby state when camera is inactive (Small, simple message with no extra blank space)
    if not st.session_state["camera_active"]:
        video_placeholder.markdown(
            """
            <div class="clean-card" style="text-align: center; padding: 28px 16px;">
                <p style="font-size: 1rem; color: #495057; margin: 0; font-weight: 500;">Camera is currently stopped</p>
            </div>
            """,
            unsafe_allow_html=True
        )


# Helper: Render Browser Camera Flow (for Streamlit Cloud and Mobile Browsers)
def render_browser_camera_flow(
    mode,
    face_cascade,
    scale_factor,
    min_neighbors,
    recognition_threshold,
    ambiguity_margin,
    metric_face_count,
    metric_known_count,
    result_card_placeholder,
    status_card_placeholder,
    debug_card_placeholder=None,
    target_name=None,
    target_samples=20
):
    st.markdown(f"#### {mode} (Browser Camera Mode)")
    st.caption("Capture a photo using your device camera or upload an image file:")

    col_cam_in, col_file_in = st.columns(2, gap="small")
    with col_cam_in:
        cam_shot = st.camera_input("Take photo with webcam:", key=f"browser_cam_{mode}")
    with col_file_in:
        file_shot = st.file_uploader("Or upload image (JPG/PNG):", type=["jpg", "jpeg", "png"], key=f"browser_file_{mode}")

    active_shot = cam_shot if cam_shot is not None else file_shot

    if active_shot is not None:
        try:
            pil_shot = Image.open(active_shot)
            frame_rgb = np.array(pil_shot.convert("RGB"))
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        except Exception as e:
            st.error(f"Error loading image: {e}")
            return

        if mode == "Face Detection":
            faces = detect_faces(frame, face_cascade, scale_factor=scale_factor, min_neighbors=min_neighbors)
            if len(faces) > 1:
                faces = sorted(faces, key=lambda b: (b[0], b[1]))
            metric_face_count.metric(label="Faces detected", value=len(faces))
            metric_known_count.metric(label="Recognized faces", value=0)
            annotated = draw_faces(frame, faces, color=(0, 255, 0), thickness=2)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption=f"Detection: {len(faces)} face(s) found", use_container_width=True)

            if len(faces) == 0:
                result_card_placeholder.markdown(
                    """
                    <div class="clean-card">
                        <div style="font-weight: 600; margin-bottom: 4px;">Face Detection</div>
                        <div style="font-size: 0.95rem; color: #6c757d;">No face detected. Please look at the camera.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if debug_card_placeholder is not None:
                    debug_card_placeholder.empty()
            else:
                faces_html = "".join([
                    f"<div style='font-size: 1.05rem; font-weight: 500; color: #0f5132; padding: 3px 0;'>Face {i + 1}</div>"
                    for i in range(len(faces))
                ])
                result_card_placeholder.markdown(
                    f"""
                    <div class="clean-card">
                        <div style="font-weight: 600; margin-bottom: 6px;">Faces Detected: {len(faces)}</div>
                        {faces_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if debug_card_placeholder is not None:
                    debug_card_placeholder.markdown(
                        f"""
                        <div class="clean-card" style="margin-top: 10px; font-size: 0.85rem; color: #495057;">
                            <b>Detection Diagnostics:</b> {len(faces)} face(s) detected. Face recognition is disabled in Detection Only mode.
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        elif mode == "Recognition":
            try:
                rec_net = load_face_recognizer(SFACE_MODEL_PATH)
                cached_embs = load_cached_embeddings(EMBEDDINGS_CACHE_PATH)
                current_people = get_registered_users()
                if not cached_embs or set(cached_embs.keys()) != set(current_people.keys()):
                    with st.spinner("Preparing face embeddings..."):
                        cached_embs = generate_dataset_embeddings(recognizer_net=rec_net, model_path=SFACE_MODEL_PATH)
                        save_embeddings(cached_embs, EMBEDDINGS_CACHE_PATH)

                if not cached_embs:
                    st.warning("No registered face profiles found! Please register a person first.")
                    return
            except Exception as e:
                st.error(f"Model initialization error: {e}")
                return

            faces = detect_faces(frame, face_cascade, scale_factor=scale_factor, min_neighbors=min_neighbors)
            if len(faces) > 1:
                faces = sorted(faces, key=lambda b: (b[0], b[1]))
            metric_face_count.metric(label="Faces detected", value=len(faces))

            recognitions = []
            known_count = 0
            recognized_cards = []
            debug_items = []

            for idx, (x, y, w, h) in enumerate(faces):
                face_num = idx + 1
                face_crop = crop_face(frame, (x, y, w, h), target_size=(112, 112))
                res = recognize_face(
                    face_crop,
                    rec_net,
                    cached_embs,
                    threshold=recognition_threshold,
                    ambiguity_margin=ambiguity_margin
                )
                if isinstance(res, (tuple, list)):
                    name = res[0]
                    score = res[1]
                    is_known = res[2] if len(res) > 2 else False
                    dbg = res[4] if len(res) > 4 else {}
                elif isinstance(res, dict):
                    name = res.get("name", "Unknown Person")
                    score = res.get("score", res.get("confidence", 0.0))
                    is_known = res.get("is_known", False)
                    dbg = res.get("debug_info", {})
                else:
                    name, score, is_known = "Unknown Person", 0.0, False
                    dbg = {}

                confidence = score * 100.0 if score <= 1.0 else score
                if is_known and name and name != "Unknown Person":
                    known_count += 1
                    label = f"Face {face_num} - {name}"
                    recognized_cards.append((face_num, name, confidence, True))
                else:
                    name = "Unknown Person"
                    is_known = False
                    label = f"Face {face_num} - Unknown Person"
                    recognized_cards.append((face_num, "Unknown Person", confidence, False))

                recognitions.append((x, y, w, h, name, confidence, is_known, label, face_num))

                debug_items.append({
                    "face_num": face_num,
                    "best_name": dbg.get("best_name", "None"),
                    "score": confidence,
                    "threshold": recognition_threshold * 100.0,
                    "decision": name,
                    "is_known": is_known,
                    "reason": dbg.get("reason", "")
                })

            metric_known_count.metric(label="Recognized faces", value=known_count)
            annotated = draw_recognition_results(frame, recognitions)
            st.image(cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB), caption="Recognition Result", use_container_width=True)

            # Update Recognition Result Card beside camera
            if len(faces) == 0:
                result_card_placeholder.markdown(
                    """
                    <div class="clean-card">
                        <div style="font-weight: 600; margin-bottom: 4px;">Recognition Results</div>
                        <div style="font-size: 0.95rem; color: #6c757d;">No face detected. Please look at the camera.</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                if debug_card_placeholder is not None:
                    debug_card_placeholder.empty()
            else:
                matches_html = "".join([
                    f"<div style='margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #dee2e6;'>"
                    f"<div style='font-size: 1.05rem; font-weight: bold; color: #0f5132;'>Face {fnum} — {n} <span style='font-size: 0.85rem; font-weight: normal; color: #198754;'>(Registered)</span></div>"
                    f"<div style='font-size: 0.85rem; color: #198754;'>Confidence: {c:.1f}%</div>"
                    f"</div>"
                    if k else
                    f"<div style='margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #dee2e6;'>"
                    f"<div style='font-size: 1.05rem; font-weight: bold; color: #842029;'>Face {fnum} — Unknown Person</div>"
                    f"</div>"
                    for fnum, n, c, k in recognized_cards
                ])
                result_card_placeholder.markdown(
                    f"""
                    <div class="clean-card">
                        <div style="font-weight: 600; font-size: 0.95rem; color: #212529; margin-bottom: 8px;">Recognition Results ({len(faces)} face(s)):</div>
                        {matches_html}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

                if debug_card_placeholder is not None:
                    debug_rows_html = "".join([
                        f"<div style='font-size: 0.85rem; padding: 6px 8px; margin-bottom: 6px; background: #ffffff; border-radius: 4px; border-left: 3px solid {'#198754' if d['is_known'] else '#dc3545'};'>"
                        f"<div><b>Face {d['face_num']}</b> &rarr; <b>{d['decision']}</b></div>"
                        f"<div style='color: #495057;'>Closest Profile: <b>{d['best_name']}</b></div>"
                        f"<div style='color: #495057;'>Matching Score: <b>{d['score']:.1f}%</b> | Threshold: <b>{d['threshold']:.0f}%</b></div>"
                        f"<div style='color: {'#0f5132' if d['is_known'] else '#842029'}; font-size: 0.8rem; margin-top: 2px;'>{d['reason']}</div>"
                        f"</div>"
                        for d in debug_items
                    ])
                    debug_card_placeholder.markdown(
                        f"""
                        <div class="clean-card" style="margin-top: 10px;">
                            <div style="font-weight: 600; font-size: 0.9rem; margin-bottom: 6px; color: #495057;">
                                Recognition Diagnostics ({len(faces)} face(s))
                            </div>
                            {debug_rows_html}
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

        elif mode == "Registration":
            reg_name = target_name
            if not reg_name:
                st.error("No target person name specified.")
                return

            faces = detect_faces(frame, face_cascade, scale_factor=scale_factor, min_neighbors=min_neighbors)
            if len(faces) == 0:
                st.warning("No face detected in the photo! Please look directly at the camera.")
            else:
                largest_face = max(faces, key=lambda box: box[2] * box[3])
                face_crop = crop_face(frame, largest_face, target_size=(160, 160))
                existing_users = get_registered_users()
                sample_idx = existing_users.get(reg_name, 0) + 1
                saved_path = save_face_sample(face_crop, reg_name, sample_idx)

                with st.spinner("Updating deep face embeddings..."):
                    try:
                        rec_net = load_face_recognizer(SFACE_MODEL_PATH)
                        embs = generate_dataset_embeddings(recognizer_net=rec_net)
                        save_embeddings(embs, EMBEDDINGS_CACHE_PATH)
                    except Exception as e:
                        st.error(f"Error updating embeddings: {e}")

                st.success(f"Successfully captured sample #{sample_idx} for {reg_name}!")
                st.image(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB), caption=f"Saved sample #{sample_idx} ({reg_name})", width=180)
                st.info("Capture more photos from different angles or click 'Start Recognition' to test.")
    else:
        st.info("Look at your camera and click 'Take photo' above (or upload an image).")


# -------------------------------------------------------------
# 9. Active Camera Processing (Webcam / Browser Camera)
# -------------------------------------------------------------
if st.session_state["camera_active"]:
    try:
        face_cascade = load_face_cascade()
    except RuntimeError as err:
        st.error(f"Error loading face detector: {err}")
        st.session_state["camera_active"] = False
        st.stop()

    # Cloud or Browser Camera Fallback
    if camera_source.startswith("Browser Camera"):
        render_browser_camera_flow(
            mode=st.session_state["mode"],
            face_cascade=face_cascade,
            scale_factor=scale_factor,
            min_neighbors=min_neighbors,
            recognition_threshold=recognition_threshold,
            ambiguity_margin=ambiguity_margin,
            metric_face_count=metric_face_count,
            metric_known_count=metric_known_count,
            result_card_placeholder=result_card_placeholder,
            status_card_placeholder=status_card_placeholder,
            debug_card_placeholder=debug_card_placeholder,
            target_name=st.session_state.get("target_name"),
            target_samples=st.session_state.get("target_samples", 20)
        )

    # Live OpenCV Webcam Stream
    else:
        cap = get_camera(camera_index)

        # Handle unavailable physical camera by cleanly switching to Browser Camera
        if cap is None or not cap.isOpened():
            st.info(
                "Cloud server environment detected (no physical camera attached to server). "
                "Switched automatically to Browser Camera mode below."
            )
            render_browser_camera_flow(
                mode=st.session_state["mode"],
                face_cascade=face_cascade,
                scale_factor=scale_factor,
                min_neighbors=min_neighbors,
                recognition_threshold=recognition_threshold,
                ambiguity_margin=ambiguity_margin,
                metric_face_count=metric_face_count,
                metric_known_count=metric_known_count,
                result_card_placeholder=result_card_placeholder,
                status_card_placeholder=status_card_placeholder,
                debug_card_placeholder=debug_card_placeholder,
                target_name=st.session_state.get("target_name"),
                target_samples=st.session_state.get("target_samples", 20)
            )
        else:
            # -------------------------------------------------------------
            # MODE 1: REAL-TIME DEEP FACE RECOGNITION
            # -------------------------------------------------------------
            if st.session_state["mode"] == "Recognition":
                try:
                    rec_net = load_face_recognizer(SFACE_MODEL_PATH)
                    cached_embs = load_cached_embeddings(EMBEDDINGS_CACHE_PATH)

                    current_people = get_registered_users()
                    if not cached_embs or set(cached_embs.keys()) != set(current_people.keys()):
                        with st.spinner("Generating face embeddings from registered dataset..."):
                            cached_embs = generate_dataset_embeddings(
                                recognizer_net=rec_net,
                                model_path=SFACE_MODEL_PATH
                            )
                            save_embeddings(cached_embs, EMBEDDINGS_CACHE_PATH)

                    if not cached_embs:
                        st.error("No face embeddings could be generated. Please register a face first.")
                        release_camera(cap)
                        st.session_state["camera_active"] = False
                        st.stop()

                except Exception as e:
                    st.error(f"Error initializing face recognition model: {e}")
                    release_camera(cap)
                    st.session_state["camera_active"] = False
                    st.stop()

                try:
                    while st.session_state.get("camera_active", False):
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Failed to capture frame from webcam. Stream stopped.")
                            break

                        faces = detect_faces(frame, face_cascade, scale_factor=scale_factor, min_neighbors=min_neighbors)
                        if len(faces) > 1:
                            faces = sorted(faces, key=lambda b: (b[0], b[1]))
                        metric_face_count.metric(label="Faces detected", value=len(faces))

                        recognitions = []
                        known_count = 0
                        recognized_cards = []
                        debug_items = []

                        for idx, (x, y, w, h) in enumerate(faces):
                            face_num = idx + 1
                            face_crop = crop_face(frame, (x, y, w, h), target_size=(112, 112))
                            res = recognize_face(
                                face_crop,
                                rec_net,
                                cached_embs,
                                threshold=recognition_threshold,
                                ambiguity_margin=ambiguity_margin
                            )

                            if isinstance(res, (tuple, list)):
                                name = res[0]
                                score = res[1]
                                is_known = res[2] if len(res) > 2 else False
                                dbg = res[4] if len(res) > 4 else {}
                            elif isinstance(res, dict):
                                name = res.get("name", "Unknown Person")
                                score = res.get("score", res.get("confidence", 0.0))
                                is_known = res.get("is_known", False)
                                dbg = res.get("debug_info", {})
                            else:
                                name, score, is_known = "Unknown Person", 0.0, False
                                dbg = {}

                            confidence = score * 100.0 if score <= 1.0 else score

                            # Bounding box label: 'Face X - {Name}' or 'Face Y - Unknown Person'
                            if is_known and name and name != "Unknown Person":
                                known_count += 1
                                label = f"Face {face_num} - {name}"
                                recognized_cards.append((face_num, name, confidence, True))
                            else:
                                name = "Unknown Person"
                                is_known = False
                                label = f"Face {face_num} - Unknown Person"
                                recognized_cards.append((face_num, "Unknown Person", confidence, False))

                            recognitions.append((x, y, w, h, name, confidence, is_known, label, face_num))

                            # Debug diagnostics item for this face
                            debug_items.append({
                                "face_num": face_num,
                                "best_name": dbg.get("best_name", "None"),
                                "score": confidence,
                                "threshold": recognition_threshold * 100.0,
                                "decision": name,
                                "is_known": is_known,
                                "reason": dbg.get("reason", "")
                            })

                        metric_known_count.metric(label="Recognized faces", value=known_count)

                        # Update Recognition Result card beside the camera
                        if len(faces) == 0:
                            result_card_placeholder.markdown(
                                """
                                <div class="clean-card">
                                    <div style="font-weight: 600; margin-bottom: 4px;">Recognition Results</div>
                                    <div style="font-size: 0.95rem; color: #6c757d;">No face detected. Please look at the camera.</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            debug_card_placeholder.empty()
                        else:
                            matches_html = "".join([
                                f"<div style='margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #dee2e6;'>"
                                f"<div style='font-size: 1.05rem; font-weight: bold; color: #0f5132;'>Face {fnum} — {n} <span style='font-size: 0.85rem; font-weight: normal; color: #198754;'>(Registered)</span></div>"
                                f"<div style='font-size: 0.85rem; color: #198754;'>Confidence: {c:.1f}%</div>"
                                f"</div>"
                                if k else
                                f"<div style='margin-bottom: 8px; padding-bottom: 6px; border-bottom: 1px solid #dee2e6;'>"
                                f"<div style='font-size: 1.05rem; font-weight: bold; color: #842029;'>Face {fnum} — Unknown Person</div>"
                                f"</div>"
                                for fnum, n, c, k in recognized_cards
                            ])
                            result_card_placeholder.markdown(
                                f"""
                                <div class="clean-card">
                                    <div style="font-weight: 600; font-size: 0.95rem; color: #212529; margin-bottom: 8px;">Recognition Results ({len(faces)} face(s)):</div>
                                    {matches_html}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                            # Live Recognition Diagnostics (Debugging Information)
                            debug_rows_html = "".join([
                                f"<div style='font-size: 0.85rem; padding: 6px 8px; margin-bottom: 6px; background: #ffffff; border-radius: 4px; border-left: 3px solid {'#198754' if d['is_known'] else '#dc3545'};'>"
                                f"<div><b>Face {d['face_num']}</b> &rarr; <b>{d['decision']}</b></div>"
                                f"<div style='color: #495057;'>Closest Profile: <b>{d['best_name']}</b></div>"
                                f"<div style='color: #495057;'>Matching Score: <b>{d['score']:.1f}%</b> | Threshold: <b>{d['threshold']:.0f}%</b></div>"
                                f"<div style='color: {'#0f5132' if d['is_known'] else '#842029'}; font-size: 0.8rem; margin-top: 2px;'>{d['reason']}</div>"
                                f"</div>"
                                for d in debug_items
                            ])
                            debug_card_placeholder.markdown(
                                f"""
                                <div class="clean-card" style="margin-top: 10px;">
                                    <div style="font-weight: 600; font-size: 0.9rem; margin-bottom: 6px; color: #495057;">
                                        Recognition Diagnostics ({len(faces)} face(s))
                                    </div>
                                    {debug_rows_html}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        # Draw bounding boxes: Green box for recognized, Red box for unknown
                        annotated = draw_recognition_results(frame, recognitions)

                        # Render live frame
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        time.sleep(0.02)

                finally:
                    release_camera(cap)

            # -------------------------------------------------------------
            # MODE 2: FACE REGISTRATION
            # -------------------------------------------------------------
            elif st.session_state["mode"] == "Registration":
                reg_name = st.session_state["target_name"]
                total_target = st.session_state.get("target_samples", 20)

                reg_progress = st.progress(0.0)
                status_feedback = st.empty()
                saved_count = 0
                sample_previews = []
                last_save_time = 0

                try:
                    while st.session_state.get("camera_active", False) and saved_count < total_target:
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Failed to capture frame from webcam.")
                            break

                        faces = detect_faces(frame, face_cascade, scale_factor=scale_factor, min_neighbors=min_neighbors)
                        metric_face_count.metric(label="Faces detected", value=len(faces))
                        metric_known_count.metric(label="Recognized faces", value=0)

                        annotated_frame = frame.copy()

                        if len(faces) == 0:
                            status_feedback.warning("No face detected! Please look directly at the webcam.")
                            cv2.putText(
                                annotated_frame,
                                "No face detected! Look at camera",
                                (20, 40),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                0.7,
                                (0, 0, 255),
                                2,
                                cv2.LINE_AA
                            )
                        else:
                            largest_face = max(faces, key=lambda box: box[2] * box[3])

                            current_time = time.time()
                            if current_time - last_save_time >= 0.12:
                                face_crop = crop_face(frame, largest_face, target_size=(160, 160))
                                save_face_sample(face_crop, reg_name, saved_count + 1)
                                saved_count += 1
                                last_save_time = current_time

                                if len(sample_previews) < 4:
                                    sample_previews.append(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

                                reg_progress.progress(saved_count / total_target)
                                status_feedback.info(f"Capturing sample {saved_count} of {total_target} for {reg_name}...")

                            annotated_frame = draw_faces(
                                frame,
                                [largest_face],
                                color=(0, 255, 0),
                                custom_label=f"{reg_name} ({saved_count}/{total_target})"
                            )

                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        time.sleep(0.02)

                    if saved_count >= total_target:
                        st.session_state["camera_active"] = False
                        st.session_state["mode"] = "Idle"
                        st.session_state["last_registered"] = {
                            "name": reg_name,
                            "count": saved_count,
                            "previews": sample_previews
                        }
                        st.session_state["status_message"] = f"Successfully registered {reg_name}."

                        try:
                            rec_net = load_face_recognizer(SFACE_MODEL_PATH)
                            embs = generate_dataset_embeddings(recognizer_net=rec_net)
                            save_embeddings(embs, EMBEDDINGS_CACHE_PATH)
                        except Exception:
                            pass

                        st.rerun()

                finally:
                    release_camera(cap)

            # -------------------------------------------------------------
            # MODE 3: FACE DETECTION ONLY
            # -------------------------------------------------------------
            elif st.session_state["mode"] == "Face Detection":
                try:
                    while st.session_state.get("camera_active", False):
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("Failed to capture frame from webcam.")
                            break

                        faces = detect_faces(
                            frame,
                            face_cascade,
                            scale_factor=scale_factor,
                            min_neighbors=min_neighbors
                        )
                        if len(faces) > 1:
                            faces = sorted(faces, key=lambda b: (b[0], b[1]))

                        annotated = draw_faces(frame, faces, color=(0, 255, 0), thickness=2)
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        metric_face_count.metric(label="Faces detected", value=len(faces))
                        metric_known_count.metric(label="Recognized faces", value=0)

                        if len(faces) == 0:
                            result_card_placeholder.markdown(
                                """
                                <div class="clean-card">
                                    <div style="font-weight: 600; margin-bottom: 4px;">Face Detection</div>
                                    <div style="font-size: 0.95rem; color: #6c757d;">No face detected. Please look at the camera.</div>
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            debug_card_placeholder.empty()
                        else:
                            faces_html = "".join([
                                f"<div style='font-size: 1.05rem; font-weight: 500; color: #0f5132; padding: 3px 0;'>Face {i + 1}</div>"
                                for i in range(len(faces))
                            ])
                            result_card_placeholder.markdown(
                                f"""
                                <div class="clean-card">
                                    <div style="font-weight: 600; margin-bottom: 6px;">Faces Detected: {len(faces)}</div>
                                    {faces_html}
                                </div>
                                """,
                                unsafe_allow_html=True
                            )
                            debug_card_placeholder.markdown(
                                f"""
                                <div class="clean-card" style="margin-top: 10px; font-size: 0.85rem; color: #495057;">
                                    <b>Detection Diagnostics:</b> {len(faces)} face(s) detected. Face recognition is disabled in Detection Only mode.
                                </div>
                                """,
                                unsafe_allow_html=True
                            )

                        time.sleep(0.02)

                finally:
                    release_camera(cap)

# -------------------------------------------------------------
# 10. Clean Footer
# -------------------------------------------------------------
st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)
st.markdown("<p style='color: #6c757d; font-size: 0.85rem; text-align: center;'>Face Detection and Recognition | Mini project</p>", unsafe_allow_html=True)
