"""
Face Detection and Recognition - Main Streamlit Application (Final Stage 5 - Debugged & Optimized)

Complete End-to-End Pipeline:
1. Stage 1: Intuitive Streamlit UI and clean modular project structure.
2. Stage 2: Real-time OpenCV face detection using Haar Cascades.
3. Stage 3: Dataset face registration with face-presence validation and multi-person support.
4. Stage 4: Pre-trained deep learning face recognition using the OpenCV SFace DNN model:
    - 128-dimensional L2-normalized embeddings.
    - Multi-sample matrix cosine matching against all registered photos per individual.
    - Calibrated confidence threshold (0.38) to accurately identify registered persons.
    - Handles unknown faces with red badges and registered faces with green badges.
5. Stage 5: Full integration with live diagnostics, score displays, and robust error handling.
"""

import time
import cv2
import numpy as np
import streamlit as st

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

# 1. Page Configuration
st.set_page_config(
    page_title="Face Detection and Recognition",
    page_icon="👤",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 2. Main Title and Header
st.title("👤 Face Detection and Recognition System")
st.caption("Deep Learning Computer Vision Mini-Project | Live Diagnostics & Recognition")
st.markdown(
    """
    An end-to-end Computer Vision & Deep Learning application built with **Python**, **OpenCV**, and **Streamlit**.
    Detect faces in real time, register individuals into an organized dataset, and identify people using **deep face embeddings (SFace DNN)**.
    """
)

# 3. Sidebar Controls
st.sidebar.header("⚙️ System Controls")

# Camera selection
camera_index = st.sidebar.number_input(
    "Camera Index:",
    min_value=0,
    max_value=5,
    value=0,
    step=1,
    help="Default is 0 (built-in webcam). Select 1 or 2 if using an external USB camera."
)

# Model sensitivity sliders
with st.sidebar.expander("🔧 Model Tuning & Thresholds", expanded=False):
    scale_factor = st.slider("Detection Scale Factor", min_value=1.05, max_value=1.5, value=1.1, step=0.05,
                             help="Image reduction scale factor for Haar Cascade.")
    min_neighbors = st.slider("Min Neighbors", min_value=3, max_value=10, value=5, step=1,
                              help="Minimum neighbor detections to confirm a face.")
    recognition_threshold = st.slider(
        "Recognition Confidence Threshold",
        min_value=0.25,
        max_value=0.85,
        value=float(DEFAULT_RECOGNITION_THRESHOLD),
        step=0.02,
        help="Calibrated threshold (0.50). Faces with cosine similarity >= this are recognized."
    )
    ambiguity_margin = st.slider(
        "Ambiguity Rejection Margin",
        min_value=0.02,
        max_value=0.20,
        value=float(DEFAULT_AMBIGUITY_MARGIN),
        step=0.01,
        help="Minimum score separation required between top two candidate identities to avoid false guessing."
    )

st.sidebar.markdown("---")

# Face Registration Section
st.sidebar.subheader("👤 Register New Person")
user_name = st.sidebar.text_input("Enter Person's Name:", placeholder="e.g. Alice", key="input_user_name")
sample_target = st.sidebar.slider("Face Samples to Capture:", min_value=10, max_value=50, value=20, step=5,
                                  help="Number of face images to collect for registration.")

# Action Buttons
st.sidebar.subheader("🚀 Actions")
col_b1, col_b2 = st.sidebar.columns(2)
with col_b1:
    btn_register = st.button("➕ Register Face", key="btn_register_face", use_container_width=True, type="primary")
with col_b2:
    btn_recognize = st.button("🔍 Recognize", key="btn_start_recognition", use_container_width=True, type="primary")

btn_detect = st.sidebar.button("▶️ Face Detection Only", key="btn_detection_only", use_container_width=True)
btn_stop = st.sidebar.button("⏹️ Stop Camera", key="btn_stop_camera", use_container_width=True, type="secondary")

# Registered Dataset & Management Section
st.sidebar.markdown("---")
st.sidebar.subheader("👥 Registered Dataset")
registered_people = get_registered_users()

if registered_people:
    for name, count in registered_people.items():
        st.sidebar.write(f"• **{name}** ({count} images)")

    # Dataset maintenance buttons
    col_sync, col_del = st.sidebar.columns(2)
    with col_sync:
        if st.button("🔄 Sync", key="btn_sync_embeddings", use_container_width=True, help="Re-generate deep embeddings for all registered faces"):
            with st.spinner("Generating embeddings..."):
                try:
                    rec_net = load_face_recognizer(SFACE_MODEL_PATH)
                    embs = generate_dataset_embeddings(recognizer_net=rec_net)
                    save_embeddings(embs, EMBEDDINGS_CACHE_PATH)
                    st.sidebar.success(f"Synced {len(embs)} profile(s)!")
                except Exception as e:
                    st.sidebar.error(f"Sync error: {e}")

    # Delete person option
    with st.sidebar.expander("🗑️ Remove a Person", expanded=False):
        person_to_remove = st.selectbox("Select Person:", options=list(registered_people.keys()), key="select_del_person")
        if st.button(f"Delete '{person_to_remove}'", key="btn_delete_person"):
            if delete_registered_user(person_to_remove):
                st.sidebar.success(f"Removed {person_to_remove}.")
                st.rerun()
else:
    st.sidebar.caption("No registered faces yet. Enter a name and click 'Register Face' above.")

# 4. Session State Management
if "camera_active" not in st.session_state:
    st.session_state["camera_active"] = False
if "mode" not in st.session_state:
    st.session_state["mode"] = "Idle"
if "status_message" not in st.session_state:
    st.session_state["status_message"] = "Camera is currently idle. Select an action from the sidebar."
if "last_registered" not in st.session_state:
    st.session_state["last_registered"] = None

# Handle Button Actions
if btn_detect:
    st.session_state["camera_active"] = True
    st.session_state["mode"] = "Face Detection"
    st.session_state["status_message"] = "🟢 Camera active: Live Face Detection running."

elif btn_register:
    clean_name = sanitize_name(user_name)
    if not clean_name:
        st.sidebar.error("⚠️ Please enter a valid person's name before registering.")
    else:
        st.session_state["camera_active"] = True
        st.session_state["mode"] = "Registration"
        st.session_state["target_name"] = clean_name
        st.session_state["target_samples"] = sample_target
        st.session_state["status_message"] = f"📸 Starting face registration for: **{clean_name}**"

elif btn_recognize:
    if not registered_people:
        st.sidebar.warning("⚠️ No registered faces found! Please register at least one person before starting recognition.")
    else:
        st.session_state["camera_active"] = True
        st.session_state["mode"] = "Recognition"
        st.session_state["status_message"] = "🔍 Camera active: Real-Time Deep Face Recognition running."

elif btn_stop:
    st.session_state["camera_active"] = False
    st.session_state["mode"] = "Stopped"
    st.session_state["status_message"] = "⏹️ Camera stopped. Ready for next action."

# 5. Main UI Layout: Video Stream & System Dashboard
col_feed, col_status = st.columns([3, 2])

with col_status:
    st.subheader("📋 System Status & Diagnostics")
    st.metric(label="Active Mode", value=st.session_state["mode"])

    col_m1, col_m2 = st.columns(2)
    with col_m1:
        metric_face_count = st.empty()
        metric_face_count.metric(label="Faces in View", value=0)
    with col_m2:
        metric_known_count = st.empty()
        metric_known_count.metric(label="Recognized Faces", value=0)

    # In-dashboard stop button
    if st.session_state["camera_active"]:
        if st.button("⏹️ Stop Camera Stream", key="main_stop_btn", use_container_width=True):
            st.session_state["camera_active"] = False
            st.session_state["mode"] = "Stopped"
            st.session_state["status_message"] = "⏹️ Camera stopped."
            st.rerun()

    # Live Recognition Diagnostics Container
    diag_box = st.empty()

    # Step-by-Step Instructions
    st.markdown("---")
    st.markdown("### 📌 User Guide")
    with st.expander("📖 Step-by-Step Instructions", expanded=False):
        st.markdown(
            """
            **Step 1: Register Face**
            - Enter a name (e.g. `Alice`) in the sidebar.
            - Click **➕ Register Face** and look at the camera until 20 samples are saved.
            - Embeddings are generated automatically upon completion.
            
            **Step 2: Start Recognition**
            - Click **🔍 Recognize**.
            - The live webcam will display a **Green box** with the name and match percentage for registered faces.
            - Unregistered individuals will be outlined in a **Red box** with **"Unknown Person"**.
            
            **Step 3: Fine-Tune Threshold**
            - Adjust the **Recognition Confidence Threshold** in the sidebar (calibrated default: `0.50`).
            - The **Ambiguity Rejection Margin** (default: `0.08`) ensures that if two candidates are too close in score, the system returns **"Unknown Person"** rather than guessing.
            """
        )

with col_feed:
    st.subheader("📹 Live Video Feed")
    video_placeholder = st.empty()

    # Registration success banner with preview thumbnails
    if st.session_state.get("last_registered"):
        reg_info = st.session_state["last_registered"]
        st.success(
            f"🎉 **Registration Successful!**\n\n"
            f"- **Person:** `{reg_info['name']}`\n"
            f"- **Samples Saved:** `{reg_info['count']}` images\n"
            f"- **Folder:** `data/faces/{reg_info['name']}/`"
        )
        if reg_info.get("previews"):
            st.write("Captured Face Samples Preview:")
            preview_cols = st.columns(len(reg_info["previews"]))
            for idx, p_col in enumerate(preview_cols):
                with p_col:
                    st.image(reg_info["previews"][idx], caption=f"Sample {idx + 1}", use_container_width=True)

    # Standby state when camera is idle
    if not st.session_state["camera_active"]:
        video_placeholder.info(f"**Status:** {st.session_state['status_message']}")

    # Active Camera Stream
    else:
        # Load Haar Cascade face detector
        try:
            face_cascade = load_face_cascade()
        except RuntimeError as err:
            st.error(f"❌ {err}")
            st.session_state["camera_active"] = False
            st.stop()

        # Connect to webcam
        cap = get_camera(camera_index)

        # Handle unavailable webcam
        if cap is None or not cap.isOpened():
            st.error(
                f"❌ **Webcam Unavailable (Camera Index {camera_index})**\n\n"
                "Could not connect to the webcam. Please verify:\n"
                "1. Your camera is plugged in and recognized by your system.\n"
                "2. No other program (e.g. Zoom, FaceTime) is currently using the camera.\n"
                "3. Your terminal or browser has permission to access the camera in OS settings.\n"
                "4. If you have an external USB webcam, try changing the **Camera Index** in the sidebar."
            )
            st.session_state["camera_active"] = False
            st.session_state["mode"] = "Error"
        else:
            # -------------------------------------------------------------
            # MODE 1: REAL-TIME DEEP FACE RECOGNITION (Stage 4 & 5)
            # -------------------------------------------------------------
            if st.session_state["mode"] == "Recognition":
                # Ensure model and embeddings are loaded
                try:
                    rec_net = load_face_recognizer(SFACE_MODEL_PATH)
                    cached_embs = load_cached_embeddings(EMBEDDINGS_CACHE_PATH)

                    # Auto-generate embeddings if cache is missing, empty, or out of sync
                    current_people = get_registered_users()
                    if not cached_embs or set(cached_embs.keys()) != set(current_people.keys()):
                        with st.spinner("Generating prototype face embeddings from dataset..."):
                            cached_embs = generate_dataset_embeddings(
                                recognizer_net=rec_net,
                                model_path=SFACE_MODEL_PATH
                            )
                            save_embeddings(cached_embs, EMBEDDINGS_CACHE_PATH)

                    if not cached_embs:
                        st.error("❌ No face embeddings could be generated. Please register at least one face first.")
                        release_camera(cap)
                        st.session_state["camera_active"] = False
                        st.stop()

                except Exception as e:
                    st.error(f"❌ Error initializing face recognition model: {e}")
                    release_camera(cap)
                    st.session_state["camera_active"] = False
                    st.stop()

                # Display initial diagnostics
                loaded_profiles_str = ", ".join([f"{k} ({v.get('count', 0) if isinstance(v, dict) else 'ok'} samples)" for k, v in cached_embs.items()])
                diag_box.info(f"**Loaded Profiles ({len(cached_embs)}):** {loaded_profiles_str}\n\n**Threshold:** `{recognition_threshold:.2f}`")

                try:
                    while st.session_state.get("camera_active", False):
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("⚠️ Failed to capture frame from webcam. Stream stopped.")
                            break

                        # 1. Detect all faces in current frame
                        faces = detect_faces(frame, face_cascade, scale_factor=scale_factor, min_neighbors=min_neighbors)
                        metric_face_count.metric(label="Faces in View", value=len(faces))

                        recognitions = []
                        known_count = 0
                        diagnostics_info = []

                        if len(faces) == 0:
                            diagnostics_info.append("🔍 **Face Detected:** No face in view.")
                        else:
                            diagnostics_info.append(f"🔍 **Face Detected:** {len(faces)} face(s) in frame.")

                        # 2. Recognize each detected face
                        for idx, (x, y, w, h) in enumerate(faces):
                            face_crop = crop_face(frame, (x, y, w, h), target_size=(112, 112))
                            res = recognize_face(
                                face_crop,
                                rec_net,
                                cached_embs,
                                threshold=recognition_threshold,
                                ambiguity_margin=ambiguity_margin
                            )

                            # Extract name, score, known-status, candidate scores, and debug info safely
                            if isinstance(res, (tuple, list)):
                                name = res[0]
                                score = res[1]
                                is_known = res[2] if len(res) > 2 else False
                                candidate_scores = res[3] if len(res) > 3 else {}
                                debug_info = res[4] if len(res) > 4 else {}
                            elif isinstance(res, dict):
                                name = res.get("name", "Unknown Person")
                                score = res.get("score", res.get("confidence", 0.0))
                                is_known = res.get("is_known", False)
                                candidate_scores = res.get("candidate_scores", {})
                                debug_info = res.get("debug_info", {})
                            else:
                                name, score, is_known, candidate_scores, debug_info = "Unknown Person", 0.0, False, {}, {}

                            # Ensure score is represented as a percentage (e.g. 0.809 -> 80.9)
                            confidence = score * 100.0 if score <= 1.0 else score

                            # Clean label: '{Name} ({confidence:.1f}%)' when recognized, 'Unknown Person' when not
                            if is_known and name and name != "Unknown Person":
                                known_count += 1
                                label = f"{name} ({confidence:.1f}%)"
                            else:
                                name = "Unknown Person"
                                is_known = False
                                label = "Unknown Person"

                            recognitions.append((x, y, w, h, name, confidence, is_known, label))

                            # Format detailed debugging diagnostics meeting Requirement 10
                            best_id = debug_info.get("best_name", name if is_known else "Unknown")
                            best_sc = debug_info.get("best_score", score)
                            best_sc_pct = best_sc * 100.0 if best_sc <= 1.0 else best_sc

                            second_id = debug_info.get("second_name")
                            second_sc = debug_info.get("second_score", 0.0)
                            second_sc_pct = second_sc * 100.0 if second_sc <= 1.0 else second_sc

                            margin_val = debug_info.get("margin", 0.0)
                            margin_pct = margin_val * 100.0 if margin_val <= 1.0 else margin_val

                            is_ambiguous = debug_info.get("is_ambiguous", False)
                            reason_text = debug_info.get("reason", "")
                            thresh_pct = recognition_threshold * 100.0

                            if second_id:
                                second_display = f"**{second_id}** (`{second_sc_pct:.1f}%`) | Margin: `{margin_pct:.1f}%`"
                            else:
                                second_display = "None (single registered profile)"

                            if is_known:
                                status_header = f"🟢 **Face {idx + 1}: Recognized as {label}**"
                            else:
                                status_header = f"🔴 **Face {idx + 1}: Unknown Person**"

                            all_scores_str = ", ".join([f"{cand}: {s*100:.1f}%" for cand, s in candidate_scores.items()])

                            diag_card = (
                                f"{status_header}\n\n"
                                f"• **Best Match Identity:** **{best_id}** (Score: `{best_sc_pct:.1f}%`)\n\n"
                                f"• **Second-Best Identity:** {second_display}\n\n"
                                f"• **Threshold Used:** `{thresh_pct:.0f}%` (`{recognition_threshold:.2f}`)\n\n"
                                f"• **Decision / Reason:** {reason_text}\n\n"
                                f"• **All Candidates:** {all_scores_str if all_scores_str else 'None'}"
                            )
                            diagnostics_info.append(diag_card)

                        metric_known_count.metric(label="Recognized Faces", value=known_count)

                        # Update live diagnostics display
                        diag_text = (
                            f"**Registered Profiles:** {loaded_profiles_str}\n\n"
                            f"**Active Threshold:** `{recognition_threshold:.2f}`\n\n"
                            + "\n\n".join(diagnostics_info)
                        )
                        diag_box.markdown(diag_text)

                        # 3. Draw bounding boxes, badges, and names
                        annotated = draw_recognition_results(frame, recognitions)

                        # 4. Render annotated frame in Streamlit
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        time.sleep(0.02)

                finally:
                    release_camera(cap)

            # -------------------------------------------------------------
            # MODE 2: FACE REGISTRATION (Stage 3 & 5)
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
                            st.warning("⚠️ Failed to capture frame from webcam.")
                            break

                        faces = detect_faces(frame, face_cascade, scale_factor=scale_factor, min_neighbors=min_neighbors)
                        metric_face_count.metric(label="Faces in View", value=len(faces))

                        annotated_frame = frame.copy()

                        # Prevent saving when no face is detected
                        if len(faces) == 0:
                            status_feedback.warning("⚠️ No face detected! Please look directly at the webcam.")
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

                            # Throttle sample saves slightly (~120ms) for natural variations
                            current_time = time.time()
                            if current_time - last_save_time >= 0.12:
                                face_crop = crop_face(frame, largest_face, target_size=(160, 160))
                                save_face_sample(face_crop, reg_name, saved_count + 1)
                                saved_count += 1
                                last_save_time = current_time

                                if len(sample_previews) < 4:
                                    sample_previews.append(cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB))

                                reg_progress.progress(saved_count / total_target)
                                status_feedback.info(f"📸 Capturing sample **{saved_count}** of **{total_target}** for **{reg_name}**...")

                            annotated_frame = draw_faces(
                                frame,
                                [largest_face],
                                color=(0, 255, 0),
                                custom_label=f"{reg_name} ({saved_count}/{total_target})"
                            )

                        frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)

                        time.sleep(0.02)

                    # Finished registration
                    if saved_count >= total_target:
                        st.session_state["camera_active"] = False
                        st.session_state["mode"] = "Idle"
                        st.session_state["last_registered"] = {
                            "name": reg_name,
                            "count": saved_count,
                            "previews": sample_previews
                        }
                        st.session_state["status_message"] = f"✅ Successfully registered {reg_name}."

                        # Automatically update deep embeddings cache
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
            # MODE 3: LIVE FACE DETECTION ONLY (Stage 2 & 5)
            # -------------------------------------------------------------
            elif st.session_state["mode"] == "Face Detection":
                try:
                    while st.session_state.get("camera_active", False):
                        ret, frame = cap.read()
                        if not ret or frame is None:
                            st.warning("⚠️ Failed to capture frame from webcam. Stopping stream.")
                            break

                        faces = detect_faces(
                            frame,
                            face_cascade,
                            scale_factor=scale_factor,
                            min_neighbors=min_neighbors
                        )

                        annotated = draw_faces(frame, faces, color=(0, 255, 0), thickness=2)
                        frame_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                        video_placeholder.image(frame_rgb, channels="RGB", use_container_width=True)
                        metric_face_count.metric(label="Faces in View", value=len(faces))

                        time.sleep(0.02)

                finally:
                    release_camera(cap)

# Footer
st.markdown("---")
st.caption("Deep Learning Mini-Project | Final Stage 5: Integrated Face Detection & Recognition")
