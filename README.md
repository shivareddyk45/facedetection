# 👤 Face Detection and Recognition System
### A Deep Learning & Computer Vision Mini-Project

---

## 📌 1. Project Objective

The objective of this project is to build an interactive, real-time **Face Detection and Recognition System** using **Deep Learning** and **Computer Vision**. The application allows users to:
1. **Detect Faces in Real Time**: Locate single or multiple human faces in live webcam streams using an OpenCV Haar Cascade classifier.
2. **Register New Faces**: Build an organized face dataset by capturing multiple face crops for named individuals while verifying face presence.
3. **Recognize Registered Individuals**: Extract 128-dimensional deep feature embeddings using a pre-trained deep convolutional neural network (**SFace**) and match live faces using **Cosine Similarity**.
4. **Identify Unknown Persons**: Accurately classify unregistered or low-confidence faces as **"Unknown Person"**.
5. **Interactive Dashboard**: Provide an intuitive web-based interface using **Streamlit** for college mini-project demonstrations.

---

## 🛠️ 2. Technologies & Libraries Used

| Technology / Library | Purpose in Project |
| :--- | :--- |
| **Python (3.9+)** | Core programming language for logic, processing, and pipeline scripting. |
| **OpenCV (`opencv-python`)** | Camera video stream capture, image preprocessing, Haar Cascade detection, and DNN module execution. |
| **Streamlit** | Web application framework providing the real-time graphical user interface, controls, and dashboard metrics. |
| **SFace Deep Neural Network** | Pre-trained deep convolutional face recognition model producing 128-D $L_2$-normalized embeddings. |
| **NumPy** | High-performance vector math for $L_2$ vector normalization and cosine similarity computation. |
| **Pillow (PIL)** | Image conversion and support for multi-channel RGB stream rendering. |

---

## 📁 3. Project Structure

```text
deep learning/
│
├── app.py                  # Main Streamlit web application & user interface
├── requirements.txt        # Python package dependencies
├── README.md               # Complete project documentation & guide
│
├── data/
│   └── faces/              # Organized dataset: subdirectories per individual (e.g. data/faces/Alice/)
│
├── models/
│   ├── face_recognition_sface_2021dec.onnx   # Pre-trained deep learning face recognition model (SFace)
│   └── embeddings.pkl                        # Cached prototype face vectors for instant matching
│
└── src/
    ├── __init__.py         # Python package initializer
    └── face_utils.py       # Helper functions: detection, cropping, embeddings, and matching
```

---

## 🚀 4. Installation Steps

### Step 1: Clone or Open the Project Directory
Navigate to the root directory in your terminal:
```bash
cd "/Users/kalagotlashivareddy/Desktop/deep learning"
```

### Step 2: Create and Activate a Python Virtual Environment

**On macOS / Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Required Dependencies
```bash
pip install -r requirements.txt
```

---

## ▶️ 5. Execution Command

Run the application using Streamlit:
```bash
streamlit run app.py
```

The application will launch automatically in your default web browser at `http://localhost:8501`.

---

## ⚙️ 6. Working Procedure & Demonstration Steps

### Step 1: Real-Time Face Detection
1. In the sidebar, click **`▶️ Face Detection Only`**.
2. The webcam activates, showing a green bounding box with labels around any detected face.
3. The dashboard metrics update with the number of faces in view.
4. Click **`⏹️ Stop Camera`** to turn off the camera.

### Step 2: Registering a Person
1. In the sidebar under **👤 Register New Person**, enter the individual's name (e.g., `Alice`).
2. Set the desired number of samples (default: `20`).
3. Click **`➕ Register Face`**.
4. Look directly at the webcam. The system validates that a face is present before saving each sample into `data/faces/Alice/`.
5. Once 20 samples are captured, the camera safely stops, a **Registration Successful** card appears with thumbnail previews, and the prototype embedding vector is automatically computed.

### Step 3: Recognizing Registered Faces
1. Click **`🔍 Recognize`** in the sidebar.
2. The system captures live frames, extracts 128-D face embeddings, and calculates cosine similarity against the registered prototypes:
   - **Registered User (e.g., Alice):** Bounding box is **Green** displaying the name and confidence percentage (e.g., `Alice (87.5%)`).
   - **Unregistered Individual / Stranger:** Bounding box is **Red** displaying **`Unknown Person`**.
3. **Multiple People:** If multiple people appear together, each face is independently detected, embedded, and classified in real time.

### Step 4: Fine-Tuning & Management
- **Confidence Threshold Slider:** Adjust the threshold (calibrated default: `0.50`, recommended range: `0.48 - 0.55`) in **🔧 Model Tuning & Thresholds**.
- **Ambiguity Rejection Margin:** Adjust candidate separation margin (default: `0.08`) to reject ambiguous matches between close candidate profiles.
- **Sync Embeddings:** Click **`🔄 Sync`** to recalculate prototype embeddings across all registered dataset folders.
- **Remove a Person:** Select a registered person from the dropdown under **🗑️ Remove a Person** and click delete to remove their data.

---

## 🧠 7. Technical Workflow & Deep Learning Concepts

```text
[Webcam Input] ──► [Haar Cascade Detection] ──► [Face ROI Crop (112x112)]
                                                           │
                                                           ▼
                                                [SFace Deep Neural Network]
                                                           │
                                                           ▼
                                                [128-D Normalized Embedding]
                                                           │
                                                           ▼
                                           [Calibrated Cosine Matching & Ambiguity Check]
                                                ┌──────────┴──────────┐
                                                ▼                     ▼
                                  Score >= 0.50 & Separated      Score < 0.50 or Ambiguous
                                        [Known Person]                [Unknown Person]
                                         (Green Badge)                  (Red Badge)
```

1. **Face Detection (Haar Cascades):**
   - The video frame is converted to grayscale (`cv2.cvtColor`).
   - `detectMultiScale` scans across the image using edge, line, and four-rectangle features to identify face candidate bounding boxes `(x, y, w, h)`.
2. **Face Cropping & Preprocessing:**
   - The detected bounding box is cropped and resized to $112 \times 112$ pixels.
   - It is converted into a 4-dimensional image blob via `cv2.dnn.blobFromImage`.
3. **Feature Extraction (Deep Embeddings):**
   - The pre-trained **SFace ONNX model** maps the face image into a compact **128-dimensional numerical vector**.
   - The vector is $L_2$-normalized such that $\|\mathbf{e}\|_2 = 1.0$.
4. **Multi-Sample & Prototype Matching:**
   - Outliers (such as overexposed, motion-blurred, or non-face frames) are filtered out during embedding generation.
   - Both high-quality sample embeddings and the $L_2$-normalized centroid mean vector $\mathbf{e}_{\text{centroid}}$ are maintained.
   - Live query face vector $\mathbf{e}_{\text{query}}$ is compared using a hybrid score:
     $$\text{Score} = 0.5 \cdot (\text{mean of top 3 sample similarities}) + 0.5 \cdot (\mathbf{e}_{\text{query}} \cdot \mathbf{e}_{\text{centroid}})$$
5. **Decision & Ambiguity Rejection Rules:**
   - If $\max(\text{Score}) < \text{Threshold}$ ($0.50$), the face is classified as **"Unknown Person"**.
   - If multiple candidates exist and the score separation between the top two candidates is less than the ambiguity margin ($\Delta < 0.08$), the identity is rejected as **"Unknown Person"** to prevent guessing.
   - Otherwise, the face is identified as the recognized person with green badge `{Name} ({Confidence:.1f}%)`.

---

## ❓ 8. Troubleshooting & FAQ

- **Webcam Unavailable Error:** Ensure no other application (Zoom, FaceTime) is using the camera, and grant camera permissions to your terminal/browser in System Settings. If using an external USB camera, change the **Camera Index** from `0` to `1` in the sidebar.
- **Lighting and Angles:** For best recognition accuracy, register face samples in good lighting with slight natural head movements.
