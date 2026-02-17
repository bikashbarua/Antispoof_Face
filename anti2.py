import os
import cv2
import numpy as np
import time
from PIL import Image
import torch
import torchvision.transforms as transforms
import onnxruntime as ort
from mtcnn.mtcnn import MTCNN
from insightface.app import FaceAnalysis
from sklearn.metrics.pairwise import cosine_similarity

# -------- Anti-spoofing ONNX model setup --------
try:
    anti_spoof_sess = ort.InferenceSession("anti_spoof_model2.onnx")
    input_name = anti_spoof_sess.get_inputs()[0].name
    print(" Anti-spoof model loaded successfully.")
except Exception as e:
    print(f" Failed to load ONNX model: {e}")
    exit()

# -------- Transform for ONNX model (MUST match training) --------
transform = transforms.Compose([
    transforms.Resize((112, 112)),  # ❗ ONNX model expects 112x112 input
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# -------- Initialize Face Detection & Embedding --------
detector = MTCNN()
app = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# -------- Load dataset embeddings for face recognition --------
def load_dataset_embeddings(dataset_path):
    known_embeddings = []
    known_labels = []

    for person_name in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_name)
        if not os.path.isdir(person_dir):
            continue

        for img_name in os.listdir(person_dir):
            img_path = os.path.join(person_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = app.get(img_rgb)
            if faces:
                emb = faces[0]['embedding']
                known_embeddings.append(emb)
                known_labels.append(person_name)
            else:
                print(f"❌ No face detected in {img_path}")

    return known_embeddings, known_labels

# -------- Load known faces --------
dataset_path = r"D:\dataset1"  
known_embeddings, known_labels = load_dataset_embeddings(dataset_path)

# -------- Webcam live processing --------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Cannot access webcam.")
    exit()

print("🎥 Webcam started. Press 'q' to quit.")

while True:
    start_time = time.time()
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame")
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = app.get(frame_rgb)

    for face in faces:
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox
        emb = face['embedding']

        # Crop and preprocess face for anti-spoofing
        face_img = frame[y1:y2, x1:x2]
        if face_img.size == 0:
            continue

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(face_rgb)
        input_tensor = transform(pil_img).unsqueeze(0).numpy()

        # Run anti-spoofing inference
        output = anti_spoof_sess.run(None, {input_name: input_tensor})
        score = output[0][0][0]
        spoof_label = "Live" if score < 0.7 else "Spoof"

        # Run face recognition
        identity = "Unknown"
        if emb is not None and known_embeddings:
            sims = cosine_similarity([emb], known_embeddings)[0]
            idx = np.argmax(sims)
            if sims[idx] > 0.5:
                identity = known_labels[idx]

        # Combine labels and draw box
        display_text = f"{identity} ({spoof_label})"
        color = (0, 255, 0) if spoof_label == "Live" else (0, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.putText(frame, display_text, (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # Show FPS
    fps = 1 / (time.time() - start_time)
    cv2.putText(frame, f"FPS: {fps:.2f}", (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # Display output
    cv2.imshow("Real-Time Face Recognition + AntiSpoofing", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# -------- Cleanup --------
cap.release()
cv2.destroyAllWindows()
