import os
import cv2
import numpy as np
from tqdm import tqdm

input_dirs = {
    "real": "../dataset/real",
    "fake": "../dataset/fake"
}
output_root = "frames"
frames_per_video = 20
output_size = (224, 224)

os.makedirs(output_root, exist_ok=True)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

def extract_faces(video_path, out_dir, num_frames=20):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open {video_path}")
        return False

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total < num_frames:
        print(f"⚠️ Not enough frames in {video_path}")
        return False

    indices = np.linspace(0, total - 1, num_frames * 2, dtype=int)  # search extra in case of misses
    os.makedirs(out_dir, exist_ok=True)

    count = 0
    current_frame = 0
    collected_indices = []

    for i in range(total):
        ret, frame = cap.read()
        if not ret:
            break
        if i in indices:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
            if len(faces) == 0:
                continue
            x, y, w, h = max(faces, key=lambda b: b[2] * b[3])
            face_crop = frame[y:y+h, x:x+w]
            face_resized = cv2.resize(face_crop, output_size)
            out_path = os.path.join(out_dir, f"frame_{count:03d}.jpg")
            cv2.imwrite(out_path, face_resized)
            count += 1
            if count >= num_frames:
                break

    cap.release()
    return count == num_frames

video_list = []
for label, folder in input_dirs.items():
    for file in os.listdir(folder):
        if file.lower().endswith((".mp4", ".avi", ".mov")):
            video_list.append((label, folder, file))

for idx, (label, folder, file) in enumerate(tqdm(video_list, desc="Extracting face frames", unit="video")):
    video_path = os.path.join(folder, file)
    video_id = os.path.splitext(file)[0]
    out_dir = os.path.join(output_root, f"{video_id}_face")
    success = extract_faces(video_path, out_dir, frames_per_video)
    if success:
        tqdm.write(f"[{idx+1}/{len(video_list)}] ✅ Extracted {frames_per_video} face frames from {file}")
    else:
        tqdm.write(f"[{idx+1}/{len(video_list)}] ⚠️ Skipped {file} (insufficient face frames)")
