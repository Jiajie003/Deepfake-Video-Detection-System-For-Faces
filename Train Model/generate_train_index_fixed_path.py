import os
import json

frame_root = "frames"
output_json = "self train/train_index.json"

train_data = []

for folder in os.listdir(frame_root):
    folder_path = os.path.join(frame_root, folder)
    if not os.path.isdir(folder_path):
        continue
    if not folder.endswith("_face"):
        continue

    video_id = folder.replace("_face", "")
    label = 0 if video_id.count("_") == 1 else 1  # one underscore = REAL, more = FAKE

    frames = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png"))
    ])

    if len(frames) < 20:
        print(f"⚠️ Skipping {folder}: less than 20 frames found")
        continue

    train_data.append({
        "video_id": video_id,
        "frame_dir": folder_path.replace("\\", "/"),
        "frames": frames,
        "label": label
    })

with open(output_json, 'w') as f:
    json.dump(train_data, f, indent=2)

print(f"✅ Saved {len(train_data)} entries to {output_json}")
