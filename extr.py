import cv2
import os

video_path = "dataset/1b.mp4"  # <- your new video path
output_folder = "dataset/new_frames/live"  # or "spoof" if it's a spoof video

os.makedirs(output_folder, exist_ok=True)

cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
frame_interval = int(fps)  # one frame per second

frame_count = 0
saved_count = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    if frame_count % frame_interval == 0:
        filename = f"{saved_count:04d}.jpg"
        filepath = os.path.join(output_folder, filename)
        cv2.imwrite(filepath, frame)
        print(f"Saved {filename}")
        saved_count += 1

    frame_count += 1

cap.release()
print(f"✅ Done! Extracted {saved_count} frames.")
