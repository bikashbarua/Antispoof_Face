import os
import cv2

# Set paths
video_folder = "dataset"  # Folder where .avi videos are located
output_folder = "dataset/train/live_frames"  # Folder to save extracted frames
frame_gap = 5  # Save every 5th frame, to reduce redundancy

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

# Iterate over all .avi files
video_files = [f for f in os.listdir(video_folder) if f.endswith(".avi")]
frame_count = 0

for video_file in video_files:
    video_path = os.path.join(video_folder, video_file)
    cap = cv2.VideoCapture(video_path)
    print(f"🎞️ Extracting from {video_file}...")

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Save every Nth frame
        if frame_idx % frame_gap == 0:
            frame_name = f"{os.path.splitext(video_file)[0]}_frame{frame_count}.jpg"
            save_path = os.path.join(output_folder, frame_name)
            cv2.imwrite(save_path, frame)
            frame_count += 1

        frame_idx += 1

    cap.release()
    print(f"✅ Done with {video_file} — {frame_count} frames saved so far.")

print("🎉 All videos processed!")
