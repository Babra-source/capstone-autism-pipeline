import os
import cv2
import numpy as np

from data.casme2_dataset import CASME2Dataset
from data.eye_crop import crop_eye_region_from_frames

DEFAULT_DATA_DIR   = "Autism/data/CASME2_Compressed_video/CASME2_compressed"
DEFAULT_LABEL_FILE = "Autism/data/CASME2-ObjectiveClasses.xlsx"


def main():
    print(f"Loading dataset from {DEFAULT_DATA_DIR} ...")
    
    if not os.path.exists(DEFAULT_LABEL_FILE):
        print(f"Label file not found at: {DEFAULT_LABEL_FILE}")
        print("Please run this script on the server where the 'Autism/data/...' folder is located.")
        return
        
    ds = CASME2Dataset(
        root_dir=DEFAULT_DATA_DIR,
        label_file=DEFAULT_LABEL_FILE,
        mode='sequence',
        split='train',
        max_frames=1
    )
    
    if len(ds.samples) == 0:
        print("Dataset is empty. Cannot visualize.")
        return

    save_dir = "results/eye_crop_videos"
    os.makedirs(save_dir, exist_ok=True)

    num_samples = 5
    valid_count = 0
    
    for sample in ds.samples:
        if valid_count >= num_samples:
            break
            
        vid_path = sample['video']
        filename = sample['filename']
        
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            print(f"Skipping {filename} (cannot open video)")
            continue
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps == 0 or np.isnan(fps):
            fps = 30.0
            
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            # Convert BGR → RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            
        cap.release()
        
        if not frames:
            print(f"Skipping {filename} (no frames)")
            continue
            
        # Extract eye regions
        eye_frames = crop_eye_region_from_frames(frames)
        
        if not eye_frames:
            print(f"Skipping {filename} (no eye frames detected)")
            continue
        
        save_path = os.path.join(save_dir, f"eye_crop_{filename}.mp4")
        
        # Store only eye frames
        combined_frames = []
        
        for eye_frame in eye_frames:
            # Resize for better visibility (optional)
            eye_display = cv2.resize(eye_frame, (400, 200))
            
            # Convert RGB → BGR for saving
            combined_frames.append(cv2.cvtColor(eye_display, cv2.COLOR_RGB2BGR))

        if not combined_frames:
            print(f"Skipping {filename} (empty processed frames)")
            continue
            
        h, w = combined_frames[0].shape[:2]
        
        # Save as MP4
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        
        for frame_bgr in combined_frames:
            out.write(frame_bgr)
            
        out.release()
        
        print(f"✓ Saved eye-only video to {save_path} ({len(combined_frames)} frames @ {fps} FPS)")
        valid_count += 1

    print(f"\nDone! Eye-only videos saved in '{save_dir}'.")


if __name__ == "__main__":
    main()