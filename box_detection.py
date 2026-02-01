#!/bin/bash python3


import os
import cv2
import time
import numpy as np
import pandas as pd
from ultralytics import YOLO
from datetime import datetime
from collections import defaultdict

# ---------------- CONFIG ----------------------
INPUT_DIR = "CCTV"
OUTPUT_DIR = "detected_videos"
MODEL_PATH = "models/yolov8_box_310126.pt"
CSV_FILE = "box_count_report.csv"
LOG_FILE = "runtime_log.txt"

CONFIDENCE = 0.7
LOG_INTERVAL = 30
VIDEO_FORMATS = (".mp4", ".avi", ".mkv", ".mov")
SEGMENT_DURATION = 5 * 60  # 5 minutes in seconds
# --------------------------------------------------

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = YOLO(MODEL_PATH)


def list_videos(folder):
    return [
        os.path.join(folder, f)
        for f in os.listdir(folder)
        if f.lower().endswith(VIDEO_FORMATS)
    ]


def classify_size(value, min_v, max_v):
    ratio = (value - min_v) / (max_v - min_v + 1e-6)
    if ratio < 0.10:
        return "Small"
    elif ratio < 0.40:
        return "Medium"
    return "Large"


def process_video(video_path):
    WINDOW_NAME = "box_detection_ipcam"
    name = os.path.basename(video_path)
    name_without_ext = os.path.splitext(name)[0]
    
    cap = cv2.VideoCapture(video_path)
    w, h = int(cap.get(3)), int(cap.get(4))
    fps = cap.get(5)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if fps <= 0:
        fps = 30  # Default fallback FPS
    
    frames_per_segment = int(SEGMENT_DURATION * fps)
    segment_count = 0
    frame_count = 0
    writer = None
    segment_start_time = time.time()
    
    frame_area = w * h
    
    area_history = defaultdict(list)
    saved_ids = set()
    counters = {"Small": 0, "Medium": 0, "Large": 0}
    
    # Create tracker generator
    tracker = model.track(
        source=video_path,
        stream=True,
        persist=True,
        conf=CONFIDENCE,
        verbose=False
    )
    
    # Create subdirectory for this video's segments
    video_output_dir = os.path.join(OUTPUT_DIR, name_without_ext)
    os.makedirs(video_output_dir, exist_ok=True)

    for result in tracker:
        frame_count += 1
        
        # Check if we need to start a new segment
        if writer is None or frame_count % frames_per_segment == 1:
            if writer is not None:
                writer.release()
                print(f"\nCompleted segment {segment_count} at frame {frame_count}")
            
            segment_count += 1
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = os.path.join(video_output_dir, 
                                      f"{name_without_ext}_segment_{segment_count:03d}_{timestamp}.mp4")
            
            # Ensure video writer uses same dimensions as input
            writer = cv2.VideoWriter(
                output_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (w, h)
            )
            segment_start_time = time.time()
            print(f"\nStarted new segment {segment_count} -> {output_path}")
        
        frame = result.orig_img.copy()

        averages = {
            tid: np.mean(vals)
            for tid, vals in area_history.items()
            if len(vals) >= 5
        }

        if len(averages) >= 3:
            min_a, max_a = min(averages.values()), max(averages.values())
        else:
            min_a = max_a = None

        if result.boxes.id is not None:
            for box in result.boxes:
                tid = int(box.id)
                x1, y1, x2, y2 = map(int, box.xyxy[0])

                area = (x2 - x1) * (y2 - y1)
                area_history[tid].append(area / frame_area)

                label = "Detecting"
                if tid in averages and min_a is not None:
                    label = classify_size(averages[tid], min_a, max_a)

                if tid not in saved_ids and len(area_history[tid]) >= 10:
                    avg_area = float(np.mean(area_history[tid]))
                    final_size = "Unknown"

                    if min_a is not None:
                        final_size = classify_size(avg_area, min_a, max_a)
                        counters[final_size] += 1

                    pd.DataFrame([{
                        "box_id": tid,
                        "avg_norm_area": avg_area,
                        "size": final_size,
                        "video": name,
                        "segment": segment_count
                    }]).to_csv(
                        CSV_FILE,
                        mode="a",
                        header=not os.path.exists(CSV_FILE),
                        index=False
                    )

                    saved_ids.add(tid)
                
                # Draw bounding box
                color = (0, 255, 0)  # Green color for boxes
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Add segment info to display
                segment_info = f"Seg {segment_count}"
                cv2.putText(
                    frame,
                    segment_info,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )
                
                # Add box label
                cv2.putText(
                    frame,
                    f"ID:{tid} | {label}",
                    (x1, max(30, y1 - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )
        
        # Write frame to current segment
        if writer is not None:
            writer.write(frame)
        
        # Display frame 
        display_frame = cv2.resize(frame, (1280, 720))
        cv2.imshow(WINDOW_NAME, display_frame)
        
        # Log progress every LOG_INTERVAL frames
        if frame_count % LOG_INTERVAL == 0:
            elapsed_segment_time = time.time() - segment_start_time
            log_line = (
                f"{name},Segment{segment_count},"
                f"Frame:{frame_count},"
                f"Time:{elapsed_segment_time:.1f}s,"
                f"TotalBoxes:{len(saved_ids)},"
                f"S:{counters['Small']},M:{counters['Medium']},L:{counters['Large']}"
            )
            
            print(
                f"\r{name} | Seg:{segment_count} | Frame:{frame_count}/{total_frames} | "
                f"Time:{elapsed_segment_time:.1f}s | "
                f"Boxes:{len(saved_ids)} | "
                f"S:{counters['Small']} M:{counters['Medium']} L:{counters['Large']}",
                end=""
            )

            with open(LOG_FILE, "a") as f:
                f.write(log_line + "\n")
        
        # Break if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    
    # Release resources
    if writer is not None:
        writer.release()
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n\nVideo processing completed!")
    print(f"Total segments created: {segment_count}")
    print(f"Total boxes detected: {len(saved_ids)}")
    print(f"Size distribution: {counters}")
    print(f"Output saved to: {video_output_dir}")
    
    return segment_count


def main():
    video_files = list_videos(INPUT_DIR)
    print(f"Found {len(video_files)} videos to process")
    
    for i, video in enumerate(video_files, 1):
        print(f"\n{'='*60}")
        print(f"Processing video {i}/{len(video_files)}: {os.path.basename(video)}")
        print(f"{'='*60}")
        
        try:
            segments = process_video(video)
            print(f"Created {segments} segments for {os.path.basename(video)}")
        except Exception as e:
            print(f"Error processing {video}: {e}")
            continue


if __name__ == "__main__":
    main()
