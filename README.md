# cv_task

## Requirements
- Python 3.8+
- ultralytics
- opencv-python
- numpy
- pandas

```bash
pip3 install ultralytics opencv-python numpy pandas

1.# CV Task – CCTV Box Detection

YOLO-based computer vision pipeline to detect, track, and classify boxes
(Small / Medium / Large) from CCTV videos and save annotated 5-minute segments.

## **Approach**

1. **Detection:**  
   - YOLO pre-trained model is used for object detection.  
   - Each frame is processed to detect boxes automatically.

2. **Size Estimation:**  
   - Bounding box area in pixels is used to classify sizes:  

3. **Reporting:**  
   - CSV report (`box_count_report.csv`) contains bounding box coordinates, area, and size.  
   - Total box count and size distribution are computed automatically.


2.Current Result Example
Video: 00000000777000000.mp4
Segment: 28
Frame: 249,390
Total Boxes: 4,183
Size Distribution:
  Small: 420
  Medium: 3,593
  Large: 168

saple_video_result = https://drive.google.com/drive/folders/13lV00MTfMCiY1_UiXEcCXUTF_u8al08B


3.for improve this box detection & counting pipeline we should use custum box detection model(to dectect all boxes) and Large number of boxes suggests
many repeated detections / possible ID switches, which can be improved using some data augmentation to handle different angles, lighting, occlusion.
at last  we can use custom YOLO with Deep SORT or ByteTrack lets us track boxes accurately, avoid duplicates, and get better size counts.
for code read Adhere to PEP 8 things etc.

4. CCTV Box Detection – Production Plan:
Multi-Camera Streams: Connect via RTSP; track boxes per stream using Deep SORT / ByteTrack.
Containerized Deployment: Package pipeline in Docker
Output & Monitoring: Save annotated segments and CSV stats; centralize logs.
Performance Optimization: I'll use TensorRT in production for high FPS, low GPU usage, and efficient tracking/counting.
