from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2, pandas as pd

VIDEO_PATH = "video.mp4"
OUTPUT_CSV = "output.csv"

yolo = YOLO("./runs/detect/ru_license_plate2/weights/best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

detections = []
frame_id = 0

def format_time(seconds):
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 100)
    return f"{m:02d}:{s:02d}.{ms:02d}"

while True:
    ret, frame = cap.read()
    if not ret:
        break
    time_now = frame_id / fps
    results = yolo(frame, verbose=False)
    for r in results:
        for box in r.boxes.xyxy:
            x1, y1, x2, y2 = map(int, box)
            crop = frame[y1:y2, x1:x2]
            ocr_res = ocr.ocr(crop, cls=True)
            if ocr_res and ocr_res[0]:
                text = ocr_res[0][0][1][0]
                detections.append({
                    "time": format_time(time_now),
                    "plate_num": text
                })
                print(f"[{format_time(time_now)}] {text}")
    frame_id += 1

cap.release()
pd.DataFrame(detections).to_csv(OUTPUT_CSV, index=False)
print(f"\n✅ Saved results to {OUTPUT_CSV}")
