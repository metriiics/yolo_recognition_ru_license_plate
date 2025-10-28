from ultralytics import YOLO
from paddleocr import PaddleOCR
import cv2
import numpy as np
from sort.sort import Sort
from util import get_car, write_csv
import time

VIDEO_PATH = "Video-1.mp4"
OUTPUT_CSV = "output.csv"

coco_model = YOLO("yolov8n.pt")              # Обнаружение машин
license_plate_detector = YOLO("./models/best.pt")  # Обнаружение номеров

mot_tracker = Sort()

ocr = PaddleOCR(use_textline_orientation=True, lang='en')

vehicles = [2, 3]  # car, motorcycle
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)

results = {}
frame_nmr = 0
start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_nmr += 1

    if frame_nmr % 2 != 0:
        continue  # пропускаем кадры для скорости

    results[frame_nmr] = {}

    # Детектим транспорт
    detections = coco_model(frame, verbose=False)[0]
    detections_ = []
    for detection in detections.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = detection
        if int(class_id) in vehicles:
            detections_.append([x1, y1, x2, y2, score])

    # Трекинг машин
    track_ids = mot_tracker.update(np.asarray(detections_) if len(detections_) > 0 else np.empty((0, 5)))

    # Детектим номера
    license_plates = license_plate_detector(frame, verbose=False)[0]

    for license_plate in license_plates.boxes.data.tolist():
        x1, y1, x2, y2, score, class_id = license_plate

        # Привязываем номер к машине
        xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
        if car_id == -1:
            continue

        # Кадрируем номер
        crop = frame[int(y1):int(y2), int(x1):int(x2)]
        if crop.size == 0:
            continue

        # Распознаем текст через PaddleOCR
        try:
            ocr_res = ocr.predict(crop)
            texts = []
            for block in ocr_res:
                if isinstance(block, dict) and "rec_texts" in block:
                    texts.extend(block["rec_texts"])
                elif isinstance(block, list):
                    texts.extend(block)
            text = " ".join(texts).strip() if texts else "UNREADABLE"

        except Exception as e:
            print(f"[OCR error]: {e}")
            text = "UNREADABLE"

        results[frame_nmr][car_id] = {
            "car": {"bbox": [float(xcar1), float(ycar1), float(xcar2), float(ycar2)]},
            "license_plate": {
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "text": text,
                "bbox_score": float(score),
                "text_score": 1.0 if text != "UNREADABLE" else 0.0
            }
        }

    if frame_nmr % 20 == 0:
        elapsed = time.time() - start_time
        print(f"[{frame_nmr}] обработано кадров за {elapsed:.1f} сек.")

cap.release()
write_csv(results, OUTPUT_CSV)
print(f"\n✅ Сохранено в {OUTPUT_CSV}")
