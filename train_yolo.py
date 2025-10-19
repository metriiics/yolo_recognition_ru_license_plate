from ultralytics import YOLO

# Загружаем базовую YOLOv8n модель
model = YOLO("yolov8n.pt")

# Обучаем на наших данных
model.train(
    data="data/data.yaml",
    epochs=50,
    imgsz=640,
    batch=16,
    name="ru_license_plate",
    workers=4
)
