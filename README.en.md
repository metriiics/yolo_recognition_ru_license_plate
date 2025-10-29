# License Plate Recognition with YOLO

**Project for the International Digital Olympiad "Volga-IT'2025" in the direction "Artificial Intelligence and Data Analysis"**  
[Official Website](https://volga-it.org/)

This project detects vehicles and reads license plates from video footage using **YOLOv8**, **SORT**, and **PaddleOCR**.


## Description

- **Vehicle detection** — YOLOv8 pre-trained model (`yolov8n.pt`)
- **License plate detection** — custom-trained YOLO model (`best.pt`)
- **Tracking** — SORT algorithm (Kalman filter based)
- **Text recognition** — PaddleOCR (English)


## Dataset

The model for license plate detection was trained on a dataset from [Roboflow Universe](https://universe.roboflow.com/science-ffxxt/russian-license-plates-detector-vcua6).


## Dependencies

This project includes components from the [SORT Tracker](https://github.com/abewley/sort).


## Requirements

- Python 3.9+
- OpenCV
- ultralytics
- paddleocr
- numpy
- filterpy


## Run step by step

1. **Clone the repository**
   ```bash
   git clone https://github.com/metriiics/yolo_recognition_ru_license_plate.git
   cd yolo_recognition_ru_license_plate

2. Create and activate a virtual environment
    ```bash
    python -m venv venv
    venv\Scripts\activate

3. Install dependencies
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt

4. Run the program
    ```bash
    python main.py


## Documentation

- [Explanatory Note](explanatory_note.md)