# Распознавание автомобильных номеров с помощью YOLO

**Проект для Международной цифровой олимпиады «Волга-IT'2025» по направлению «Искусственный интеллект и анализ данных»**  
[Официальный сайт](https://volga-it.org/)

Этот проект обнаруживает транспортные средства и считывает номерные знаки с видеозаписей с использованием **YOLOv8**, **SORT** и **PaddleOCR**.

## Описание

- **Детекция транспортных средств** — предобученная модель YOLOv8 (`yolov8n.pt`)
- **Детекция номерных знаков** — кастомно обученная YOLO модель (`best.pt`)
- **Трекинг** — алгоритм SORT (на основе фильтра Калмана)
- **Распознавание текста** — PaddleOCR (английский)


## Датасет

Модель для детекции номерных знаков была обучена на датасете с [Roboflow Universe](https://universe.roboflow.com/science-ffxxt/russian-license-plates-detector-vcua6).


## Зависимости

Этот проект включает компоненты из [SORT Tracker](https://github.com/abewley/sort).

## Требования

- Python 3.9+
- OpenCV
- ultralytics
- paddleocr
- numpy
- filterpy

## Запуск по шагам

1. **Клонируйте репозиторий**
   ```bash
   git clone https://github.com/metriiics/yolo_recognition_ru_license_plate.git
   cd yolo_recognition_ru_license_plate

2. **Создайте и активируйте виртуальное окружение**
    python -m venv venv
    venv\Scripts\activate

3. **Установите зависимости**
    pip install --upgrade pip
    pip install -r requirements.txt

4. **Запустите программу**
    python main.py


## Документация

- [Пояснительная записка](explanatory_note.md)