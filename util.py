import cv2
import re
import csv
import numpy as np
from paddleocr import PaddleOCR

# === PaddleOCR инициализация ===
ocr = PaddleOCR(lang='en', use_textline_orientation=True)

# === Вспомогательные функции ===

def clean_license_text(text: str) -> str:
    """Очищает текст от мусора и нормализует формат."""
    if not text:
        return ""
    text = text.upper()
    text = re.sub(r'[^A-Z0-9]', '', text)  # только латиница и цифры
    return text.strip()


def is_valid_plate(text: str) -> bool:
    """
    Проверяет, что номер соответствует формату A111AA.
    Разрешены только латинские буквы и цифры.
    """
    if not text:
        return False
    text = text.upper()
    pattern = r'^[A-Z]{1}[0-9]{3}[A-Z]{2}$'
    return bool(re.match(pattern, text))


def format_time(seconds: float) -> str:
    """Форматирует секунды в ММ:СС.МС"""
    m = int(seconds // 60)
    s = int(seconds % 60)
    ms = int((seconds - int(seconds)) * 100)
    return f"{m:02d}:{s:02d}.{ms:02d}"


def preprocess_variants(img):
    """Создаёт несколько вариантов изображения для OCR."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    variants = []
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    base = clahe.apply(gray)
    variants.append(('clahe', base))

    _, otsu = cv2.threshold(base, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    variants.append(('otsu', otsu))

    adaptive = cv2.adaptiveThreshold(base, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                     cv2.THRESH_BINARY, 15, 3)
    variants.append(('adaptive', adaptive))

    inv = cv2.bitwise_not(otsu)
    variants.append(('inverted', inv))

    kernel = np.array([[-1, -1, -1],
                       [-1, 9, -1],
                       [-1, -1, -1]])
    sharp = cv2.filter2D(base, -1, kernel)
    variants.append(('sharp', sharp))

    contrast = cv2.convertScaleAbs(base, alpha=1.5, beta=10)
    contrast = cv2.resize(contrast, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(('contrast', contrast))

    variants = [(name, cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)) for name, v in variants]
    return variants


def read_license_plate(crop):
    """Распознаёт номер с помощью PaddleOCR и выбирает лучший результат."""
    try:
        if crop is None or crop.size == 0:
            return None, 0.0

        variants = preprocess_variants(crop)
        best_text, best_score = None, 0.0

        for name, img in variants:
            try:
                result = ocr.predict(img)
                if not result:
                    continue

                texts, scores = [], []
                for block in result:
                    if isinstance(block, dict):
                        if "rec_texts" in block and "rec_scores" in block:
                            texts.extend(block["rec_texts"])
                            scores.extend(block["rec_scores"])
                    elif isinstance(block, list):
                        for item in block:
                            if isinstance(item, list) and len(item) >= 2:
                                text, score = item
                                texts.append(str(text))
                                scores.append(float(score))

                for text, score in zip(texts, scores):
                    text = clean_license_text(text)
                    if not is_valid_plate(text):
                        continue
                    if score > best_score:
                        best_text, best_score = text, score
                        print(f"✅ [{name}] {text} ({score:.2f})")

            except Exception as e:
                print(f"[{name}] OCR error: {e}")
                continue

        return (best_text, best_score) if best_text else (None, 0.0)
    except Exception as e:
        print(f"OCR global error: {e}")
        return None, 0.0


def write_csv(results, output_path, fps=30):
    """
    Запись CSV в формате:
    time,plate_num
    """
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'plate_num'])

        for frame_nmr in results.keys():
            for car_id in results[frame_nmr].keys():
                plate_data = results[frame_nmr][car_id].get('license_plate', {})
                text = clean_license_text(plate_data.get('text', ''))

                # Только корректные номера
                if not is_valid_plate(text):
                    continue

                # Вычисляем время по номеру кадра
                time_sec = frame_nmr / fps
                time_str = format_time(time_sec)

                writer.writerow([time_str, text])


def get_car(license_plate, vehicle_track_ids):
    """Возвращает ID автомобиля, которому принадлежит номер."""
    x1, y1, x2, y2, score, class_id = license_plate
    for xcar1, ycar1, xcar2, ycar2, car_id in vehicle_track_ids:
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            return xcar1, ycar1, xcar2, ycar2, int(car_id)
    return -1, -1, -1, -1, -1
