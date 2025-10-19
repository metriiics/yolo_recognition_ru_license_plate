from ultralytics import YOLO
#from paddleocr import PaddleOCR
import cv2
import pandas as pd

VIDEO_PATH = "video.mp4"
OUTPUT_CSV = "results/output.csv"

yolo = YOLO("models/ru_plate_best.pt")
ocr = PaddleOCR(use_angle_cls=True, lang='en')

cap = cv2.VideoCapture(VIDEO_PATH)