import os
from dotenv import load_dotenv

load_dotenv()

VIDEO_SOURCE = os.getenv("VIDEO_SOURCE", "rtsp://192.168.1.120:64554/mpeg4")
YOLO_MODEL = os.getenv("YOLO_MODEL", "yolov8m.pt")
IMAGE_DIR = os.getenv("IMAGE_DIR", "images")
MAX_IMAGES = int(os.getenv("MAX_IMAGES", 10))
DETECTION_INTERVAL = int(os.getenv("DETECTION_INTERVAL", 60))  # секунд

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID")
ALLOWED_CLASSES = os.getenv("ALLOWED_CLASSES", "person,cat,dog").split(",")
