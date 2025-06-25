import os
import cv2
import time
import datetime
import subprocess
import numpy as np
import imageio
import asyncio

from ultralytics import YOLO
from telegram import Bot
from telegram.constants import ParseMode
from config import *

# Инициализация
bot = Bot(token=TELEGRAM_TOKEN)
model = YOLO(YOLO_MODEL)
os.makedirs(IMAGE_DIR, exist_ok=True)

last_notification_time = 0

def log(msg):
    print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {msg}")

def start_ffmpeg_process():
    cmd = [
        r"c:\Users\User\Documents\ffmpeg\ffmpeg.exe",
        "-rtsp_transport", "tcp",
        "-i", VIDEO_SOURCE,
        "-vf", "scale=1280:720",
        "-f", "image2pipe",
        "-pix_fmt", "bgr24",
        "-vcodec", "rawvideo",
        "-"
    ]
    return subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)

def read_frame(pipe, width=1280, height=720):
    raw_size = width * height * 3
    raw_frame = pipe.stdout.read(raw_size)
    if len(raw_frame) != raw_size:
        return None
    frame = np.frombuffer(raw_frame, dtype=np.uint8).reshape((height, width, 3)).copy()
    if frame.flags['WRITEABLE'] == False:
        frame = frame.copy()
        log("🛠️ Копируем кадр из-за readonly")
    return frame

def clean_old_images():
    files = sorted(
        [os.path.join(IMAGE_DIR, f) for f in os.listdir(IMAGE_DIR)],
        key=os.path.getmtime,
        reverse=True
    )
    for file in files[MAX_IMAGES:]:
        os.remove(file)

def detect_objects(frame):
    results = model(frame, verbose=False)[0]
    detections = []
    for box in results.boxes:
        cls = model.names[int(box.cls)]
        if cls in ALLOWED_CLASSES:
            detections.append({
                "class": cls,
                "box": box.xyxy[0].cpu().numpy().astype(int)
            })
    return detections

def save_frame_image(frame, frame_index=5):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = os.path.join(IMAGE_DIR, f"detected_{timestamp}_frame{frame_index}.jpg")
    cv2.imwrite(filename, frame)
    clean_old_images()
    return filename

def save_gif(frames):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(IMAGE_DIR, f"alert_{timestamp}.gif")
    rgb_frames = [cv2.cvtColor(f, cv2.COLOR_BGR2RGB) for f in frames]
    imageio.mimsave(gif_path, rgb_frames, duration=0.1)
    clean_old_images()
    return gif_path

async def send_telegram_alert_gif(gif_path, detected_class, img_path):
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    caption = f"🚨 Обнаружен: *{detected_class}*\n🕒 {now}"
    with open(gif_path, "rb") as gif:
        await bot.send_animation(chat_id=TELEGRAM_CHAT_ID, animation=gif, caption=caption, parse_mode=ParseMode.MARKDOWN)
    log(f"✅ Отправлено: {detected_class}, кадр: {img_path}")

async def main_loop():
    global last_notification_time
    log("🚀 Запуск детектора через ffmpeg...")

    pipe = start_ffmpeg_process()
    width, height = 1280, 720

    while True:
        frame = read_frame(pipe, width, height)
        if frame is None:
            log("⚠️ Кадр не получен, перезапуск ffmpeg...")
            pipe.kill()
            time.sleep(2)
            pipe = start_ffmpeg_process()
            continue

        detections = detect_objects(frame)
        now = time.time()

        if detections and now - last_notification_time > DETECTION_INTERVAL:
            label = detections[0]["class"]
            log(f"📸 Детектировано: {label}, начинаем сбор кадров...")

            frames = [frame.copy()]
            for _ in range(9):
                next_frame = read_frame(pipe, width, height)
                if next_frame is not None:
                    frames.append(next_frame.copy())
                time.sleep(0.1)

            # Добавим подписи и рамки на кадры
            for i, f in enumerate(frames):
                cv2.putText(f, f"{label}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
                now_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                cv2.putText(f, now_str, (10, f.shape[0] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            img_path = save_frame_image(frames[4], frame_index=5)
            gif_path = save_gif(frames)
            await send_telegram_alert_gif(gif_path, label, img_path)
            last_notification_time = now

if __name__ == "__main__":
    asyncio.run(main_loop())
