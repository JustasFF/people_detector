# 👁️ People Detector with YOLO & Telegram Alerts

Автоматическая система обнаружения людей, кошек и собак с IP-камеры (RTSP-поток), отправляющая уведомления и GIF-анимации в Telegram.

## 🔧 Возможности

- 📷 Получение видеопотока с RTSP-камеры через `ffmpeg`
- 🧠 Обнаружение объектов с помощью модели YOLO (Ultralytics)
- 🚨 Отправка оповещений в Telegram с изображением и меткой времени
- 📦 Сохранение последних 10 изображений при обнаружении
- 🎞️ Отправка GIF из первых 5 кадров после детекции
- 🔁 Автоматическое переподключение при потере потока

## 📁 Структура проекта

```
people_detector/
├── main.py              # Основной скрипт
├── config.py            # Конфигурационные переменные
├── .env                 # Переменные окружения (Telegram токен и т.д.)
├── images/              # Папка с сохраненными кадрами
└── requirements.txt     # Зависимости Python
```

## ⚙️ Установка

1. Клонируй репозиторий:

```bash
git clone https://github.com/your-username/people_detector.git
cd people_detector
```

2. Установи зависимости:

```bash
pip install -r requirements.txt
```

3. Создай файл `.env`:

```env
TELEGRAM_TOKEN=your_telegram_bot_token
TELEGRAM_CHAT_ID=your_chat_id
YOLO_MODEL=yolov8n.pt
VIDEO_SOURCE=rtsp://000.000.000.000:000/stream ## свой видео поток
```

4. Запусти:

```bash
python main.py
```

## 🐳 Docker (опционально)

Убедись, что `ffmpeg` установлен в контейнере или добавлен в `Dockerfile`.

## 🔐 .gitignore пример

```
.env
__pycache__/
*.jpg
*.gif
.idea/
.vscode/
```

## 📝 Зависимости

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FFmpeg](https://ffmpeg.org/)
- [python-telegram-bot](https://github.com/python-telegram-bot/python-telegram-bot)
- OpenCV

## 📬 Автор

[JustasFF] — Telegram: [@JustasF16](https://t.me/JustasF16)
