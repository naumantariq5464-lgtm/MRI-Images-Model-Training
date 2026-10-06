# Official Python image use kar rahe hain
FROM python:3.11-slim

# Container ke andar hamari working directory
WORKDIR /app

# OpenCV (cv2) ko Linux (Docker) mein chalane ke liye zaroori system files
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Pehle sirf lightweight backend requirements copy karte hain
COPY requirements-docker.txt requirements.txt

# AI libraries install karna (with high timeout for slow networks)
RUN --mount=type=cache,target=/root/.cache/pip pip install --default-timeout=1000 --retries 5 -r requirements.txt

# Ab backend code aur trained model ko container mein copy karna
COPY backend/ ./backend/
COPY models/ ./models/

# Server ka port open karna
EXPOSE 8000

# Server ko start karne ki command
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
