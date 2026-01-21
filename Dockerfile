# 1. Base Image: Python 3.11 Slim (เบาและเร็ว)
FROM python:3.11-slim

# 2. Install System Dependencies (จำเป็นสำหรับ OpenCV/Pyzbar)
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    libzbar0 \
    && rm -rf /var/lib/apt/lists/*

# 3. Setup Working Directory
WORKDIR /app

# 4. Install Python Libraries (CPU Optimized for PyTorch)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cpu

# 5. Copy Code & Model
COPY main.py .
COPY best.pt .

# 6. Expose Port 8000 (สำหรับ API)
EXPOSE 8000

# 7. Start API Server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]