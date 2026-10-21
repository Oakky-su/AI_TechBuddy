# ---- Base image ----
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# ติดตั้ง PyTorch รุ่น CPU ก่อน — สำคัญมาก
# ถ้าปล่อยให้ pip เลือกเอง มันจะลาก torch รุ่น CUDA มาพร้อมแพ็กเกจ nvidia-* รวม ~2.5 GB
# (cudnn, cublas, nccl, cusolver ฯลฯ) ซึ่งโปรเจคนี้ไม่ได้ใช้เลยเพราะรันบน CPU ล้วน
# ผลคือ image พองเป็น ~16 GB และ build นานเป็นชั่วโมงเพราะมัวแต่โหลด
# --extra-index-url pypi: แพ็กเกจประกอบบางตัวของ torch รุ่นใหม่ไม่มีในคลัง PyTorch
#   ถ้าไม่ใส่ pip จะถอยไปโหลด torch รุ่นเก่าทีละรุ่น (ไฟล์ละ ~190 MB) จนหมดเวลา
#   (torch ยังได้รุ่น +cpu เพราะเลขเวอร์ชัน +cpu ถือว่าใหม่กว่าตัวบน PyPI)
RUN pip install --no-cache-dir --timeout 120 --retries 10 \
    --index-url https://download.pytorch.org/whl/cpu \
    --extra-index-url https://pypi.org/simple torch

# ติดตั้ง dependencies ที่เหลือ (ใช้ layer cache — โค้ดเปลี่ยนแล้วไม่ต้องติดตั้งใหม่)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# โหลดโมเดล embedding เข้า image เลย เพื่อให้ตอนรันไม่ต้องรอดาวน์โหลด ~2 GB
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# copy โค้ด + laptops.json
COPY . .

EXPOSE 5000

# gunicorn — 1 worker หลายเธรด (โมเดลกินแรม + session secret ต้องอยู่ process เดียว)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
