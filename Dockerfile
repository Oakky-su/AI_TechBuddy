# ---- Base image ----
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

# ติดตั้ง dependencies ก่อน (ใช้ layer cache — โค้ดเปลี่ยนแล้วไม่ต้องติดตั้งใหม่)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ดาวน์โหลดโมเดล embedding bge-m3 (~2GB) ฝังใน image เลย (กันการโหลดตอน runtime)
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('BAAI/bge-m3')"

# copy โค้ด + laptops.json
COPY . .

# สร้าง Chroma vector index ตอน build เลย (image พร้อมใช้งานทันที ไม่ต้อง embed ตอนรัน)
RUN python build_index.py

EXPOSE 5000

# gunicorn — 1 worker หลายเธรด (โมเดลกินแรม + session secret ต้องอยู่ process เดียว)
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "1", "--threads", "4", "--timeout", "120", "app:app"]
