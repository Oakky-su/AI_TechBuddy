# 🤖 AI TechBuddy — แชทบอทแนะนำโน้ตบุ๊ค

> **"เพื่อนที่รู้เรื่องโน้ตบุ๊ค"** — ระบบ RAG Chatbot ภาษาไทยสำหรับแนะนำโน้ตบุ๊คแบบตรงไปตรงมา ไม่เชียร์ขาย

---

## 📋 สารบัญ

- [ภาพรวม](#ภาพรวม)
- [Tech Stack](#tech-stack)
- [สถาปัตยกรรม](#สถาปัตยกรรม)
- [การติดตั้งและรัน](#การติดตั้งและรัน)
- [Environment Variables](#environment-variables)
- [โครงสร้างโปรเจกต์](#โครงสร้างโปรเจกต์)

---

## ภาพรวม

AI TechBuddy เป็น **Flask Web Application** ที่ใช้เทคนิค **Retrieval-Augmented Generation (RAG)** เพื่อแนะนำโน้ตบุ๊คให้ผู้ใช้ โดย:

- ดึงข้อมูลสเปก ราคา และรีวิวจาก `laptops.json` มาทำ **Vector Embedding**
- เมื่อผู้ใช้ถาม ระบบจะค้นหาข้อมูลที่เกี่ยวข้องจาก **Chroma Vector DB**
- ส่งข้อมูลนั้นให้ **ThaiLLM / OpenThaiGPT** แต่งคำตอบเป็นภาษาไทย แบบเป็นกันเอง

---

## Tech Stack

| ชั้น | เทคโนโลยี |
|------|-----------|
| Web Framework | Flask 3.0 + Gunicorn |
| Embedding Model | `BAAI/bge-m3` (หลายภาษา, ~2 GB) |
| Vector Database | ChromaDB (Persistent) |
| LLM | OpenThaiGPT via ThaiLLM API |
| Container | Docker + Docker Compose |
| Python | 3.12-slim |

---

## สถาปัตยกรรม

```
laptops.json
     │
     ▼ (build_index.py)
bge-m3 Embedding
     │
     ▼
Chroma Vector DB  ◄──── User Query (Embed) ────► Flask App (app.py)
                                                       │
                                                       ▼
                                              ThaiLLM / OpenThaiGPT API
                                                       │
                                                       ▼
                                                  คำตอบภาษาไทย
```

**ขั้นตอนการทำงาน:**
1. `build_index.py` อ่าน `laptops.json` → แปลงเป็นเอกสารภาษาไทย (สเปก + รีวิว) → embed → เก็บลง Chroma
2. เมื่อมี request → Embed คำถาม → Vector Search (Top 5) → ส่งเป็น context ให้ LLM
3. LLM แต่งคำตอบตาม system prompt "เพื่อนที่รู้เรื่องโน้ตบุ๊ค"

---

## การติดตั้งและรัน

### วิธีที่ 1: Docker Compose (แนะนำ)

```bash
# 1. สร้างไฟล์ .env (ดู Environment Variables ด้านล่าง)
cp .env.example .env   # หรือสร้างเองตาม template

# 2. Build และรัน
docker compose up --build

# 3. เปิดเบราว์เซอร์
open http://localhost:5000
```

> ⚠️ **หมายเหตุ:** Docker image จะ download โมเดล `bge-m3` (~2 GB) และสร้าง Chroma index ตอน build — ใช้เวลานานในครั้งแรก

### วิธีที่ 2: รันตรงบนเครื่อง (Local)

```bash
# 1. ติดตั้ง dependencies
pip install -r requirements.txt

# 2. สร้าง vector index (รันครั้งเดียว)
python build_index.py

# 3. รันแอป
flask run
# หรือใช้ gunicorn
gunicorn --bind 0.0.0.0:5000 --workers 1 --threads 4 --timeout 120 app:app
```

---

## Environment Variables

สร้างไฟล์ `.env` ในโฟลเดอร์โปรเจกต์:

```env
# ThaiLLM / OpenThaiGPT API
THAILLM_API_KEY=your-api-key-here
THAILLM_BASE_URL=http://thaillm.or.th/api/v1
THAILLM_MODEL=openthaigpt-thaillm-8b-instruct-v7.2

# ปิดโหมดคิด (reasoning) → เร็วขึ้น แต่คุณภาพอาจลด
DISABLE_THINKING=false

# Flask session secret key
SECRET_KEY=change-me-to-something-random
```

| Variable | Default | คำอธิบาย |
|----------|---------|-----------|
| `THAILLM_API_KEY` | (บังคับ) | API Key จาก ThaiLLM |
| `THAILLM_BASE_URL` | `http://thaillm.or.th/api/v1` | Endpoint ของ ThaiLLM |
| `THAILLM_MODEL` | `openthaigpt-thaillm-8b-instruct-v7.2` | โมเดลที่ใช้ |
| `DISABLE_THINKING` | `false` | `true` = ปิด reasoning mode |
| `SECRET_KEY` | random (ชั่วคราว) | Flask session secret |

---

## โครงสร้างโปรเจกต์

```
AI_TechBuddy/
├── app.py              # Flask app หลัก + RAG pipeline + routes
├── build_index.py      # สคริปต์สร้าง Chroma Vector Index
├── laptops.json        # ข้อมูลโน้ตบุ๊ค (สเปก, ราคา, รีวิว)
├── requirements.txt    # Python dependencies
├── Dockerfile          # Docker image (bge-m3 + Chroma index อยู่ใน image)
├── docker-compose.yml  # Compose config + hot-reload สำหรับพัฒนา
├── templates/          # HTML templates (Jinja2)
├── static/             # CSS, JS, รูปภาพ
└── chroma_db/          # Vector DB (สร้างอัตโนมัติ, ไม่ควร commit)
```

---

## 💡 Tips สำหรับนักพัฒนา

- **Hot-reload**: `docker-compose.yml` mount เฉพาะ `app.py`, `templates/`, `static/` → แก้โค้ดแล้วเห็นผลทันทีโดยไม่ต้อง build ใหม่
- **ห้าม** mount ทั้งโฟลเดอร์ด้วย `".:/app"` เพราะจะบัง `chroma_db/` และ `.cache/huggingface/` ที่อยู่ใน image
- รันด้วย **1 worker** เท่านั้น เพราะโมเดลกินแรม ~2 GB และ session ต้องอยู่ใน process เดียวกัน
