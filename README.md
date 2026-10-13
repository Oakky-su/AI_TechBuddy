# 🤖 Tech Buddy — แชทบอทแนะนำโน้ตบุ๊ค

> **"เพื่อนที่รู้เรื่องโน้ตบุ๊ค"** — RAG Chatbot ภาษาไทยที่แนะนำโน้ตบุ๊คจากสินค้าจริง 99 รุ่นในร้าน
> ตรงไปตรงมา ไม่เชียร์ขาย และไม่แต่งตัวเลขขึ้นเอง

ผู้ใช้พิมพ์ว่า *"โน้ตบุ๊คเล่นเกม งบ 25,000"* แล้วได้คำแนะนำ 1 รุ่น พร้อมเหตุผล สเปก
คะแนนเบนช์มาร์กที่แปลเป็นภาษาคน ข้อดี และข้อเสียจากรีวิวผู้ใช้จริง

---

## ความสามารถหลัก

- **RAG แยกโควตาเอกสาร** — ค้นเอกสารสเปก 4 ชิ้น + รีวิว 3 ชิ้น แยกกัน แล้วแนบภาพรวมร้านทุกครั้ง
  เพื่อการันตีว่า AI เห็นราคาและสเปกเสมอ (ฐานข้อมูลมีรีวิวมากกว่าสเปก 3 เท่า ถ้าค้นรวมจะได้แต่รีวิว)
- **กรองงบตั้งแต่ชั้นฐานข้อมูล** — อ่านงบจากประโยค (`งบ 2 หมื่น`, `งบ 20k`, `งบ 20000-25000`,
  `งบสามหมื่นห้า`) แล้วให้ Chroma กรองราคาตอน query ไม่ใช่ค้นมาแล้วค่อยทิ้ง
- **ไม่แต่งตัวเลข** — ทุกราคาและคะแนนมาจากฐานข้อมูลจริง มีชุดทดสอบคอยจับ
- **ส่วนต่างงบคำนวณด้วยโค้ด** — ไม่ปล่อยให้ LLM บวกลบเลขเอง เพราะพลาดได้
- **รูปแบบคำตอบคงที่** — เหตุผล (CPU/GPU/RAM/STORAGE/ราคา) → สรุปเบนช์มาร์ก → ข้อดี → ข้อเสีย
- **หน้าเว็บธีมแดง** — โหมดสว่าง/มืด, เรนเดอร์ Markdown, ประวัติแชทพร้อมปุ่มลบ

---

## Tech Stack

| ชั้น | เทคโนโลยี |
|---|---|
| Web | Flask 3.0 + Gunicorn |
| Embedding | `BAAI/bge-m3` (หลายภาษา, 1024 มิติ) |
| Vector DB | ChromaDB (Persistent) |
| LLM | OpenThaiGPT 8B ผ่าน ThaiLLM API |
| Frontend | HTML + Vanilla JS + CSS |
| Container | Docker + Docker Compose |

---

## เริ่มใช้งาน

```bash
cp .env.example .env     # ใส่ THAILLM_API_KEY
docker compose up --build
# เปิด http://localhost:5000
```

> ครั้งแรกใช้เวลานาน เพราะต้องโหลดโมเดล bge-m3 (~2 GB) และสร้าง vector index ตอน build

รันบนเครื่องโดยตรง:

```bash
pip install -r requirements.txt
python build_index.py    # สร้าง vector index (รันครั้งเดียว)
python app.py
```

---

## API

| Method | Path | คำอธิบาย |
|---|---|---|
| `GET` | `/` | หน้าแชท |
| `POST` | `/chat` | `{"message": "..."}` → `{"message": "..."}` |
| `POST` | `/new_chat` | ล้างประวัติและงบที่จำไว้ |
| `POST` | `/restore` | โหลดบทสนทนาเก่ากลับเข้า session |

---

## โครงสร้าง

```
app.py              สมองทั้งหมด — route, ค้นหา, prompt, เรียก LLM
build_index.py      แปลง laptops.json เป็น vector index (397 เอกสาร)
laptops.json        ข้อมูลสินค้า 99 รุ่น
templates/          หน้าเว็บ
static/             CSS (ตัวแปรสีอยู่บนสุด) + JS
tests/              ชุดทดสอบ 5 ไฟล์
```

> ⚠️ แก้ `laptops.json` แล้วต้องรัน `python build_index.py` ใหม่ ไม่งั้นบอทยังใช้ข้อมูลเดิม

---

## ทดสอบ

```bash
PYTHONIOENCODING=utf-8 python tests/test_unit.py          # 66 เคส ไม่ต้องใช้โมเดล
PYTHONIOENCODING=utf-8 python tests/test_api.py           # ต้องมีแอปรันอยู่
PYTHONIOENCODING=utf-8 python tests/test_hallucination.py # เช็คว่า AI แต่งตัวเลขไหม

# ชั้น retrieval ต้องรันในที่ที่มี chromadb
docker cp tests/test_retrieval.py ai_teddy-teddy-1:/app/
docker exec ai_teddy-teddy-1 python /app/test_retrieval.py
```

---

## ตั้งค่า (.env)

| ตัวแปร | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|
| `THAILLM_API_KEY` | **(บังคับ)** | API Key จาก ThaiLLM |
| `THAILLM_MODEL` | `openthaigpt-thaillm-8b-instruct-v7.2` | ชื่อโมเดล |
| `DISABLE_THINKING` | `false` | `true` = ตอบเร็วและนิ่งขึ้น แต่คุณภาพลด |
| `SECRET_KEY` | สุ่มชั่วคราว | ถ้าไม่ตั้ง ประวัติจะหายทุกครั้งที่รีสตาร์ต |

ดูทั้งหมดใน `.env.example`
