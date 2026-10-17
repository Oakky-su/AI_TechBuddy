# 🤖 Tech Buddy — แชทบอทแนะนำโน้ตบุ๊ค

> **"เพื่อนที่รู้เรื่องโน้ตบุ๊ค"** — RAG Chatbot ภาษาไทยที่แนะนำโน้ตบุ๊คจากสินค้าจริง 99 รุ่นในร้าน
> ตรงไปตรงมา ไม่เชียร์ขาย และไม่แต่งตัวเลขขึ้นเอง

**ระบบที่ใช้งานเป็น RAG ไม่ใช่ agentic RAG** — ทุกคำถามทำตามขั้นตอนเดิม เรียก AI ไม่เกิน 2 ครั้ง
(แปลงคำถามเป็นเงื่อนไข → โค้ดคัดรุ่นจากข้อมูลจริง → เขียนคำตอบ) AI ไม่ได้ตัดสินใจค้นเพิ่มหรือเรียกเครื่องมือเอง

ผู้ใช้พิมพ์ว่า *"โน้ตบุ๊คเล่นเกม งบ 25,000"* แล้วได้รุ่นที่แนะนำ พร้อมเหตุผล สเปก
คะแนนเบนช์มาร์กที่แปลเป็นภาษาคน ข้อดี และข้อเสียจากรีวิวผู้ใช้จริง

---

## ความสามารถหลัก

- **RAG v2: AI แปลงคำถามเป็นเงื่อนไขก่อนค้น (self-query)** — ขั้นแรก AI อ่านคำถาม + บทสนทนาก่อนหน้า
  แล้วตอบเป็น JSON ว่าลูกค้าถามอะไร (ขอรายชื่อ / จัดอันดับ / แนะนำ / เทียบรุ่น / ถามสเปก / นอกเรื่อง) และเงื่อนไข
  (งบ แรม การ์ดจอ CPU ยี่ห้อ ขนาดจอ เรียงตาม — ส่วนความละเอียดจอ SSD และรุ่น Windows โค้ดอ่านจากคำถามเอง)
  จากนั้นโค้ดคัดจากข้อมูลจริงครบทั้ง 99 รุ่น แล้วจึงให้ AI เรียบเรียงคำตอบ
  — นับครบ ไม่หลุดรุ่น จำเงื่อนไขข้ามข้อความได้ ดูขั้นตอนคิดได้ที่ปุ่ม "กระบวนการคิด" ใต้คำตอบ (`rag_v2.py`)
- **RAG v1 (สลับกลับได้ที่ `RAG_PIPELINE=v1`)** — ค้นเอกสารสเปก 6 ชิ้น + รีวิว 3 ชิ้น แยกกัน (ปรับได้ใน `.env`) แล้วแนบภาพรวมร้านทุกครั้ง
  เพื่อการันตีว่า AI เห็นราคาและสเปกเสมอ (ฐานข้อมูลมีรีวิวมากกว่าสเปก 3 เท่า ถ้าค้นรวมจะได้แต่รีวิว)
- **กฎกรองตัวเลข (hybrid retrieval)** — การค้นตามความหมายแยก "แรม 16" กับ "แรม 24" ไม่ออก
  และเห็นแค่ไม่กี่รุ่น โค้ดจึงอ่านเงื่อนไขจากคำถาม (การ์ดจอ แรม จัดอันดับคะแนน CPU/GPU ถูก/แพงสุดในกลุ่ม
  รุ่นที่เกินงบ) แล้วกรองหรือเรียงจากข้อมูลจริงทั้ง 99 รุ่นก่อนส่งให้ AI — เปิด/ปิดที่ `HYBRID_FILTERS`
- **กรองงบตั้งแต่ชั้นฐานข้อมูล** — อ่านงบจากประโยค (`งบ 2 หมื่น`, `งบ 20k`, `งบ 20000-25000`,
  `งบสามหมื่นห้า`) แล้วให้ Chroma กรองราคาตอน query ไม่ใช่ค้นมาแล้วค่อยทิ้ง
- **ไม่แต่งตัวเลข** — ทุกราคาและคะแนนมาจากฐานข้อมูลจริง มีชุดทดสอบคอยจับ
- **ส่วนต่างงบคำนวณด้วยโค้ด** — ไม่ปล่อยให้ LLM บวกลบเลขเอง เพราะพลาดได้
- **รูปแบบคำตอบคงที่** — เหตุผล (CPU/GPU/RAM/STORAGE/ราคา) → สรุปเบนช์มาร์ก → ข้อดี → ข้อเสีย
- **หน้าเว็บธีมแดง** — โหมดสว่าง/มืด, เรนเดอร์ Markdown, ประวัติแชทพร้อมปุ่มลบ
- **เปลี่ยน AI ได้ใน `.env`** — `DEFAULT_PROVIDER` / `VISIBLE_PROVIDERS` (รายการอยู่ใน `providers.json`)
  ถ้าเปิดให้เลือกหลายตัว หน้าเว็บจะมีเมนูมุมบน ใต้คำตอบทุกข้อบอกว่าโมเดลไหนตอบ ใช้เวลาเท่าไหร่

---

## Tech Stack

| ชั้น | เทคโนโลยี |
|---|---|
| Web | Flask 3.0 + Gunicorn |
| Embedding | `BAAI/bge-m3` (หลายภาษา, 1024 มิติ) |
| Vector DB | ChromaDB (Persistent) |
| LLM | endpoint แบบ OpenAI เลือกได้: Ollama Cloud · Gemma 4 31B (ค่าเริ่มต้น) / OpenThaiGPT 8B, Typhoon-S 8B (ThaiLLM) |
| Frontend | HTML + Vanilla JS + CSS |
| Container | Docker + Docker Compose — service เดียวคือ `teddy` |

```
เบราว์เซอร์ ──/chat──▶ teddy (Flask)
   ① LLM แปลงคำถามเป็นเงื่อนไข (JSON)
   ② โค้ดตรวจเงื่อนไข แล้วคัด/เรียงรุ่นจาก laptops.json ครบ 99 รุ่น (+ bge-m3 จัดลำดับตามการใช้งาน)
   ③ LLM เขียนคำตอบจากรุ่นที่คัดมา ──▶ คำตอบ (ทยอยขึ้นจอ)

```

---

## เริ่มใช้งาน

ค่าตั้งทุกอย่าง (API key, วิธีค้น, หน้าเว็บ) อยู่ใน `.env` ไฟล์เดียว ไม่ต้องแก้โค้ด

1. `cp .env.example .env`
2. เปิด `.env` ใส่ `OLLAMA_API_KEY` (หรือ key ของ AI ตัวอื่นในหัวข้อ 1) และตั้ง `SECRET_KEY` เป็นค่าสุ่ม
3. `docker compose up -d --build` แล้วเปิด http://localhost:5000

แก้ `.env` ทีหลัง สั่ง `docker compose up -d` ให้แอปรับค่าใหม่ (ไม่ต้อง build ใหม่)

> ครั้งแรกใช้เวลานาน (ประมาณ 5–15 นาที) เพราะต้องโหลดโมเดล bge-m3 (~2 GB) และสร้าง vector index ตอน build

### เลือกวิธีค้น (ทางเลือก RAG)

| ทางเลือก | ตั้งใน `.env` |
|---|---|
| **RAG v2: AI แปลงคำถามเป็นเงื่อนไข แล้วคัดจากข้อมูลจริง (ค่าเริ่มต้น)** | `RAG_PIPELINE=v2` |
| RAG ค้นตามความหมายอย่างเดียว | `RAG_PIPELINE=v1` `HYBRID_FILTERS=false` `TOP_K_SPEC=6` `TOP_K_REVIEW=3` |
| RAG ดึงสเปกเยอะขึ้น ไม่ดึงรีวิว | `RAG_PIPELINE=v1` `HYBRID_FILTERS=false` `TOP_K_SPEC=15` `TOP_K_REVIEW=0` |
| RAG + กฎกรองตัวเลข | `RAG_PIPELINE=v1` `HYBRID_FILTERS=true` `TOP_K_SPEC=6` `TOP_K_REVIEW=3` |

ผลทดสอบของแต่ละทางเลือกอยู่ในเอกสาร "ข้อจำกัดของ RAG" — วัดซ้ำได้ด้วย `tests/test_question_types.py`

รันบนเครื่องโดยตรง:

```bash
pip install -r requirements.txt
python build_index.py    # สร้าง vector index (รันครั้งเดียว)
python app.py
```

### เปิดออกอินเทอร์เน็ต (Cloudflare Tunnel)

ใช้สแตก [cf-tunnel-panel](https://github.com/WaveCC2/cf-tunnel-panel) (cloudflared + Traefik + panel ที่สร้าง DNS ให้เอง)
รันสแตกนั้นก่อน แล้วรัน Tech Buddy คู่กับ `docker-compose.tunnel.yml`:

```bash
docker compose -f docker-compose.yml -f docker-compose.tunnel.yml up -d
```

หรือใส่ใน `.env` แล้วสั่ง `docker compose up -d` ธรรมดาได้เลย (กันลืมแล้วเว็บสาธารณะหลุด):

```
COMPOSE_FILE=docker-compose.yml;docker-compose.tunnel.yml
COMPOSE_PATH_SEPARATOR=;
TECHBUDDY_HOST=techbuddy.example.com   # ค่าเริ่มต้น techbuddy.pemaai.online
PUBLIC_RATE_LIMIT_PER_MIN=20
```

ไฟล์ tunnel ปิด `/debug`, ใช้ rate limit จริง, ผูกพอร์ต 5000 ไว้แค่ localhost และให้ Traefik
ไม่ส่ง `/debug` ต่อให้แอปเลย — ตอนเปิดโหมดนี้ `tests/test_api.py` จะโดน 429
เพราะ rate limit ต่ำ ให้รันเทสต์ด้วย `docker compose -f docker-compose.yml up -d` (โหมดพัฒนา) แทน

> ⚠️ cf-tunnel-panel ล็อก `traefik:v3.1` ไว้ ซึ่งคุยกับ Docker Engine 29+ ไม่ได้ (ทุกหน้าได้ 404)
> ต้องแก้เป็น `traefik:v3.6` ใน docker-compose.yml ของสแตกนั้น

---

## API

| Method | Path | คำอธิบาย |
|---|---|---|
| `GET` | `/` | หน้าแชท |
| `POST` | `/chat` | `{"message": "..."}` → `{"message": "..."}` |
| `POST` | `/chat/stream` | body เดียวกับ `/chat` แต่ตอบเป็น Server-Sent Events (หน้าเว็บใช้อันนี้) |
| `POST` | `/chat/commit` | `{"token": "..."}` จาก event `done` → บันทึกคำถาม-คำตอบลง session |
| `POST` | `/new_chat` | ล้างประวัติและงบที่จำไว้ |
| `POST` | `/restore` | โหลดบทสนทนาเก่ากลับเข้า session |
| `GET` | `/providers` | รายการ AI ที่เลือกได้ (ไม่มี API key) |

`/chat` รับ `provider` (id ใน `providers.json`) เพิ่มได้
ไม่ส่งมา = ใช้ค่าเริ่มต้นใน `.env` และคืน `meta` บอกว่า AI ตัวไหนตอบ ใช้เวลาเท่าไหร่

`/chat/stream` ส่ง event `status` (เช่น "กำลังเรียบเรียงคำตอบ…"),
`delta` / `replace` (คำตอบที่กำลังพิมพ์ — ตัด `<think>` แล้ว), แล้วจบด้วย `done`
(คำตอบ + `meta` + `commit`) หรือ `error` — ประวัติแชทต้องบันทึกผ่าน `/chat/commit` แยก เพราะ
cookie ของ session ถูกส่งไปตั้งแต่ event แรก ก่อนจะรู้คำตอบ (RAG v1 ส่งคำตอบทีเดียวใน `done`)

### โหมด Debug (กระบวนการคิดของ AI)

ตั้ง `DEBUG_PANEL=true` แล้วทุกคำตอบจะมีปุ่ม **กระบวนการคิด** ในบรรทัดเล็กใต้คำตอบ
กดแล้วแผงด้านขวาเลื่อนออกมาโชว์ว่าคำตอบนั้นคิดยังไง (จอกว้างวางข้างแชท จอแคบลอยทับ, กดซ้ำหรือ X = ปิด)
ไม่มีสวิตช์บนหน้าเว็บ — เปิด/ปิดทั้งระบบที่ `.env` อย่างเดียว

แผงไล่ตามลำดับที่ระบบทำจริง: ตีความคำถาม (งบ ช่วงราคา การ์ดจอ/ชื่อรุ่นที่จับได้) →
embed ด้วย bge-m3 (เวลา + ตัวอย่างเวกเตอร์) → ผลค้นแต่ละรอบพร้อมความใกล้เคียง (cosine) →
กฎที่ปรับผลค้น → เอกสารที่ส่งให้ AI พร้อมที่มา → prompt ทั้งก้อน → เวลา/โทเคน/ความคิดของ AI (ถ้าโมเดลส่งมา)

เบื้องหลังคือส่ง `"debug": true` ไปกับ `/chat` หรือ `/chat/stream` แล้วข้อมูลจะมาใน `debug`
ของ event `done` / `error` — ตอน `DEBUG_PANEL=false` เซิร์ฟเวอร์ไม่สนค่านี้ (`/providers` บอกสถานะใน `debug`)

### เพิ่ม AI ตัวใหม่

เพิ่มรายการใน `providers.json` แล้วรีสตาร์ต `teddy` (ไม่ต้อง build ใหม่)

- รองรับ endpoint แบบ OpenAI (vLLM, Ollama, LM Studio ฯลฯ) → `"kind": "openai-compatible"` + `base_url` + `model`
- ใส่ `api_key_env` เป็นชื่อตัวแปรใน `.env` — ถ้า `key_required` แล้วยังไม่ตั้ง ตัวเลือกจะเป็นสีเทา

---

## โครงสร้าง

```
app.py              route ของเว็บ + RAG v1 (ค้น vector, prompt, เรียก LLM)
rag_v2.py           RAG v2 — AI แปลงคำถามเป็นเงื่อนไข -> คัดจากข้อมูลจริง -> AI เรียบเรียงคำตอบ
build_index.py      แปลง laptops.json เป็น vector index (397 เอกสาร)
laptops.json        ข้อมูลสินค้า 99 รุ่น
templates/          หน้าเว็บ
static/             CSS (ตัวแปรสีอยู่บนสุด) + JS
llm_providers.py    อ่าน providers.json + .env -> รายการ AI ที่เลือกได้
providers.json      รายการ AI (เพิ่ม/ลบโมเดลที่นี่)
tests/              ชุดทดสอบ (test_unit / test_question_types / test_api / ...)
```

> ⚠️ แก้ `laptops.json` แล้วต้องรัน `python build_index.py` ใหม่ ไม่งั้นบอทยังใช้ข้อมูลเดิม

---

## ทดสอบ

```bash
PYTHONIOENCODING=utf-8 python tests/test_unit.py          # ไม่ต้องใช้โมเดล (ส่วนที่ต้องใช้ Chroma ข้ามเองถ้าไม่มี)
PYTHONIOENCODING=utf-8 python tests/test_api.py           # ต้องมีแอปรันอยู่
PYTHONIOENCODING=utf-8 python tests/test_hallucination.py # เช็คว่า AI แต่งตัวเลขไหม

# ถามจริง 42 เคส 9 หมวด (ภาพรวม ถูก/แพงสุด กรองสเปก จัดอันดับ ถามสเปก แนะนำ เทียบรุ่น ถามต่อหลายข้อความ ปฏิเสธ)
# แล้วตรวจกับ laptops.json — รวมข้อความจริงที่ผู้ใช้เคยเจอตอบผิด / QT_ONLY=คำ เลือกรันบางเคส
# ต้องมีแอปรันอยู่ / QT_OUT=ไฟล์ บันทึกคำตอบเต็ม
PYTHONIOENCODING=utf-8 python tests/test_question_types.py http://localhost:5000

# ชั้น retrieval ต้องรันในที่ที่มี chromadb
docker cp tests/test_retrieval.py ai_teddy-teddy-1:/app/
docker exec ai_teddy-teddy-1 python /app/test_retrieval.py
```

---

## ตั้งค่า (.env)

| ตัวแปร | ค่าเริ่มต้น | คำอธิบาย |
|---|---|---|
| `DEFAULT_PROVIDER` | `ollama-gemma4` | AI เริ่มต้น (id ใน `providers.json`) |
| `VISIBLE_PROVIDERS` | ว่าง (ทุกตัว) | id ที่เปิดให้เลือกบนเว็บ คั่นด้วยจุลภาค ตัวอื่นซ่อนจากเมนูและเรียกผ่าน `/chat` ไม่ได้ (รัน `tests/compare_models.py` กับตัวที่ซ่อน ต้องเอาออกจากค่านี้ก่อน) |
| `OLLAMA_API_KEY` | — | key ของ Ollama Cloud (ollama.com/settings) สำหรับ `ollama-gemma4` |
| `RAG_PIPELINE` | `v2` | `v2` = AI แปลงคำถามเป็นเงื่อนไขแล้วคัดจากข้อมูลจริง (`rag_v2.py`) / `v1` = RAG แบบเดิม (3 ค่าด้านล่างใช้กับ v1 เท่านั้น) |
| `HYBRID_FILTERS` | `true` | `true` = เปิดกฎกรองตัวเลข (แรม จัดอันดับคะแนน ถูก/แพงสุดในกลุ่ม รุ่นที่เกินงบ ไล่รายชื่อครบ) / `false` = RAG ค้นตามความหมายอย่างเดียว |
| `TOP_K_SPEC` | `6` | จำนวนเอกสารสเปกที่ค้นมาให้ AI ต่อคำถาม |
| `TOP_K_REVIEW` | `3` | จำนวนเอกสารรีวิวต่อคำถาม (`0` = ไม่ดึงรีวิว) |
| `RATE_LIMIT_PER_MIN` | `1000` (พัฒนา) | ข้อความต่อนาทีต่อคน — เปิดผ่าน tunnel ใช้ `PUBLIC_RATE_LIMIT_PER_MIN` (ค่าเริ่มต้น 20) |
| `DEBUG_PANEL` | `false` | `true` = ทุกคำตอบมีปุ่ม "กระบวนการคิด" เปิดแผงด้านขวาที่โชว์ว่าคำตอบนั้นคิดยังไง (ดูหัวข้อโหมด Debug) — ทุกคนที่เข้าเว็บกดดูได้รวม prompt ทั้งก้อน ปิดก่อนเปิดให้คนทั่วไปใช้ |
| `THAILLM_API_KEY` | — | ใส่แล้วตัวเลือก ThaiLLM จะใช้ได้ |
| `DISABLE_THINKING` | `false` | `true` = ThaiLLM ตอบเร็วและนิ่งขึ้น แต่คุณภาพลด |
| `SECRET_KEY` | สุ่มชั่วคราว | ถ้าไม่ตั้ง ประวัติจะหายทุกครั้งที่รีสตาร์ต |

ดูทั้งหมดใน `.env.example`
