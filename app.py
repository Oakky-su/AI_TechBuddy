# ไฟล์หลักของแอป Flask — RAG chatbot "TechBuddy" แนะนำโน้ตบุ๊ค
# สถาปัตยกรรม: laptops.json -> bge-m3 embedding -> Chroma vector search -> LLM แต่งคำตอบ
# LLM สลับได้จากหน้าเว็บ (providers.json) และเลือกได้ 2 โหมด:
#   classic : ค้นครั้งเดียวแล้วส่งให้ LLM แต่งคำตอบ (แบบเดิม)
#   pi      : ส่งต่อให้ pi-agent (pi.dev) ซึ่งเรียก tool ที่ /internal/tools ค้นข้อมูลเพิ่มเองได้หลายรอบ
import hmac
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import requests
from flask import Flask, Response, render_template, request, jsonify, session
from itsdangerous import BadSignature, URLSafeTimedSerializer
from sentence_transformers import SentenceTransformer  # โมเดล embedding ภาษาไทย/หลายภาษา
import chromadb                                          # vector database
from dotenv import load_dotenv

import llm_providers as llm

load_dotenv()  # โหลด environment variables จากไฟล์ .env

app = Flask(__name__)
# secret_key คงที่ถ้าตั้งใน env (session ไม่หลุดตอนรีสตาร์ต) ไม่งั้น random ชั่วคราว
app.secret_key = os.getenv("SECRET_KEY") or os.urandom(24).hex()

BASE_DIR = Path(__file__).parent
LAPTOPS_PATH = BASE_DIR / "laptops.json"
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "laptops"
EMBED_MODEL = "BAAI/bge-m3"
SUMMARY_ID = "summary_0"     # id เอกสารภาพรวมร้าน (สร้างใน build_index.py)
TOP_K_SPEC = int(os.getenv("TOP_K_SPEC", "6"))      # โควตาเอกสารสเปก — ตัวที่มีราคา/สเปก/คะแนนเบนช์มาร์ก
TOP_K_REVIEW = int(os.getenv("TOP_K_REVIEW", "3"))  # โควตาเอกสารรีวิว — ความเห็นผู้ใช้จริง (0 = ไม่ดึงรีวิว)
DEFAULT_COMPARE = 3          # ขอ "เปรียบเทียบ" ลอย ๆ โดยไม่บอกจำนวน -> เทียบกี่รุ่น
MAX_COMPARE = 5              # เทียบได้มากสุดกี่รุ่น (กันคำตอบยาวจน token หมด)
ENABLE_DEBUG = os.getenv("ENABLE_DEBUG", "false").lower() == "true"  # เปิด /debug
# แผง "ดูกระบวนการคิด" ใต้คำตอบบนหน้าเว็บ: ตีความคำถามยังไง embed แล้วค้นเจออะไร คะแนนเท่าไหร่
# ส่ง prompt อะไรให้ AI และ AI ใช้เวลา/โทเคนเท่าไหร่ — ไว้ทดสอบกับลูกค้า
# เปิดแล้วทุกคนที่เข้าเว็บกดดูได้ รวม system prompt ทั้งก้อน -> ปิดก่อนเปิดให้คนทั่วไปใช้จริง
DEBUG_PANEL = os.getenv("DEBUG_PANEL", "false").lower() == "true"
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))  # ข้อความต่อนาทีต่อ IP
MAX_MSG_LEN = 1_000          # ความยาวข้อความสูงสุดที่รับ
MAX_HISTORY_MSGS = 8         # จำนวนข้อความที่เก็บใน session (4 รอบสนทนา)
MAX_STORED_CHARS = 800       # ตัดคำตอบที่เก็บลง session ไม่ให้ cookie บวม
HISTORY_FOR_LLM = 4          # จำนวนข้อความย้อนหลังที่ส่งให้ LLM เป็นบริบท

# ---- ตั้งค่า LLM ----
# endpoint / โมเดล / API key ของแต่ละเจ้าอยู่ใน providers.json + .env (ดู llm_providers.py)
# ปิดโหมดคิด (reasoning) ของโมเดลที่คิดใน <think> เช่น ThaiLLM -> เร็ว/ถูกกว่า แต่คุณภาพอาจลด
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "false").lower() == "true"

# ---- Pi Agent (pi.dev) ----
# pi-agent เป็น service แยก (โฟลเดอร์ pi-agent/) ที่รัน agent loop ของ pi แล้วเรียก tool
# กลับมาที่ /internal/tools ของแอปนี้ — INTERNAL_TOKEN กันคนนอกยิง endpoint นั้นตรง ๆ
PI_AGENT_URL = os.getenv("PI_AGENT_URL", "http://pi-agent:7070").rstrip("/")
PI_AGENT_TIMEOUT = int(os.getenv("PI_AGENT_TIMEOUT", "150"))
INTERNAL_TOKEN = os.getenv("INTERNAL_TOKEN", "")

# cache โมเดลและ collection ไว้ใน memory ป้องกันการโหลดซ้ำทุก request
_embedder: SentenceTransformer = None
_collection = None
_summary_doc: str | None = None


def get_embedder():
    # โหลด bge-m3 ครั้งแรกแล้ว cache ไว้ (โหลดช้า + กินแรม ~2GB)
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def get_collection():
    # เปิด Chroma persistent client แล้วดึง collection ที่ build ไว้ (จาก build_index.py)
    global _collection
    if _collection is None:
        client = chromadb.PersistentClient(path=str(CHROMA_DIR))
        _collection = client.get_collection(COLLECTION_NAME)
    return _collection


# ---- จำกัดจำนวนคำขอต่อ IP ----
# ทุก /chat = 1 คำขอไปยัง ThaiLLM ซึ่งมีโควตาจำกัด (200 คำขอ/นาที)
# ถ้าไม่จำกัด คนเดียวยิงรัว ๆ ก็กินโควตาหมดและทำให้คนอื่นใช้ไม่ได้
# เก็บใน memory ของ process ได้เพราะรัน 1 worker (ถ้าเพิ่ม worker ต้องย้ายไป Redis)
_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def client_ip() -> str:
    """IP จริงของผู้ใช้ — ถ้าอยู่หลัง proxy (Cloudflare/Render) ต้องอ่านจาก header

    CF-Connecting-IP มาก่อน: Cloudflare เขียนทับค่านี้เองทุก request ผู้ใช้ปลอมไม่ได้
    ส่วนตัวแรกของ X-Forwarded-For ผู้ใช้ใส่มาเองได้ (Cloudflare แค่ต่อท้าย) ถ้าเชื่อตัวนั้น
    ตอนเปิดผ่าน Cloudflare Tunnel คนจะหลบ rate limit ได้ด้วยการสุ่ม header ใหม่ทุกครั้ง
    """
    cf_ip = request.headers.get("CF-Connecting-IP", "").strip()
    if cf_ip:
        return cf_ip
    fwd = request.headers.get("X-Forwarded-For", "")
    return fwd.split(",")[0].strip() if fwd else (request.remote_addr or "unknown")


def too_many_requests(ip: str) -> bool:
    """True ถ้า IP นี้ยิงเกินโควตาในนาทีที่ผ่านมา"""
    now = time.time()
    with _hits_lock:
        q = _hits[ip]
        while q and now - q[0] > 60:
            q.popleft()
        if len(q) >= RATE_LIMIT_PER_MIN:
            return True
        q.append(now)

        # กัน dict โตไม่สิ้นสุดเมื่อมี IP แปลก ๆ เข้ามาเรื่อย ๆ
        if len(_hits) > 5_000:
            for k in [k for k, v in _hits.items() if not v or now - v[-1] > 300]:
                del _hits[k]
    return False


SUMMARY_PREFIXES = ("ภาพรวมข้อมูลโน้ตบุ๊ค", "ร้านมีโน้ตบุ๊ค")


def is_summary_doc(doc: str) -> bool:
    """เอกสารภาพรวมหรือไม่ — รองรับข้อความเก่าด้วย เผื่อ index ยังไม่ได้ rebuild"""
    return doc.startswith(SUMMARY_PREFIXES)


def get_summary_doc() -> str:
    # เอกสารภาพรวมร้าน (จำนวนรุ่น ช่วงราคา รุ่นถูกสุด/แพงสุด) — ดึงตรงด้วย id ไม่ต้องค้นหา
    global _summary_doc
    if _summary_doc is None:
        try:
            docs = get_collection().get(ids=[SUMMARY_ID]).get("documents") or []
            _summary_doc = docs[0] if docs else ""
        except Exception:
            _summary_doc = ""
    return _summary_doc


# ---- System prompt: บุคลิก "AI Tech Buddy ผู้ช่วยเลือกโน้ตบุ๊ค" + เกณฑ์แปลคะแนนเบนช์มาร์ก ----
SYSTEM_PROMPT = """คุณคือ "AI Tech Buddy" ผู้ช่วยเลือกโน้ตบุ๊ค ไม่ใช่ร้านค้าและไม่ใช่พนักงานขาย
หน้าที่ของคุณคือช่วยลูกค้า "เลือก" จากรายการโน้ตบุ๊คที่คุณมีข้อมูลอยู่ ให้คำแนะนำแบบเป็นกันเอง
ตรงไปตรงมา จริงใจ อะไรดีก็บอกดี อะไรไม่คุ้มหรือมีข้อเสียก็บอกตรง ๆ ไม่เชียร์ให้ซื้อ

การใช้คำ: เรียกข้อมูลที่มีว่า "ข้อมูลที่ผมมี" หรือ "รายการที่ผมมี"
ห้ามพูดว่า "ร้านเรา" "ร้านของเรา" "สินค้าในร้าน" หรือทำเหมือนตัวเองเป็นผู้ขาย
และห้ามพูดเรื่องการสั่งซื้อ การจัดส่ง สต็อก หรือโปรโมชั่น เพราะคุณไม่ใช่ร้านค้า

ขอบเขตการตอบ (สำคัญที่สุด อ่านก่อนทุกครั้ง):

คุณมีข้อมูลแค่ "ข้อมูลสินค้า" ที่แนบมาท้าย prompt เท่านั้น ซึ่งประกอบด้วย
ชื่อรุ่น / ราคา / สเปก (CPU, GPU, RAM, หน้าจอ, ความจุ, ระบบปฏิบัติการ) /
คะแนนเบนช์มาร์ก CPU-GPU / รีวิวจากผู้ใช้จริง — **นอกเหนือจากนี้คุณไม่รู้อะไรเลย**

สิ่งที่คุณทำได้ (ทำให้เต็มที่):
- แนะนำรุ่นตามงบและการใช้งานที่ลูกค้าบอก
- เปรียบเทียบหลายรุ่นให้เห็นความต่าง
- บอกราคาและสเปกของรุ่นที่อยู่ในข้อมูลที่ผมมี
- ยกข้อดี-ข้อเสียจากรีวิวผู้ใช้จริง
- คุยโต้ตอบต่อเนื่องเรื่องรุ่นที่กำลังคุยกันอยู่
- ตอบคำถามเรื่องข้อมูลโน้ตบุ๊คที่มี เช่น มีกี่รุ่น ราคาเริ่มต้น/สูงสุด มีรุ่นไหนบ้าง
  (ดูจาก "ภาพรวมข้อมูลโน้ตบุ๊ค" และ "ผลค้นตรงตามสเปก" ใน "ข้อมูลสินค้า") — คำถามแบบนี้ไม่ใช่เรื่องนอกเรื่อง

สิ่งที่ห้ามทำเด็ดขาด:
- **ห้ามตอบคำถามความรู้ทั่วไปที่ไม่มีใน "ข้อมูลสินค้า"** เช่น "TDP คืออะไร"
  "ทำไมโน้ตบุ๊คเกมมิ่งแบตหมดเร็ว" "ดูยังไงว่าอัปเกรด RAM ได้" "ควรรอรุ่นใหม่ไหม"
  ต่อให้คุณรู้คำตอบก็ห้ามตอบ ให้บอกว่าตอบไม่ได้แล้วหยุด
- **ห้ามเอ่ยถึงหรือแนะนำสินค้าที่ไม่มีใน "ข้อมูลสินค้า"** เช่น MacBook, Dell, ThinkPad,
  Surface แม้ลูกค้าจะเอ่ยชื่อมาก่อนก็ตาม ให้บอกว่าไม่มีรุ่นนั้นในข้อมูลที่ผมมี แล้วเสนอรุ่นที่มีแทน
- **ห้ามอธิบายต่อจากความรู้ของตัวเองหลังบอกว่า "ไม่มีข้อมูล" แล้ว** ให้จบตรงนั้นเลย
- ห้ามบอกว่ารุ่นไหนเปิดตัวแล้วหรือยัง ยกเลิกการผลิตหรือไม่ หรือข้อมูลตลาดใด ๆ
  เพราะข้อมูลสินค้าไม่ได้บอกเรื่องพวกนี้

เมื่อตอบไม่ได้ ให้ใช้รูปแบบนี้ (สั้น ๆ 2 ประโยค ห้ามยาวกว่านี้):
"ขอโทษครับ ข้อมูลที่ผมมีไม่ครอบคลุมเรื่องนี้ครับ ถ้าอยากให้ช่วยเลือกโน้ตบุ๊ค
บอกงบและการใช้งานมาได้เลยครับ"

เมื่อลูกค้าคุยนอกเรื่องโน้ตบุ๊คเลย (เช่น ดินฟ้าอากาศ อาหาร ข่าว การเมือง การบ้าน เขียนโค้ด
ขอให้ค้นอินเทอร์เน็ต/เปิดเว็บ ถามราคาจากร้านอื่น หรือสินค้าอื่นที่ไม่ใช่โน้ตบุ๊ค)
(คำถามเรื่องจำนวนรุ่น ราคา ยี่ห้อ หรือสเปกของโน้ตบุ๊คที่มีในข้อมูล ไม่นับเป็นนอกเรื่อง ให้ตอบตามปกติ):
- ห้ามตอบเนื้อหาเรื่องนั้นแม้แต่ประโยคเดียว และห้ามรับปากว่าจะไปค้นหรือเปิดเว็บให้
  (คุณค้นอินเทอร์เน็ตไม่ได้ มีแค่ข้อมูลโน้ตบุ๊คที่แนบมา)
- ตอบสั้น ๆ 2 ประโยคแบบเป็นมิตร: บอกว่าเรื่องนี้ตอบไม่ได้เพราะช่วยได้เฉพาะเรื่องเลือกโน้ตบุ๊ค
  แล้วชวนกลับด้วยคำถาม 1 ข้อเรื่องการใช้งานหรืองบ เช่น
  "เรื่องนี้ผมตอบไม่ได้ครับ ผมช่วยได้เฉพาะเรื่องเลือกโน้ตบุ๊ค
  ตอนนี้มองหาโน้ตบุ๊คไว้ใช้ทำอะไร งบประมาณประมาณเท่าไหร่ครับ?"
- ถ้าเป็นแค่คำทักทายหรือขอบคุณ ให้ทักกลับสั้น ๆ แล้วถามคำถามเดียวกันได้เลย ไม่ต้องขอโทษ
- ถ้ากำลังคุยเรื่องรุ่นไหนค้างอยู่ ให้ชวนกลับไปที่รุ่นนั้นแทนการถามใหม่ตั้งแต่ต้น

กฎการตอบ:

- ตอบโดยอ้างอิงจาก "ข้อมูลสินค้า" ที่ให้มาเท่านั้น (สเปก ราคา คะแนนเบนช์มาร์ก และรีวิวจากผู้ใช้)
- ค่าเริ่มต้นคือแนะนำทีละ 1 รุ่น เลือกรุ่นที่เหมาะที่สุดกับงบ/การใช้งานของผู้ใช้ พร้อมเหตุผลสั้น ๆ (สเปก ราคา หรือรีวิว) แล้วค่อยถามว่าอยากดูรุ่นอื่นเทียบไหม
- **แต่ถ้าลูกค้าขอหลายรุ่นหรือขอให้เปรียบเทียบ ให้ทำตามที่ลูกค้าขอ** เช่น "แนะนำ 3 รุ่น" "เทียบให้หน่อย" ให้ตอบด้วยโครงเปรียบเทียบด้านล่างแทน ห้ามยืนยันจะให้รุ่นเดียว
- พูดข้อดีและข้อเสียตามจริงจากรีวิว ไม่กลบข้อเสีย
- เรื่องงบสำคัญมาก: ถ้าลูกค้าบอกงบมา ห้ามแนะนำรุ่นที่ราคาเกินงบเด็ดขาด ให้เช็คราคาของรุ่นนั้นกับงบก่อนตอบทุกครั้ง ถ้าไม่มีรุ่นไหนอยู่ในงบเลยให้บอกตรง ๆ ว่าไม่มี แล้วค่อยเสนอรุ่นที่ใกล้เคียงพร้อมบอกว่าเกินงบไปเท่าไหร่ อย่าแอบยัดรุ่นที่แพงกว่างบมาเป็นตัวเลือกหลัก
- ถ้าข้อมูลผู้ใช้ยังไม่พอ (ไม่รู้งบหรือการใช้งาน) ให้ถามกลับสั้น ๆ เพียง 1-2 ประโยคเท่านั้น แล้วหยุดรอคำตอบ ห้ามเอ่ยชื่อรุ่น สเปก ราคา หรือรีวิวใด ๆ และห้ามใส่ส่วนสรุปเบนช์มาร์กในข้อความที่ถามกลับ ถึงจะมี "ข้อมูลสินค้า" แนบมาให้ก็ตาม (ข้อมูลนั้นแนบมาทุกครั้งอัตโนมัติ ไม่ได้แปลว่าผู้ใช้ถามถึง) อย่าเดาแทนผู้ใช้
- ถ้าข้อมูลที่ให้มาไม่มีคำตอบ ให้บอกตรง ๆ ว่าไม่แน่ใจ อย่าเดาหรือแต่งสเปก/ราคา/รีวิวขึ้นเอง
- ระบุราคาเป็นบาท ตอบเป็นภาษาไทย น้ำเสียงเป็นกันเองแบบเพื่อนคุยกัน กระชับ ตรงคำถาม

ตอนท้ายของทุกครั้งที่แนะนำรุ่น ให้เพิ่มส่วน "สรุปคะแนนเบนช์มาร์ก" โดยแปลตัวเลข CPU Score และ GPU Score ของรุ่นนั้นเป็นภาษาคน อธิบายว่าอยู่ในระดับไหนและเหมาะกับงานแบบใด ใช้เกณฑ์ต่อไปนี้:

เกณฑ์คะแนน CPU:

- ต่ำกว่า 18,000 → ระดับเริ่มต้น เหมาะกับงานทั่วไป เช่น Office ดูหนัง-ฟังเพลง ท่องเว็บ เรียนออนไลน์ แต่ไม่เหมาะกับงานหนัก
- 18,000–25,000 → ระดับกลาง รองรับงานทั่วไปได้สบาย เล่นเกมและตัดต่อวิดีโอเบา-ปานกลางได้
- 25,000–32,000 → ระดับกลาง-สูง รองรับงานหลากหลาย ทั้งงานทั่วไปและงานหนัก เช่น ตัดต่อวิดีโอ เล่นเกม งาน 3D ได้ลื่น
- สูงกว่า 32,000 → ระดับสูง เหมาะกับงานหนักจริงจัง ตัดต่อวิดีโอความละเอียดสูง เรนเดอร์ 3D งาน AI/Machine Learning และเล่นเกม AAA ได้เต็มที่

เกณฑ์คะแนน GPU:

- ต่ำกว่า 4,000 → การ์ดจอออนบอร์ด/ระดับเริ่มต้น เล่นเกมได้แค่เกมเบา ๆ เน้นงานเอกสารและดูหนัง ไม่เหมาะกับเกมหนักหรืองานกราฟิก
- 4,000–8,000 → ระดับเริ่มต้น-กลาง เล่นเกม eSports (Valorant, Genshin) ปรับกลาง-สูงได้ ตัดต่อเบา ๆ ได้
- 8,000–14,000 → ระดับกลาง-สูง เล่นเกมได้หลากหลายรวมถึงเกม AAA (ปรับกราฟิกปานกลาง-สูง) ทำงานกราฟิก-ตัดต่อได้ดี
- สูงกว่า 14,000 → ระดับสูง เล่นเกม AAA ปรับ Ultra ได้ลื่น รองรับงาน 3D เรนเดอร์ และงานกราฟิกหนัก ๆ

รูปแบบคำตอบ (บังคับ):

โครงนี้ใช้ "ตอนแนะนำรุ่นเดียว" ให้ตอบตามนี้ทุกครั้ง ห้ามสลับลำดับ ห้ามข้ามหัวข้อ
(ถ้าลูกค้าขอหลายรุ่นหรือขอเปรียบเทียบ ให้ข้ามไปใช้โครงเปรียบเทียบที่อยู่ถัดลงไปแทน):

จากงบ [งบ] บาท และความต้องการ[การใช้งานที่ลูกค้าบอก] แนะนำรุ่น **[ชื่อรุ่นเต็ม]** ครับ

**เหตุผล:**

- **CPU:** [ชื่อ CPU] คะแนน CPU [ตัวเลข] อยู่ในระดับ[ระดับ] [อธิบายสั้น ๆ ว่ารองรับงานแบบใด]
- **GPU:** [ชื่อ GPU] คะแนน GPU [ตัวเลข] อยู่ในระดับ[ระดับ] [อธิบายสั้น ๆ ว่าเล่นเกม/ทำงานกราฟิกได้แค่ไหน]
- **RAM:** [แรม] [อธิบายสั้น ๆ ว่าพอกับการใช้งานแบบไหน]
- **STORAGE:** [ที่เก็บข้อมูล] [อธิบายสั้น ๆ ว่าพอไหม เร็วแค่ไหน]
- **ราคา:** [ราคา] บาท [หมายเหตุราคา]

**สรุปเบนช์มาร์ก:**

คะแนน CPU [ตัวเลข] ถือว่าอยู่ในระดับ [ระดับ] ซึ่ง [อธิบายว่ารองรับงานแบบใดได้] ส่วนคะแนน GPU [ตัวเลข] อยู่ในระดับ [ระดับ] [อธิบายว่าเล่นเกม/ทำงานกราฟิกได้แค่ไหน]

**ข้อดี:**

- [ข้อดีจากสเปกหรือรีวิวจริง อย่างน้อย 2 ข้อ]

**ข้อเสีย:**

- [ข้อเสียจากสเปกหรือรีวิวจริง อย่างน้อย 1 ข้อ]

อยากดูรุ่นอื่นเทียบไหมครับ?

รูปแบบคำตอบเมื่อลูกค้าขอหลายรุ่นหรือขอเปรียบเทียบ (บังคับ):

จากงบ [งบ] บาท และความต้องการ[การใช้งาน] ขอเสนอ [จำนวน] รุ่นนี้ครับ

**1. [ชื่อรุ่นเต็ม] — [ราคา] บาท [หมายเหตุราคา]**

- **CPU:** [ชื่อ CPU] คะแนน [ตัวเลข] อยู่ในระดับ[ระดับ]
- **GPU:** [ชื่อ GPU] คะแนน [ตัวเลข] อยู่ในระดับ[ระดับ]
- **RAM / STORAGE:** [แรม] / [ที่เก็บข้อมูล]
- **จุดเด่น:** [ข้อดีจากสเปกหรือรีวิวจริง]
- **ข้อควรรู้:** [ข้อเสียจากสเปกหรือรีวิวจริง]

**2. [ชื่อรุ่นเต็ม] — [ราคา] บาท [หมายเหตุราคา]**

(เขียนหัวข้อย่อยชุดเดียวกับข้อ 1)

**3. [ชื่อรุ่นเต็ม] — [ราคา] บาท [หมายเหตุราคา]**

(เขียนหัวข้อย่อยชุดเดียวกับข้อ 1)

**สรุปเปรียบเทียบ:**

- **เน้น[ด้านที่รุ่นนั้นเด่น] →** [ชื่อรุ่น] เพราะ[เหตุผลสั้น ๆ อิงตัวเลขจริง]
- **เน้น[ด้านอื่น] →** [ชื่อรุ่น] เพราะ[เหตุผลสั้น ๆ]
- **คุ้มที่สุดในงบ →** [ชื่อรุ่น] เพราะ[เหตุผลสั้น ๆ]

สนใจรุ่นไหนเป็นพิเศษ บอกได้เลยครับ เดี๋ยวลงรายละเอียดให้

กฎของโครงเปรียบเทียบ:

- ใช้เฉพาะรุ่นที่อยู่ใน "ข้อมูลสินค้า" เท่านั้น ถ้ามีไม่ครบตามจำนวนที่ขอ ให้เสนอเท่าที่มีจริงแล้วบอกตรง ๆ ว่ามีให้เลือกเท่านี้
- ทุกรุ่นต้องอยู่ในงบ ถ้าลูกค้าบอกงบมา
- ตัวเลขทุกตัวต้องมาจากข้อมูลสินค้า ห้ามเดา ช่องที่ไม่มีข้อมูลให้เขียน "ไม่มีข้อมูลในระบบ"
- ส่วน "สรุปเปรียบเทียบ" ห้ามข้าม เพราะเป็นเหตุผลที่ลูกค้าขอให้เทียบ
- ในโหมดนี้ไม่ต้องใส่หัวข้อ "สรุปเบนช์มาร์ก" แยก เพราะระบุระดับคะแนนไว้ในแต่ละรุ่นแล้ว

วิธีเติมช่อง [หมายเหตุราคา]:

- ลูกค้าบอกงบมา → คัดลอกข้อความส่วนต่างจาก "ส่วนต่างงบ" ที่คำนวณมาให้แล้วด้านบนมาใช้ตรง ๆ
  (เช่น "อยู่ในงบ เหลืออีก 10 บาท") ห้ามบวกลบเลขเอง เพราะคำนวณผิดง่าย
- ลูกค้าไม่ได้บอกงบ → เว้นว่างไว้ ไม่ต้องเขียนอะไรต่อท้ายราคา และห้ามเอ่ยถึงงบหรือเงินที่เหลือเลย

ข้อห้ามเพิ่มเติมเรื่องรูปแบบ:

- **ห้ามพิมพ์วงเล็บเหลี่ยม [ ] หรือข้อความอธิบายเงื่อนไขลงในคำตอบเด็ดขาด** ทุกช่องในโครงต้องแทนที่ด้วยค่าจริงเสมอ
- **ห้ามตั้งหมวดหมู่ช่วงราคาขึ้นมาเองเด็ดขาด** เช่น "งบประมาณต่ำ (ไม่เกิน 30,000)" "งบกลาง" "งบสูง"
  เพราะเมื่อตั้งหมวดเองจะต้องเทียบตัวเลขเองแล้วมักใส่ผิดหมวด (เคยใส่รุ่น 36,990 ไว้ในหมวด "ไม่เกิน 30,000")
  ให้ใช้เฉพาะโครงที่กำหนดไว้ข้างบนเท่านั้น ถ้าจะเรียงหลายรุ่นให้เรียงตามความเหมาะสมกับลูกค้า ไม่ใช่แบ่งเป็นช่วงราคา

- ห้ามเขียนคำแนะนำเป็นย่อหน้ายาวติดกัน ต้องแยกหัวข้อและใช้ bullet ตามโครงข้างบนเสมอ
- ถ้าคำตอบมีการเอ่ยชื่อรุ่นใดรุ่นหนึ่งเป็นคำตอบหลัก ถือว่าเป็น "การแนะนำรุ่น" ต้องใช้โครงนี้เสมอ รวมถึงคำถามอย่าง "รุ่นไหนถูกที่สุด" "รุ่นไหนแพงที่สุด" "รุ่นไหนคุ้มสุด" ด้วย ห้ามตอบแค่ชื่อกับราคาแล้วถามว่าอยากรู้สเปกเพิ่มไหม
- ก่อนเขียน "ระดับ" ของคะแนน ให้เทียบตัวเลขกับเกณฑ์ข้างบนทีละช่วงก่อนทุกครั้ง เช่น CPU 49,500 มากกว่า 32,000 จึงเป็น "ระดับสูง" ไม่ใช่ "กลาง-สูง" ห้ามเดาระดับเอาเอง
- ถ้าเอกสารสเปกมี "ระดับคะแนน (คำนวณแล้ว)" ต่อท้าย ให้ใช้ระดับนั้นตรง ๆ ห้ามตีความใหม่
- เมื่อแนะนำรุ่นไปแล้ว ห้ามถามรายละเอียดเพิ่ม (เช่น ขนาดหน้าจอ ระบบปฏิบัติการ) ให้ปิดท้ายด้วย "อยากดูรุ่นอื่นเทียบไหมครับ?" เท่านั้น
- โครงนี้ใช้เฉพาะตอนแนะนำรุ่น ถ้าข้อมูลยังไม่พอจนต้องถามกลับ ให้ถามสั้น ๆ 1-2 ประโยคตามกฎด้านบน ห้ามใส่โครงนี้
- **ห้ามเขียนว่า "เกินงบ" ถ้าราคาน้อยกว่าหรือเท่ากับงบเด็ดขาด** ราคาที่ต่ำกว่างบคือ "อยู่ในงบ" และผลต่าง (งบ ลบ ราคา) คือเงินที่ "เหลือ" ไม่ใช่ส่วนที่ "เกิน" ให้เทียบตัวเลขสองตัวก่อนเขียนทุกครั้ง
- **ใช้โครงนี้เสมอ ห้ามข้ามไปตอบแบบสั้น** ถ้าช่องไหนหาค่าใน "ข้อมูลสินค้า" ไม่เจอจริง ๆ ให้เขียนเฉพาะช่องนั้นว่า "ไม่มีข้อมูลในระบบ" แล้วเขียนช่องที่เหลือตามปกติ ห้ามทิ้งทั้งโครง
- **ห้ามเดาค่าที่ไม่มีในข้อมูลสินค้าเด็ดขาด** ห้ามคาดเดาสเปกจากชื่อรุ่น แบรนด์ หรือราคา"""

# ต่อท้าย SYSTEM_PROMPT เฉพาะโหมด Pi Agent — บอกว่ามีเครื่องมืออะไร และใช้เมื่อไหร่
# ชื่อ tool ต้องตรงกับ TOOLS ด้านล่างและ pi-agent/server.mjs
PI_TOOL_GUIDE = """

เครื่องมือค้นข้อมูล (โหมดนี้เท่านั้น):

"ข้อมูลสินค้า" ที่แนบมากับคำถามถูกค้นมาให้แล้วหนึ่งรอบ แต่อาจยังไม่ครบ คุณเรียกเครื่องมือเหล่านี้เพื่อค้นเพิ่มได้:
- filter_laptops: กรองจากสเปกจริงของทุกรุ่น เช่น GPU หรือ CPU ที่ระบุ, RAM ขั้นต่ำ, คะแนนขั้นต่ำ, ช่วงราคา แล้วเรียงตามราคาหรือคะแนน — ใช้เมื่อคำถามมีเงื่อนไขสเปกชัดเจน หรือต้องการ "ทุกรุ่น" ที่ตรงเงื่อนไข
- search_laptops: ค้นตามความหมาย เช่น การใช้งาน หรือความรู้สึกจากรีวิว (แบตอึด เครื่องเบา จอสวย เสียงพัดลม)
- get_laptop_details: ดูสเปกเต็มและรีวิวทั้งหมดของรุ่นที่ลูกค้าเอ่ยชื่อ

กติกาการใช้เครื่องมือ:
- ถ้าข้อมูลที่แนบมาพอตอบแล้ว ให้ตอบเลย ไม่ต้องเรียกเครื่องมือ
- ถ้าลูกค้าเอ่ยชื่อรุ่นหรือสเปกที่ไม่มีในข้อมูลที่แนบ หรือขอเปรียบเทียบหลายรุ่นแต่ข้อมูลยังไม่ครบ ให้เรียกเครื่องมือก่อนตอบ
- ลูกค้าบอกงบ -> ส่ง max_price ทุกครั้ง แล้วคัดลอกตัวเลขจาก "ส่วนต่างงบ" ที่เครื่องมือคำนวณมาให้ ห้ามคำนวณเอง
- ข้อมูลจากเครื่องมือนับเป็น "ข้อมูลสินค้า" เช่นกัน กฎทุกข้อด้านบน (ขอบเขต งบ รูปแบบคำตอบ) ยังใช้เหมือนเดิม
- ค้นแล้วไม่เจอ ให้บอกตรง ๆ ว่าไม่มีในข้อมูลที่ผมมี ห้ามเดา
- เครื่องมือทั้งหมดค้นได้แค่ข้อมูลโน้ตบุ๊คในระบบ ไม่มีเครื่องมือค้นอินเทอร์เน็ต — ห้ามใช้เครื่องมือกับเรื่องนอกโน้ตบุ๊ค
- ห้ามเล่าในคำตอบว่าเรียกเครื่องมืออะไร ให้ตอบลูกค้าตามรูปแบบที่กำหนดเท่านั้น"""

# ต่อท้าย "คำถามลูกค้า" ในโหมด Pi Agent — ตำแหน่งท้ายสุดคือจุดที่โมเดลให้น้ำหนักมากที่สุด
# (ทดสอบกับ Gemma 4 31B: ถ้ามีแค่ PI_TOOL_GUIDE ใน system prompt โมเดลไม่เรียก tool เลยสักครั้ง
#  และตอบ "ไม่มีข้อมูล" กับคำถามเทียบรุ่นที่มีอยู่จริง เพราะกฎ "ตอบไม่ได้ให้ปฏิเสธ" ด้านบนชนะ)
PI_TURN_NOTE = (
    "\n\n(ข้อมูลสินค้าด้านบนเป็นผลค้นรอบแรกเท่านั้น อาจยังไม่ครบ "
    "ถ้าคำถามเอ่ยชื่อรุ่น ชื่อซีรีส์ หรือสเปกเฉพาะ เช่น การ์ดจอ/ซีพียูรุ่นไหน หรือขอเปรียบเทียบ/ขอหลายรุ่น "
    "ให้เรียกเครื่องมือค้นก่อนตอบ: ชื่อรุ่น -> get_laptop_details, เงื่อนไขสเปก -> filter_laptops "
    "**ห้ามตอบว่าไม่มีข้อมูลโดยยังไม่ได้ค้นด้วยเครื่องมือ** "
    "ยกเว้นคำถามความรู้ทั่วไปที่ไม่เกี่ยวกับรุ่นใดในข้อมูล ให้ปฏิเสธตามกฎได้เลย "
    "และถ้าเป็นเรื่องนอกโน้ตบุ๊คเลย ไม่ต้องเรียกเครื่องมือ ให้ชวนกลับมาเรื่องโน้ตบุ๊คตามกฎ)"
)


# ---- ดึงงบประมาณจากคำถาม (ใช้กรองราคาก่อนค้นหา) ----
# ปัญหา: vector search จัดอันดับด้วย "ความหมายใกล้เคียง" ไม่เข้าใจตัวเลข
# ถามงบ 20,000 แต่คำว่า "เล่นเกม" ไปตรงกับรุ่น 26,000 มันก็ดึงรุ่นนั้นมา
# วิธีแก้: อ่านงบจากคำถาม แล้วใช้ metadata filter ของ Chroma กรองราคาตั้งแต่ตอน query

# คำที่บ่งบอกว่าตัวเลขถัดไปคือ "งบ" (บังคับต้องมีคำพวกนี้ ไม่งั้น "RTX 5050" จะโดนอ่านเป็นราคา)
_BUDGET_CUE = (
    r"(?:งบประมาณ|งบ|ไม่เกิน|ราคาไม่เกิน|ต่ำกว่า|ถูกกว่า|ในราคา|ราคาประมาณ|budget"
    r"|ราคาอยู่ระหว่าง|ราคาระหว่าง|อยู่ระหว่าง|ระหว่าง|ช่วงราคา|ตั้งแต่|ราคา)"
)
_THAI_NUM_WORD = {
    "หนึ่ง": 1, "สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5,
    "หก": 6, "เจ็ด": 7, "แปด": 8, "เก้า": 9, "สิบ": 10,
}
_THAI_ALT = "|".join(_THAI_NUM_WORD)
_NUM = r"\d[\d,]*(?:\.\d+)?|" + _THAI_ALT   # ตัวเลขอารบิก หรือคำไทย
_GAP = r"\D{0,20}?"                          # คำคั่นระหว่าง cue กับตัวเลข เช่น "ที่ตั้งไว้คือประมาณ"
_UNIT = r"หมื่น|พัน|k|K"
_UNIT_MULT = {"หมื่น": 10_000, "พัน": 1_000, "k": 1_000, "K": 1_000}

# 3 รูปแบบ เรียงตามความเฉพาะเจาะจง (เจาะจงกว่าชนะเมื่อเริ่มที่ตำแหน่งเดียวกัน)
#   0) ช่วงราคา  "งบ 20000-25000"        -> เอาเพดานบน
#   1) มีหน่วยคูณ "งบ 2 หมื่น" "งบสามหมื่นห้า" "งบ 20k"
#   2) ตัวเลขล้วน "งบ 20000"
_RE_RANGE = re.compile(
    _BUDGET_CUE + _GAP + rf"(?P<lo>{_NUM})\s*(?:-|–|—|ถึง)\s*(?P<hi>{_NUM})\s*(?P<unit>{_UNIT})?"
)
_RE_UNIT = re.compile(
    _BUDGET_CUE + _GAP + rf"(?:(?P<num>{_NUM})\s*)?(?P<unit>{_UNIT})\s*(?P<tail>{_THAI_ALT})?"
)
_RE_PLAIN = re.compile(_BUDGET_CUE + _GAP + rf"(?P<num>{_NUM})")

# "ราคา 50,000 ขึ้นไป" / "แพงกว่า 40,000" = มีแต่ขอบล่าง ไม่มีเพดานบน
_RE_MIN_ONLY = re.compile(
    rf"(?P<lo>{_NUM})\s*(?P<unit>{_UNIT})?\s*(?:บาท)?\s*ขึ้นไป"
    rf"|(?:แพงกว่า|มากกว่า|สูงกว่า)\D{{0,10}}?(?P<lo2>{_NUM})\s*(?P<unit2>{_UNIT})?"
)

MIN_BUDGET = 1_000  # ต่ำกว่านี้ถือว่าไม่ใช่ราคา (เช่น "จอไม่เกิน 15 นิ้ว", "ไม่เกิน 2 ปี")


def _to_number(raw: str) -> float:
    if raw in _THAI_NUM_WORD:
        return float(_THAI_NUM_WORD[raw])
    return float(raw.replace(",", ""))


def _match_value(prio: int, m: re.Match) -> float:
    if prio == 0:  # ช่วงราคา -> ใช้เพดานบน
        value = _to_number(m.group("hi"))
        unit = m.group("unit")
        return value * _UNIT_MULT[unit] if unit else value

    if prio == 1:  # มีหน่วยคูณ
        mult = _UNIT_MULT[m.group("unit")]
        num = m.group("num")
        value = (_to_number(num) if num else 1.0) * mult
        # ภาษาพูด "สามหมื่นห้า" = 35,000 / "หมื่นห้า" = 15,000 (เศษท้ายคือหลักถัดลงมา)
        if m.group("tail"):
            value += _THAI_NUM_WORD[m.group("tail")] * mult / 10
        return value

    return _to_number(m.group("num"))


def extract_budget(text: str) -> int | None:
    """อ่านงบสูงสุดจากข้อความ เช่น 'งบไม่เกิน 20000' / 'งบ 2 หมื่น' / 'งบ 20k' -> 20000
    คืน None ถ้าไม่พบงบ (จะได้ไม่กรองราคามั่วตอนผู้ใช้ไม่ได้พูดถึงงบ)

    เก็บผู้สมัครทุกตำแหน่งแล้วไล่จากซ้ายไปขวา ไม่ใช่เอาแค่ match แรก —
    ไม่งั้นประโยคอย่าง "จอไม่เกิน 15 นิ้ว งบ 25000" จะติดที่ "15" (ต่ำกว่าเกณฑ์)
    แล้วเลิกทั้งประโยค ทั้งที่งบจริงอยู่ถัดไป
    """
    # ช่วงที่เป็น "ขอบล่างล้วน" เช่น "ราคา 50,000 ขึ้นไป" ต้องไม่ถูกอ่านเป็นเพดานบน
    # (ไม่งั้น "50,000 ขึ้นไป" จะกลายเป็น "ไม่เกิน 50,000" ซึ่งตรงข้ามกับที่ลูกค้าต้องการ)
    min_only_spans = [m.span() for m in _RE_MIN_ONLY.finditer(text)]

    def inside_min_only(pos: int) -> bool:
        return any(a <= pos < b for a, b in min_only_spans)

    candidates = []
    for prio, regex in enumerate((_RE_RANGE, _RE_UNIT, _RE_PLAIN)):
        for m in regex.finditer(text):
            # ช่วงราคา (prio 0) มีเพดานบนในตัว จึงไม่ต้องกัน
            if prio > 0 and inside_min_only(m.end() - 1):
                continue
            candidates.append((m.start(), prio, _match_value(prio, m)))

    for _, _, value in sorted(candidates, key=lambda c: (c[0], c[1])):
        if value >= MIN_BUDGET:
            return int(value)
    return None


def extract_min_price(text: str) -> int | None:
    """อ่าน "ขอบล่าง" ของช่วงราคา เช่น "ราคาอยู่ระหว่าง 30,000-40,000" -> 30,000

    แยกจาก extract_budget() ที่คืนเฉพาะเพดานบน เพราะคำถาม 2 แบบนี้ความหมายต่างกัน:
      "งบไม่เกิน 40,000"              -> เอารุ่นไหนก็ได้ที่ <= 40,000  (ไม่มีขอบล่าง)
      "ราคาอยู่ระหว่าง 30,000-40,000" -> ต้องอยู่ในช่วงนี้เท่านั้น     (มีขอบล่าง)
    คืน None ถ้าไม่ใช่คำถามแบบช่วงราคา
    """
    for m in _RE_RANGE.finditer(text):
        lo, hi = _to_number(m.group("lo")), _to_number(m.group("hi"))
        unit = m.group("unit")
        if unit:                      # "งบ 2-3 หมื่น" -> หน่วยคูณใช้กับทั้งคู่
            mult = _UNIT_MULT[unit]
            lo, hi = lo * mult, hi * mult
        if lo >= MIN_BUDGET and hi >= lo:
            return int(lo)

    # แบบมีแต่ขอบล่าง: "ราคา 50,000 ขึ้นไป" / "แพงกว่า 4 หมื่น"
    for m in _RE_MIN_ONLY.finditer(text):
        raw = m.group("lo") or m.group("lo2")
        unit = m.group("unit") or m.group("unit2")
        if not raw:
            continue
        value = _to_number(raw) * (_UNIT_MULT[unit] if unit else 1)
        if value >= MIN_BUDGET:
            return int(value)
    return None


# คำถามแบบ "รุ่นถูก/แพงที่สุด" — vector search ตอบไม่ได้เพราะมันจัดอันดับด้วยความหมาย
# ไม่ได้เรียงตามราคา ต้องดึงเอกสารสเปกของรุ่นนั้นตรง ๆ จาก metadata แทน
_RE_CHEAPEST = re.compile(r"ถูกที่สุด|ถูกสุด|ราคาต่ำสุด|ประหยัดที่สุด|ราคาถูกที่สุด")
_RE_PRICIEST = re.compile(r"แพงที่สุด|แพงสุด|ราคาสูงสุด|ท็อปสุด")

# ---- ตรวจเจตนา "ขอหลายรุ่น / ขอเปรียบเทียบ" ----
# ค่าเริ่มต้นของบอทคือแนะนำทีละรุ่น ถ้าลูกค้าขอหลายรุ่นต้องสลับไปใช้โครงเปรียบเทียบ
_THAI_COUNT = {"สอง": 2, "สาม": 3, "สี่": 4, "ห้า": 5}
_RE_COUNT = re.compile(
    # เลขไม่เกิน 2 หลัก: "RTX 4060 รุ่นไหนคุ้ม" / "งบ 30,000 รุ่นไหน" ไม่ใช่การขอ 4060 รุ่น
    r"(?<![\d,.])(\d{1,2}|" + "|".join(_THAI_COUNT) + r")\s*(?:รุ่น|ตัว|เครื่อง|ยี่ห้อ)"
)
_RE_COMPARE = re.compile(r"เปรียบเทียบ|เทียบ|ต่างกันยังไง|อันไหนดีกว่า|เลือกอันไหน")
_RE_MANY = re.compile(r"หลายรุ่น|หลายตัว|มีอะไรบ้าง|มีรุ่นไหนบ้าง|แนะนำ.{0,10}บ้าง")


def wanted_model_count(question: str) -> int | None:
    """ลูกค้าขอกี่รุ่น — คืน None ถ้าไม่ได้ขอหลายรุ่น (ใช้โครงรุ่นเดียวตามปกติ)"""
    m = _RE_COUNT.search(question)
    if m:
        raw = m.group(1)
        n = _THAI_COUNT.get(raw) or int(raw)
        if n >= 2:
            return min(n, MAX_COMPARE)
        return None   # "1 รุ่น" = ขอรุ่นเดียว ใช้โครงเดิม

    if _RE_COMPARE.search(question) or _RE_MANY.search(question):
        return DEFAULT_COMPARE
    return None

_specs_by_price: list[tuple[int, str]] | None = None


def get_specs_by_price() -> list[tuple[int, str]]:
    """เอกสารสเปกทั้งหมดเรียงตามราคาจากถูกไปแพง (ดึงครั้งเดียวแล้ว cache)"""
    global _specs_by_price
    if _specs_by_price is None:
        try:
            res = get_collection().get(
                where={"type": {"$eq": "spec"}}, include=["documents", "metadatas"]
            )
            pairs = [(m["price"], d) for m, d in zip(res["metadatas"], res["documents"])]
            _specs_by_price = sorted(pairs, key=lambda p: p[0])
        except Exception:
            _specs_by_price = []
    return _specs_by_price


def _extreme_spec_doc(question: str, budget: int | None) -> str | None:
    """คืนเอกสารสเปกของรุ่นถูกสุด/แพงสุด ถ้าคำถามถามถึง (ถ้ามีงบ จะอิงเฉพาะรุ่นในงบ)"""
    cheap, pricey = _RE_CHEAPEST.search(question), _RE_PRICIEST.search(question)
    if not (cheap or pricey):
        return None
    # "RTX ที่ถูกที่สุด" / "แรม 32GB ที่แพงที่สุด" = ถูก/แพงสุดในกลุ่ม -> exact_match เลือกให้แล้ว
    # ถ้ายังยัดรุ่นถูกสุดของทั้งร้านไว้หัวแถว AI จะหยิบผิดตัว
    if HYBRID_FILTERS and spec_filters(question):
        return None

    pairs = get_specs_by_price()
    if not pairs:
        return None
    if budget is not None:
        pairs = [p for p in pairs if p[0] <= budget] or pairs
    return pairs[0][1] if cheap else pairs[-1][1]


def _specs_nearest_to(
    budget: int | None, min_price: int | None, n: int = 2
) -> list[tuple[int, str]]:
    """รุ่นที่ "ราคาใกล้ช่วงที่ขอที่สุด" n อันดับ — ใช้ตอนไม่มีรุ่นไหนอยู่ในช่วงราคาเลย
    วัดระยะจากขอบที่ใกล้ที่สุด เช่น งบ 10,000 -> รุ่น 12,990 ห่าง 2,990 (ใกล้สุด)
    """
    pairs = get_specs_by_price()
    if not pairs:
        return []

    def distance(price: int) -> int:
        if budget is not None and price > budget:
            return price - budget
        if min_price is not None and price < min_price:
            return min_price - price
        return 0

    return sorted(pairs, key=lambda p: (distance(p[0]), p[0]))[:n]


def _query_by_type(
    q_emb, n: int, doc_type: str, budget: int | None, min_price: int | None = None,
    trace: list | None = None,
) -> list[str]:
    """ค้นเฉพาะเอกสารประเภทที่ต้องการ (spec / review) พร้อมกรองราคาถ้ามีงบ
    trace (โหมด debug): ต่อท้ายผลค้นรอบนี้ — ชื่อรุ่น ราคา และความใกล้เคียงกับคำถาม"""
    if n <= 0:
        return []
    conds = [{"type": {"$eq": doc_type}}]
    if budget is not None:
        # metadata "price" ถูกเก็บไว้ตอน build_index.py แล้ว (ทั้งเอกสารสเปกและรีวิว)
        conds.append({"price": {"$lte": budget}})
    if min_price is not None:
        # คำถามแบบ "ราคาอยู่ระหว่าง X-Y" ต้องมีขอบล่างด้วย ไม่งั้นจะได้รุ่นถูก ๆ ที่ไม่ได้ขอ
        conds.append({"price": {"$gte": min_price}})
    where = conds[0] if len(conds) == 1 else {"$and": conds}
    res = get_collection().query(query_embeddings=q_emb, n_results=n, where=where,
                                 include=["documents", "metadatas", "distances"])
    if trace is not None:
        price_filter = " · ".join(filter(None, [
            f"ราคาตั้งแต่ {min_price:,}" if min_price is not None else "",
            f"ราคาไม่เกิน {budget:,}" if budget is not None else "",
        ]))
        trace.append({
            "type": doc_type,
            "want": n,
            "filter": price_filter or "ไม่กรองราคา",
            # collection ใช้ระยะ l2 กับเวกเตอร์ที่ normalize แล้ว: ระยะ = 2 - 2cos -> cos = 1 - ระยะ/2
            "hits": [{"name": m.get("name"), "price": m.get("price"), "similarity": round(1 - d / 2, 4)}
                     for m, d in zip(res["metadatas"][0], res["distances"][0])],
        })
    return res["documents"][0]


# ---- ค้นตรงตามชื่อรุ่น/สเปก (เสริม vector search) ----
# vector search จัดอันดับด้วย "ความหมายใกล้คำถาม" จึงพลาดคำถามที่ต้องการ "ทุกรุ่นที่ตรงเป๊ะ"
# ทดสอบจริง: "RTX 4060 งบไม่เกิน 40,000" มีในข้อมูล 2 รุ่นแต่ค้นได้ 1 / "เทียบ Legion 5 กับ HP Victus 16"
# ไม่ได้เอกสาร Victus 16 มาเลย โมเดลจึงตอบว่าไม่มี — เลยจับสเปกและชื่อรุ่นจากข้อความด้วยโค้ด
# แล้วดึงรุ่นที่ตรงจากข้อมูลจริงมาไว้หัวแถว ช่วยทั้งโหมดค้นเร็ว (ค้นรอบเดียว) และค้นละเอียด (เรียก tool น้อยลง)
MAX_EXACT = 6           # รุ่นที่ตรงเงื่อนไขสูงสุดต่อคำถาม กัน prompt ยาวเกิน
EXACT_REVIEWS = 4       # รีวิวของรุ่นที่ลูกค้าเอ่ยชื่อ (รุ่นละ 1 ข้อ)
_RE_GPU_ASK = re.compile(
    r"(?<![a-z0-9])(?:(?:rtx|gtx)\s*\d{4}(?:\s*ti)?|rx\s*\d{4}m?|radeon\s*\d{3}m|arc\s*[a-z]?\d{3}[a-z]?)(?![a-z0-9])",
    re.I,
)
_series_index: list[tuple[str, str]] | None = None


def _squash(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def gpu_terms(question: str) -> list[str]:
    """การ์ดจอที่ลูกค้าเอ่ยถึง เช่น "rtx4060" / "RTX 4060 Ti" -> ["rtx 4060", "rtx 4060 ti"]"""
    terms = []
    for m in _RE_GPU_ASK.finditer(question):
        term = re.sub(r"([a-z])(\d)", r"\1 \2", _squash(m.group(0)))
        if term not in terms:
            terms.append(term)
    return terms


def gpu_matches(gpu: str, term: str) -> bool:
    """"rtx 5070" ต้องไม่ไปโดน "RTX 5070 Ti" (คนละรุ่น ราคาต่างกันมาก)"""
    tail = "" if term.endswith(" ti") else r"(?!\s*ti)"
    return re.search(re.escape(term) + tail + r"(?![a-z0-9])", _squash(gpu)) is not None


def _name_tokens(name: str) -> list[str]:
    """คำในชื่อรุ่นที่คนพิมพ์ถึง — หยุดที่รหัสรุ่น ("16-r0356TX" เหลือแค่ "16")"""
    out = []
    for tok in name.split():
        if "-" in tok:
            head = tok.split("-")[0]
            if head.isdigit() and len(head) <= 3:
                out.append(head)
            break
        out.append(tok)
    return out


def get_series_index() -> list[tuple[str, str]]:
    """(วลีชื่อรุ่น 2-3 คำ, ชื่อสินค้าเต็ม) เช่น ("legion 5", "Lenovo Legion 5 15IRX10-...")"""
    global _series_index
    if _series_index is None:
        idx = set()
        for lp in get_laptops():
            toks = _name_tokens(lp["ชื่อสินค้า"])
            for n in (2, 3):
                for i in range(len(toks) - n + 1):
                    phrase = _squash(" ".join(toks[i:i + n]))
                    if re.search(r"[a-z]", phrase):
                        idx.add((phrase, lp["ชื่อสินค้า"]))
        _series_index = sorted(idx, key=lambda p: -len(p[0]))
    return _series_index


def asked_model_names(question: str) -> list[str]:
    """รุ่นที่ลูกค้าเอ่ยชื่อ (ชื่อซีรีส์ หรือรหัสรุ่นเต็ม) — ไม่มี = []"""
    q = _squash(question)
    names = []
    for lp in get_laptops():  # รหัสรุ่นเต็ม เช่น "ANV15-51-56QE"
        code = next((t for t in lp["ชื่อสินค้า"].split() if "-" in t), "")
        if len(code) >= 6 and code.lower() in q and lp["ชื่อสินค้า"] not in names:
            names.append(lp["ชื่อสินค้า"])
    hits = [(phrase, name) for phrase, name in get_series_index()
            if re.search(r"(?<![a-z0-9])" + re.escape(phrase) + r"(?![a-z0-9])", q)]
    # เก็บเฉพาะวลีที่เจาะจงที่สุด: พิมพ์ "HP Victus 16" ต้องได้แค่ Victus 16
    # ไม่ใช่ทุกรุ่นที่ขึ้นต้น "HP Victus" (วลีสั้นที่อยู่ในวลีที่ยาวกว่าซึ่งก็ตรงด้วย = ทิ้ง)
    matched = {p for p, _ in hits}
    specific = {p for p in matched if not any(p != o and f" {p} " in f" {o} " for o in matched)}
    for phrase, name in hits:
        if phrase in specific and name not in names:
            names.append(name)
    return names


# ---- เงื่อนไขตัวเลขในคำถาม (แรม / จัดอันดับคะแนน / ถูก-แพงสุดในกลุ่ม) ----
# ค้นตามความหมายแยก "แรม 16GB" กับ "แรม 24GB" ไม่ออก และเห็นแค่ top-k จึงจัดอันดับทั้งร้านไม่ได้
# ทดสอบจริง: "แรม 24GB มีรุ่นไหนบ้าง" ได้สเปกแรม 16GB มาทั้ง 6 ชิ้น (ในร้านมีแรม 24GB 13 รุ่น)
# "รุ่นไหนคะแนน CPU สูงที่สุด" ได้ Legion 9i ทั้งที่ MSI Raider 16 Max สูงกว่า — เลยอ่านเงื่อนไข
# แล้วกรอง/เรียงจากข้อมูลจริงทั้ง 99 รุ่นด้วยโค้ด (hybrid retrieval)
# ปิด (false) = RAG ค้นตามความหมายอย่างเดียว เหลือแค่กฎการ์ดจอ + ชื่อรุ่น — ไว้เทียบผลหรือถ้าลูกค้าเลือกแบบนั้น
HYBRID_FILTERS = os.getenv("HYBRID_FILTERS", "true").lower() == "true"
RAM_SIZES = {4, 8, 12, 16, 24, 32, 48, 64, 96, 128}
MAX_LIST = 15           # รุ่นที่ไล่ชื่อในบรรทัดสรุปผลค้นตรงได้สูงสุด
_RE_RAM_ASK = re.compile(
    r"(?:แรม|ram|หน่วยความจำ)\s*(?:ขนาด\s*)?(\d{1,3})(?!\d)|(?<!\d)(\d{1,3})\s*(?:gb|กิ๊ก|กิก|จิ๊ก)\s*(?:แรม|ram)",
    re.I,
)
_RE_AT_LEAST = re.compile(r"ขึ้นไป|อย่างน้อย|ไม่ต่ำกว่า|อย่างต่ำ|หรือมากกว่า|\d\s*(?:gb)?\s*\+", re.I)
_RE_GPU_FAMILY = re.compile(r"(?<![a-z])(rtx|gtx)(?!\s*\d)", re.I)
_TOP = r"(?:สูงสุด|สูงที่สุด|แรงสุด|แรงที่สุด|ดีสุด|ดีที่สุด|เร็วสุด|เร็วที่สุด|มากสุด|มากที่สุด|อันดับ\s*1)"
_RE_CPU_TOP = re.compile(rf"(?:cpu|ซีพียู|โปรเซสเซอร์).{{0,25}}{_TOP}|{_TOP}.{{0,10}}(?:cpu|ซีพียู)", re.I)
_RE_GPU_TOP = re.compile(rf"(?:gpu|การ์ดจอ|กราฟิก).{{0,25}}{_TOP}|{_TOP}.{{0,10}}(?:gpu|การ์ดจอ)", re.I)
_RE_DESC = re.compile(r"แพงไปถูก|มากไปน้อย|สูงไปต่ำ")
# ลูกค้าขอ "รายชื่อ" -> ไล่ชื่อครบตามผลค้นตรง แทนโครงแนะนำ/เปรียบเทียบ (ใช้ใน build_prompt)
_RE_LIST = re.compile(r"มีรุ่นไหนบ้าง|มีตัวไหนบ้าง|รุ่นไหนบ้าง|ตัวไหนบ้าง|มีอะไรบ้าง|กี่รุ่น|ลิสต์|list|ไล่ชื่อ|รายชื่อ|เรียง", re.I)
EXACT_NOTE = "ผลค้นตรงตามสเปก (นับจากข้อมูลทั้งหมด): "


def ram_condition(question: str) -> tuple[int, bool] | None:
    """(แรมกี่ GB, ขึ้นไปไหม) — "แรม 24" = 24 GB พอดี / "แรม 24GB ขึ้นไป" = ตั้งแต่ 24 GB"""
    m = _RE_RAM_ASK.search(question)
    if not m:
        return None
    gb = int(m.group(1) or m.group(2))
    if gb not in RAM_SIZES:
        return None
    around = question[max(0, m.start() - 15): m.end() + 15]
    return gb, _RE_AT_LEAST.search(around) is not None


def spec_filters(question: str) -> list[tuple[str, object]]:
    """เงื่อนไขสเปกที่อ่านได้จากคำถาม -> [(คำอธิบาย, ฟังก์ชันตรวจรุ่น)] ใช้กรองข้อมูลจริงทั้งร้าน"""
    out = []
    terms = gpu_terms(question)
    if terms:
        out.append((f"การ์ดจอ {', '.join(t.upper() for t in terms)}",
                    lambda lp: any(gpu_matches(lp["spec"]["GPU"], t) for t in terms)))
    if not HYBRID_FILTERS:
        return out   # RAG ล้วน: เหลือแค่กฎการ์ดจอรุ่นเฉพาะแบบเดิม
    if not terms and (fam := _RE_GPU_FAMILY.search(question)):
        word = fam.group(1).lower()
        out.append((f"การ์ดจอ {word.upper()}", lambda lp: word in lp["spec"]["GPU"].lower()))
    ram = ram_condition(question)
    if ram:
        gb, at_least = ram
        if at_least:
            out.append((f"แรม {gb} GB ขึ้นไป", lambda lp: _ram_gb(lp["spec"]["RAM"]) >= gb))
        else:
            out.append((f"แรม {gb} GB", lambda lp: _ram_gb(lp["spec"]["RAM"]) == gb))
    return out


def exact_match(question: str, budget: int | None = None, min_price: int | None = None
                ) -> tuple[list[str], list[str]]:
    """คืน (เอกสารสเปก + บรรทัดสรุปผลค้นตรง, ชื่อรุ่นที่ลูกค้าเอ่ยถึง) — ไม่เข้าเงื่อนไขเลย = ([], [])
    บรรทัดสรุปขึ้นต้นด้วย EXACT_NOTE เสมอ (หน้า debug และ build_prompt ใช้คำนี้แยกจากเอกสารสเปก)"""
    def in_range(lp):
        return ((budget is None or lp["ราคา"] <= budget)
                and (min_price is None or lp["ราคา"] >= min_price))

    def price_cond():
        return "".join([f" ราคาตั้งแต่ {min_price:,} บาท" if min_price else "",
                        f" ราคาไม่เกิน {budget:,} บาท" if budget else ""])

    name = lambda lp: lp["ชื่อสินค้า"]
    laptops = get_laptops()
    names, notes = [], []
    filters = spec_filters(question)
    label = " · ".join(text for text, _ in filters) + price_cond()
    pool = [lp for lp in laptops if in_range(lp) and all(ok(lp) for _, ok in filters)]
    rank = "CPU" if _RE_CPU_TOP.search(question) else "GPU" if _RE_GPU_TOP.search(question) else None
    cheap, pricey = _RE_CHEAPEST.search(question), _RE_PRICIEST.search(question)
    if not HYBRID_FILTERS:
        rank = cheap = pricey = None

    if rank:
        # จัดอันดับคะแนนจากทุกรุ่นที่เข้าเงื่อนไข (คะแนนเท่ากันให้รุ่นถูกกว่าขึ้นก่อน)
        score = lambda lp: lp["benchmark"][f"{rank} Score"]
        top = sorted(pool, key=lambda lp: (-score(lp), lp["ราคา"]))[:5]
        best = [lp for lp in top if score(lp) == score(top[0])] if top else []
        ranking = ", ".join(f"{i}. {name(lp)} (คะแนน {rank} {score(lp):,}, ราคา {lp['ราคา']:,} บาท)"
                            for i, lp in enumerate(top, 1))
        notes.append(f"{EXACT_NOTE}เรียงตามคะแนน {rank} จากสูงไปต่ำ"
                     f"{' เฉพาะ' + label if label else ' จากทุกรุ่น'}: {ranking or 'ไม่มีรุ่นที่ตรงเงื่อนไข'}"
                     + (f" — รุ่นที่คะแนน {rank} สูงที่สุดคือ {' และ '.join(name(lp) for lp in best)}"
                        + (" (คะแนนเท่ากัน)" if len(best) > 1 else "") if best else ""))
        names += [name(lp) for lp in top]
    elif filters and (cheap or pricey):
        ordered = sorted(pool, key=lambda lp: lp["ราคา"], reverse=bool(pricey))
        if ordered:
            lp = ordered[0]
            notes.append(f"{EXACT_NOTE}รุ่นที่{'แพง' if pricey else 'ถูก'}ที่สุดในกลุ่ม{label} คือ {name(lp)}"
                         f" ราคา {lp['ราคา']:,} บาท (กลุ่มนี้มีทั้งหมด {len(ordered)} รุ่น)")
        else:
            notes.append(f"{EXACT_NOTE}ไม่มีรุ่นที่ตรงเงื่อนไข{label}")
        names += [name(lp) for lp in ordered[:3]]
    elif filters:
        hits = sorted(pool, key=lambda lp: lp["ราคา"], reverse=bool(_RE_DESC.search(question)))
        listing = ", ".join(f"{name(lp)} ({lp['ราคา']:,} บาท)" for lp in hits[:MAX_LIST])
        more = f" และอีก {len(hits) - MAX_LIST} รุ่น" if len(hits) > MAX_LIST else ""
        # บอกจำนวนรุ่นที่ตรงจริงไว้ด้วย โมเดลจะได้ตอบ "มีเท่านี้" ได้มั่นใจ ไม่แต่งรุ่นเพิ่ม
        notes.append(f"{EXACT_NOTE}{label} มีทั้งหมด {len(hits)} รุ่น"
                     f" (เรียงราคาจาก{'แพงไปถูก' if _RE_DESC.search(question) else 'ถูกไปแพง'}): "
                     f"{listing + more if hits else 'ไม่มีรุ่นที่ตรงเงื่อนไข'}")
        names += [name(lp) for lp in hits]

    mentioned = asked_model_names(question)
    asked = [n for n in mentioned if any(name(lp) == n and in_range(lp) for lp in laptops)]
    # รุ่นที่ลูกค้าเอ่ยถึงแต่ราคาอยู่นอกงบ — ถูกกรองออกจากผลค้น ถ้าไม่บอก AI จะตอบว่า "ไม่มีในข้อมูล"
    outside = [lp for lp in laptops if HYBRID_FILTERS and name(lp) in mentioned and not in_range(lp)]
    if outside:
        def why(lp):
            if budget is not None and lp["ราคา"] > budget:
                return f"เกินงบ {lp['ราคา'] - budget:,} บาท"
            return "ต่ำกว่าช่วงราคาที่ขอ"
        notes.append(f"{EXACT_NOTE}รุ่นที่ลูกค้าเอ่ยถึงมีอยู่ในข้อมูล แต่ราคาอยู่นอกงบที่ขอ: "
                     + ", ".join(f"{name(lp)} ราคา {lp['ราคา']:,} บาท ({why(lp)})" for lp in outside[:MAX_EXACT])
                     + " — ให้บอกลูกค้าว่ารุ่นนี้มี แต่ราคาเกินงบ ห้ามตอบว่าไม่มีในข้อมูล")
    names += [n for n in asked if n not in names]

    by_name = get_spec_doc_by_name()
    docs = [by_name[n] for n in names[:MAX_EXACT] if n in by_name]
    return (notes + docs if docs or notes else []), asked[:MAX_EXACT]


def _reviews_of(names: list[str], limit: int) -> list[str]:
    """รีวิวรุ่นละ 1 ข้อของรุ่นที่ลูกค้าเอ่ยชื่อ — ใช้ตอบข้อดี/ข้อเสียตอนเทียบรุ่น"""
    out = []
    for name in names[:limit]:
        res = get_collection().get(
            where={"$and": [{"type": {"$eq": "review"}}, {"name": {"$eq": name}}]},
            include=["documents"], limit=1,
        )
        out += res["documents"]
    return out


def retrieve(
    question: str, k_spec: int = TOP_K_SPEC, k_review: int = TOP_K_REVIEW,
    budget: int | None = None, min_price: int | None = None, trace: dict | None = None,
) -> tuple[list[str], bool]:
    """ค้นเอกสารที่เกี่ยวข้องที่สุด ถ้ามีงบจะกรองเฉพาะรุ่นที่ราคาไม่เกินงบ
    คืน (เอกสาร, over_budget) — over_budget=True แปลว่าไม่มีรุ่นไหนอยู่ในงบเลย
    trace (โหมด debug): บันทึกทุกขั้น — embed, ผลค้นแต่ละรอบ, ขั้นที่ปรับผลค้น และที่มาของเอกสารแต่ละชิ้น

    สำคัญ: ต้องแยกโควตา spec/review แทนที่จะค้นรวมแล้วเอา top-k
    เพราะในฐานข้อมูลมีรีวิว 297 ต่อสเปก 99 (3:1) ถ้าค้นรวม รีวิวจะกวาด
    ทุกช่องไปหมดสำหรับคำถามเชิงความรู้สึก ("แบตอึดไหม", "เครื่องร้อนไหม")
    แล้ว LLM จะไม่เห็นราคา/สเปก/คะแนนเบนช์มาร์กเลย -> แต่งตัวเลขเอง
    """
    started = time.perf_counter()
    embedder = get_embedder()
    loaded = time.perf_counter()   # คำถามแรกหลังรีสตาร์ตต้องโหลดโมเดลก่อน (หลายวินาที) แยกเวลาไว้ไม่ให้ดูเหมือน embed ช้า
    q_emb = embedder.encode([question], normalize_embeddings=True).tolist()
    searches = adjust = None
    if trace is not None:
        load_ms = round((loaded - started) * 1000)
        trace["embed"] = {
            "model": EMBED_MODEL, "dims": len(q_emb[0]),
            "ms": round((time.perf_counter() - loaded) * 1000),
            "load_ms": load_ms if load_ms >= 50 else 0,
            "preview": [round(v, 4) for v in q_emb[0][:8]],
        }
        searches = trace["searches"] = []
        adjust = trace["adjust"] = []
    source: dict[str, str] = {}   # เอกสาร -> มาจากขั้นไหน (เฉพาะที่ไม่ได้มาจาก vector search ตรง ๆ)

    specs = _query_by_type(q_emb, k_spec, "spec", budget, min_price, trace=searches)
    over_budget = (budget is not None or min_price is not None) and not specs
    if over_budget:
        # ไม่มีรุ่นไหนในช่วงราคาเลย -> ค้นใหม่แบบไม่กรอง เพื่อให้ AI บอกได้ว่ามีอะไรใกล้เคียง
        specs = _query_by_type(q_emb, k_spec, "spec", None, trace=searches)
        reviews = _query_by_type(q_emb, k_review, "review", None, trace=searches)
        # แต่ vector search คืนรุ่นที่ "ความหมายใกล้คำถาม" ไม่ใช่รุ่นที่ "ราคาใกล้งบ"
        # ถาม "เล่นเกม งบ 10,000" จะได้โน้ตบุ๊คเกมมิ่งราคา 25,990 มาแทนรุ่นถูกสุด 12,990
        # จึงต้องยัดรุ่นที่ราคาใกล้งบจริง ๆ เข้าไปหัวแถวเอง
        near = [doc for _, doc in _specs_nearest_to(budget, min_price, n=2)]
        specs = near + [d for d in specs if d not in near][: max(0, k_spec - len(near))]
        if adjust is not None:
            source.update(dict.fromkeys(near, "near_price"))
            adjust.append("ไม่มีรุ่นไหนอยู่ในช่วงราคาที่ขอ -> ค้นใหม่แบบไม่กรองราคา "
                          f"แล้วดึงรุ่นที่ราคาใกล้งบที่สุดขึ้นหัวแถว: {', '.join(map(doc_name, near))}")
    else:
        reviews = _query_by_type(q_emb, k_review, "review", budget, min_price, trace=searches)

    # ถามถึงรุ่นถูกสุด/แพงสุด -> ยัดเอกสารสเปกของรุ่นนั้นไว้หัวแถวเสมอ
    # ไม่งั้น AI จะรู้แค่ชื่อกับราคาจากเอกสารภาพรวมร้าน แล้วตอบได้ไม่ครบตามรูปแบบ
    extreme = _extreme_spec_doc(question, budget)
    if extreme and extreme not in specs:
        specs = [extreme] + specs[:-1]
        if adjust is not None:
            source[extreme] = "extreme"
            adjust.append(f"ถามถึงรุ่นถูก/แพงที่สุด -> ใส่ {doc_name(extreme)} ไว้หัวแถว")

    # ลูกค้าเอ่ยสเปกหรือชื่อรุ่นตรง ๆ -> รุ่นที่ตรงจากข้อมูลจริงขึ้นหัวแถว ตามด้วยผล vector search
    exact, asked = exact_match(question, budget, min_price)
    if exact:
        rest = [d for d in specs if d not in exact]
        specs = exact + rest[: max(2, k_spec - len(exact))]
        named_reviews = _reviews_of(asked, EXACT_REVIEWS)
        reviews = named_reviews + [r for r in reviews if r not in named_reviews]
        if adjust is not None:
            source.update(dict.fromkeys(exact, "exact"))
            source.update(dict.fromkeys(named_reviews, "exact"))
            adjust += [d for d in exact if d.startswith("ผลค้นตรงตามสเปก")]
            if asked:
                adjust.append(f"ลูกค้าเอ่ยชื่อรุ่น -> ดึงสเปกและรีวิวมาตรง ๆ: {', '.join(asked)}")

    # แนบภาพรวมร้านทุกครั้ง — ตอบคำถามแบบ "มีกี่รุ่น / ถูกสุดเท่าไหร่" ได้โดยไม่กินโควตาค้นหา
    summary = get_summary_doc()
    context = specs + reviews + ([summary] if summary else [])
    if trace is not None:
        if summary:
            source[summary] = "summary"
        # สิ่งที่ AI เห็นจริง ตามลำดับใน prompt (ต่อท้ายระดับคะแนนแบบเดียวกับ build_prompt)
        trace["context"] = [{"source": source.get(d, "vector"), "kind": doc_kind(d), "text": with_levels(d)}
                            for d in context]
        trace["search_ms"] = round((time.perf_counter() - started) * 1000)
    return context, over_budget


def doc_kind(doc: str) -> str:
    if is_summary_doc(doc):
        return "summary"
    if doc.startswith("รีวิวจากผู้ใช้รุ่น"):
        return "review"
    if doc.startswith("ผลค้นตรงตามสเปก"):
        return "note"
    return "spec"


def doc_name(doc: str) -> str:
    """ชื่อรุ่นจากเอกสารสเปก/รีวิว (รูปแบบมาจาก build_index.py)"""
    return doc.removeprefix("รีวิวจากผู้ใช้รุ่น ").split(" ราคา ")[0]


def strip_think(text: str) -> str:
    """ตัดบล็อก <think>...</think> ทิ้ง เหลือเฉพาะคำตอบจริง (โมเดลเป็น reasoning model)"""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # เผื่อกรณีแท็กเปิดหายไป เหลือแต่แท็กปิด
    cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL)
    # เผื่อกรณี token หมดกลางคันจนแท็กปิดไม่ครบ — ถ้าไม่ตัด ผู้ใช้จะเห็นความคิดดิบของโมเดล
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


# ดึง "ชื่อรุ่น + ราคา" ออกจากเอกสารสเปก เพื่อคำนวณส่วนต่างงบให้ LLM
# (LLM บวกลบเลขพลาดได้ เช่น 25,000 - 24,990 ตอบเป็น 1 แทนที่จะเป็น 10)
_SPEC_PRICE_RE = re.compile(r"^(.+?) ราคา ([\d,]+) บาท \|")


def budget_diff_table(context: list[str], budget: int) -> str:
    """ตารางส่วนต่างงบที่คำนวณด้วยโค้ด ไม่ปล่อยให้ LLM คิดเลขเอง"""
    lines = []
    for doc in context:
        # เอกสารภาพรวมร้านมีรูปแบบ "... ราคา 12,990 บาท | ..." เหมือนเอกสารสเปก
        # ถ้าไม่กันไว้ regex จะจับมันเป็น "รุ่น" ปลอมที่ชื่อยาวทั้งประโยค
        if is_summary_doc(doc):
            continue
        m = _SPEC_PRICE_RE.match(doc)
        if not m:
            continue
        name, price = m.group(1), int(m.group(2).replace(",", ""))
        if price <= budget:
            lines.append(f"- {name}: ราคา {price:,} บาท -> อยู่ในงบ เหลืออีก {budget - price:,} บาท")
        else:
            lines.append(f"- {name}: ราคา {price:,} บาท -> เกินงบไป {price - budget:,} บาท")
    if not lines:
        return ""
    return (
        "\nส่วนต่างงบ (คำนวณมาให้แล้ว ให้คัดลอกตัวเลขนี้ไปใช้ ห้ามคำนวณเอง):\n"
        + "\n".join(lines) + "\n"
    )


# ---- ระดับคะแนนเบนช์มาร์ก (คำนวณด้วยโค้ดตามเกณฑ์ใน SYSTEM_PROMPT) ----
# LLM เทียบตัวเลขกับช่วงพลาดบ่อย (ทดสอบกับ Gemma: GPU 10,500 ถูกเขียนว่า "เริ่มต้น-กลาง"
# ทั้งที่อยู่ช่วง 8,000-14,000 = "กลาง-สูง") จึงคำนวณให้แล้วแนบท้ายเอกสารสเปก แบบเดียวกับส่วนต่างงบ
# แก้เกณฑ์ใน SYSTEM_PROMPT ต้องแก้ตัวเลขที่นี่ด้วย
CPU_LEVELS = ((18_000, "เริ่มต้น"), (25_000, "กลาง"), (32_000, "กลาง-สูง"))
GPU_LEVELS = ((4_000, "เริ่มต้น"), (8_000, "เริ่มต้น-กลาง"), (14_000, "กลาง-สูง"))
_SCORE_RE = re.compile(r"คะแนนเบนช์มาร์ก CPU (\d+) / GPU (\d+)$")


def score_level(score: int, levels: tuple) -> str:
    for upper, name in levels:
        if score < upper:
            return name
    # ขอบบนของช่วงสุดท้ายนับเป็นช่วงนั้น เช่น GPU 14,000 พอดี = กลาง-สูง (เกณฑ์เขียน "8,000–14,000")
    return levels[-1][1] if score == levels[-1][0] else "สูง"


def with_levels(doc: str) -> str:
    """ต่อท้ายเอกสารสเปกด้วยระดับคะแนนที่คำนวณแล้ว (เอกสารรีวิว/ภาพรวมคืนค่าเดิม)"""
    m = _SCORE_RE.search(doc)
    if not m or doc.startswith("รีวิวจากผู้ใช้รุ่น"):
        return doc
    cpu, gpu = int(m.group(1)), int(m.group(2))
    # เขียนแค่ชื่อระดับเปล่า ๆ — โครงคำตอบมีคำว่า "อยู่ในระดับ[...]" อยู่แล้ว
    # (เคยใส่ "CPU ระดับกลาง" แล้วโมเดลคัดลอกไปเป็น "อยู่ในระดับระดับกลาง")
    return (f"{doc} | ระดับคะแนน (คำนวณแล้ว): CPU = {score_level(cpu, CPU_LEVELS)} / "
            f"GPU = {score_level(gpu, GPU_LEVELS)}")


def build_prompt(
    question: str,
    context: list[str],
    history: list,
    budget: int | None = None,
    over_budget: bool = False,
    min_price: int | None = None,
    include_system: bool = True,
    no_think: bool = False,
) -> str:
    """ประกอบข้อความทั้งก้อนที่จะส่งให้ LLM
    แยกออกมาจาก generate() เพื่อให้ /debug ดู prompt ตัวจริงได้โดยไม่ต้องเรียก LLM
    (ถ้าปล่อยให้ประกอบคนละที่ สองฝั่งจะเพี้ยนจากกันเมื่อแก้โค้ดภายหลัง)

    โมเดล 8B รับ system role ได้ไม่ดีเท่ารุ่นใหญ่ เลยยุบ system + ประวัติ + context + คำถาม
    ไว้ใน user message เดียว — ส่วนโหมด Pi Agent ส่ง system prompt แยก (include_system=False)
    เพราะ agent ต้องวาง system prompt ไว้หน้าคำประกาศ tool
    """
    context_text = "\n".join(f"- {with_levels(c)}" for c in context)

    history_text = ""
    if history:
        recent = history[-HISTORY_FOR_LLM:]  # ประวัติล่าสุดให้ AI จำบริบท
        history_text = "\n".join(
            f"{'ลูกค้า' if h['role'] == 'user' else 'ผู้ช่วย'}: {h['content']}"
            for h in recent
        )

    # บอก AI ให้ชัดว่า context ถูกกรองราคามาแล้ว (หรือกรองแล้วไม่เจอ) จะได้ไม่แนะนำเกินงบ
    budget_note = ""
    if budget is not None and over_budget:
        # ระบุรุ่นที่ใกล้งบที่สุดให้ชัดเจน ไม่งั้นโมเดลจะหยิบรุ่นแพงที่ความหมายใกล้คำถาม
        # มาบอกว่า "ใกล้เคียงงบที่สุด" ทั้งที่มีรุ่นถูกกว่าอยู่
        nearest = _specs_nearest_to(budget, min_price, n=1)
        nearest_note = ""
        if nearest:
            price, doc = nearest[0]
            name = doc.split(" ราคา ")[0]
            nearest_note = (
                f"รุ่นที่ราคาใกล้งบที่สุดคือ {name} ราคา {price:,} บาท "
                f"(เกินงบไป {price - budget:,} บาท) — ให้เสนอรุ่นนี้ ห้ามเสนอรุ่นอื่นว่าใกล้เคียงกว่า\n"
            )
        budget_note = (
            f"\n**งบของลูกค้า: ไม่เกิน {budget:,} บาท — ไม่มีรุ่นไหนในข้อมูลที่ราคาไม่เกินงบนี้เลย**\n"
            f"ให้บอกลูกค้าตรง ๆ ว่าไม่มีรุ่นในงบ แล้วบอกชัดว่าราคาเกินงบไปเท่าไหร่\n"
            + nearest_note
        )
    elif budget is not None:
        budget_note = (
            f"\n**งบของลูกค้า: ไม่เกิน {budget:,} บาท — ข้อมูลสินค้าด้านล่างคัดมาเฉพาะรุ่นที่ราคาไม่เกินงบแล้ว**\n"
            f"ห้ามแนะนำรุ่นที่ราคาเกิน {budget:,} บาท และห้ามเอ่ยถึงรุ่นที่ไม่ได้อยู่ในข้อมูลด้านล่าง\n"
            f"ทุกรุ่นด้านล่างราคาไม่เกินงบแล้ว ดังนั้นห้ามเขียนว่า 'เกินงบ' เด็ดขาด\n"
        )
    # คำถามแบบ "ราคาอยู่ระหว่าง X-Y" ต้องย้ำขอบล่างด้วย ไม่งั้นโมเดลจะเสนอรุ่นถูกกว่าช่วงที่ขอ
    if min_price is not None and budget is not None:
        budget_note = (
            f"\n**ลูกค้าต้องการรุ่นที่ราคาอยู่ระหว่าง {min_price:,} ถึง {budget:,} บาท**\n"
            f"ห้ามเสนอรุ่นที่ถูกกว่า {min_price:,} บาท หรือแพงกว่า {budget:,} บาท เด็ดขาด "
            f"ข้อมูลสินค้าด้านล่างคัดมาเฉพาะรุ่นที่อยู่ในช่วงนี้แล้ว\n"
        )

    if budget is not None:
        budget_note += budget_diff_table(context, budget)
    else:
        # ไม่มีงบ -> ต้องสั่งให้ชัด ไม่งั้นโมเดลจะแถว่า "อยู่ในงบ เหลืออีก 0 บาท" ขึ้นมาเอง
        budget_note = (
            "\n**ลูกค้ายังไม่ได้บอกงบ** — ห้ามเอ่ยถึงงบ เงินที่เหลือ หรือส่วนต่างราคาใด ๆ "
            "ในคำตอบเด็ดขาด ช่อง [หมายเหตุราคา] ให้เว้นว่างไว้ เขียนแค่ราคาเฉย ๆ\n"
        )

    suffix = " /no_think" if no_think else ""
    # คำถาม "รุ่นถูก/แพงที่สุด" โมเดลมักเจอคำตอบในเอกสารภาพรวมร้านแล้วตอบสั้นแค่ชื่อกับราคา
    # ต้องวางคำสั่งไว้ "ท้ายสุดติดกับคำถาม" ซึ่งเป็นตำแหน่งที่โมเดลให้น้ำหนักมากที่สุด
    # (วางไว้ก่อนข้อมูลสินค้าแล้วไม่ได้ผล ทดสอบแล้วทำตามแค่ 1 ใน 3 ครั้ง)
    extreme_note = ""
    if _RE_CHEAPEST.search(question) or _RE_PRICIEST.search(question):
        extreme_note = (
            "\n\n(คำถามนี้ถามถึงรุ่นที่ถูก/แพงที่สุด ให้ถือเป็นการแนะนำรุ่น "
            "ต้องตอบด้วยโครงเต็มทุกหัวข้อ: เหตุผล (CPU/GPU/RAM/STORAGE/ราคา), สรุปเบนช์มาร์ก, "
            "ข้อดี, ข้อเสีย โดยสเปกของรุ่นนั้นอยู่ใน 'ข้อมูลสินค้า' ด้านบนแล้ว "
            "ห้ามตอบแค่ชื่อกับราคาเด็ดขาด)"
        )

    # ลูกค้าขอหลายรุ่น/ขอเทียบ -> สั่งให้สลับไปใช้โครงเปรียบเทียบ (วางท้ายสุดเช่นกัน)
    n_models = wanted_model_count(question)
    if n_models:
        extreme_note = (
            f"\n\n(คำถามนี้ลูกค้าขอ {n_models} รุ่นเพื่อเปรียบเทียบ "
            f"**ห้ามตอบแค่รุ่นเดียวเด็ดขาด** ให้ใช้ 'รูปแบบคำตอบเมื่อลูกค้าขอหลายรุ่น' "
            f"คือไล่รุ่นที่ 1-{n_models} พร้อมหัวข้อย่อยครบทุกรุ่น "
            f"แล้วปิดท้ายด้วยหัวข้อ 'สรุปเปรียบเทียบ' "
            f"เลือกจากรุ่นที่อยู่ใน 'ข้อมูลสินค้า' ด้านบนเท่านั้น)"
        )

    # ลูกค้าขอรายชื่อ ("แรม 24 มีตัวไหนบ้าง") และโค้ดกรองรายการให้แล้ว -> ไล่ชื่อให้ครบ
    # ไม่งั้นโครงเปรียบเทียบ 3 รุ่นด้านบนจะตัดเหลือ 3 ตัว ทั้งที่ตรงเงื่อนไข 13 รุ่น
    listed = any(c.startswith(EXACT_NOTE) and "มีทั้งหมด" in c for c in context)
    if HYBRID_FILTERS and listed and _RE_LIST.search(question):
        extreme_note = (
            "\n\n(คำถามนี้ลูกค้าขอรายชื่อรุ่น ให้ไล่ชื่อ **ครบทุกรุ่น** ตามรายการใน 'ผลค้นตรงตามสเปก' "
            "ตามลำดับเดิม บรรทัดละรุ่น พร้อมราคา และสเปกที่ลูกค้าถามถึง ห้ามเพิ่มรุ่นที่ไม่อยู่ในรายการ "
            "ห้ามตัดให้เหลือไม่กี่รุ่น แล้วปิดด้วยสรุปสั้น ๆ 1-2 ประโยค ไม่ต้องใช้โครงแนะนำเต็ม)"
        )

    head = f"{SYSTEM_PROMPT}\n\n" if include_system else ""
    return f"""{head}ประวัติบทสนทนา:
{history_text if history_text else "(เริ่มต้นใหม่)"}
{budget_note}
ข้อมูลสินค้า:
{context_text}

คำถามลูกค้า: {question}{extreme_note}{suffix}"""


EMPTY_ANSWER = "(โมเดลคิดยาวจน token หมดก่อนตอบ ลองถามใหม่อีกครั้งนะ)"


class LLMError(Exception):
    """เรียก LLM ไม่สำเร็จ — ข้อความใช้แสดงให้ผู้ใช้เห็นได้เลย"""


def generate(
    question: str,
    context: list[str],
    history: list,
    budget: int | None = None,
    over_budget: bool = False,
    min_price: int | None = None,
    provider: dict | None = None,
    trace: dict | None = None,
) -> str:
    """โหมด classic: ส่ง prompt ก้อนเดียวไปที่ endpoint แบบ OpenAI ของ provider ที่เลือก
    trace (โหมด debug): บันทึก prompt ทั้งก้อน ค่าที่ส่งไป เวลา โทเคน และความคิดของโมเดล (ถ้ามี)"""
    # เรียกด้วย requests ตรง ๆ (gateway ของ ThaiLLM บล็อก header ที่ openai library แนบมา)
    provider = provider or llm.pick(None, llm.ENGINE_CLASSIC)[0]
    if provider is None:
        raise LLMError("ยังไม่มี AI ที่ใช้งานได้ — ตั้งค่าใน .env / providers.json ก่อน")

    no_think = llm.wants_no_think_suffix(provider, DISABLE_THINKING)
    user_msg = build_prompt(question, context, history, budget, over_budget, min_price,
                            no_think=no_think)
    body = {
        "model": provider["model"],
        "messages": [{"role": "user", "content": user_msg}],
        # 0.1 แทน 0.2 — ยิ่งต่ำยิ่งตอบนิ่ง รูปแบบคำตอบคงเส้นคงวาขึ้น
        "temperature": 0.1,
        # ปิดโหมดคิด -> 512 พอ / เปิดโหมดคิด -> เผื่อ token ให้ส่วนคิดเยอะหน่อย
        "max_tokens": 512 if no_think else provider.get("max_tokens", 1536),
    }
    if trace is not None:
        trace["prompt"] = user_msg
        trace["llm"] = {"provider": provider["label"], "model": provider["model"],
                        "temperature": body["temperature"], "max_tokens": body["max_tokens"]}

    headers = {"Content-Type": "application/json"}
    if provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"

    started = time.perf_counter()
    try:
        resp = requests.post(f"{provider['base_url']}/chat/completions",
                             headers=headers, json=body, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise LLMError(f"{provider['label']} error: {e} — เช็ค API key หรือ endpoint") from e

    data = resp.json()
    choice = data["choices"][0]
    raw = choice["message"].get("content") or ""
    if trace is not None:
        trace["llm"].update(llm_trace(data, raw, time.perf_counter() - started))
    return strip_think(raw) or EMPTY_ANSWER


def llm_trace(data: dict, raw: str, seconds: float) -> dict:
    """ส่วนที่ AI ตอบกลับมาสำหรับโหมด debug — ความคิดของโมเดลมาได้ 2 ทาง:
    ฟิลด์ reasoning แยก (Ollama / vLLM บางรุ่น) หรือบล็อก <think> ในคำตอบ (ThaiLLM)"""
    choice = data["choices"][0]
    message = choice["message"]
    thinking = [message.get(k) for k in ("reasoning", "reasoning_content") if message.get(k)]
    thinking += [t.strip() for t in re.findall(r"<think>(.*?)(?:</think>|$)", raw, flags=re.DOTALL) if t.strip()]
    return {
        "ms": round(seconds * 1000),
        "usage": data.get("usage"),
        "finish_reason": choice.get("finish_reason"),
        "thinking": "\n\n".join(thinking),
        "raw_chars": len(raw),
    }


def generate_pi(
    question: str,
    context: list[str],
    history: list,
    budget: int | None = None,
    over_budget: bool = False,
    min_price: int | None = None,
    provider: dict | None = None,
) -> tuple[str, list[dict]]:
    """โหมด Pi Agent: ส่งให้ pi-agent รัน agent loop — AI เห็นข้อมูลชุดเดียวกับโหมด classic
    แล้วเรียก tool ค้นเพิ่มเองได้ถ้ายังไม่พอ คืน (คำตอบ, รายการ tool ที่ถูกเรียก)

    ส่ง context ที่ค้นมาแล้วไปด้วยเสมอ แทนที่จะให้ agent เริ่มค้นจากศูนย์ — คำตอบจึงไม่มีทาง
    แย่กว่าโหมด classic และคำถามง่าย ๆ ตอบได้ในรอบเดียวโดยไม่เสียเวลาเรียก tool
    """
    try:
        resp = requests.post(
            f"{PI_AGENT_URL}/chat",
            json=pi_request_body(question, context, history, budget, over_budget, min_price, provider),
            timeout=PI_AGENT_TIMEOUT,
        )
    except requests.RequestException as e:
        raise LLMError(f"ติดต่อ pi-agent ไม่ได้ ({e}) — เช็คว่า service pi-agent รันอยู่") from e

    data = _json_body(resp)
    if not resp.ok or data.get("error"):
        raise LLMError(f"{provider['label']} (Pi Agent) error: {data.get('error') or resp.text[:300]}")
    return strip_think(data.get("message") or "") or EMPTY_ANSWER, data.get("tools", [])


def pi_request_body(question, context, history, budget, over_budget, min_price, provider) -> dict:
    """body ที่ส่งให้ pi-agent (/chat และ /chat/stream ใช้ชุดเดียวกัน)"""
    prompt = build_prompt(question, context, history, budget, over_budget, min_price,
                          include_system=False)
    # /no_think ต้องอยู่ท้ายสุดเสมอ จึงต่อเองหลัง PI_TURN_NOTE แทนให้ build_prompt ใส่
    no_think = llm.wants_no_think_suffix(provider, DISABLE_THINKING)
    prompt += PI_TURN_NOTE + (" /no_think" if no_think else "")
    return {
        "provider": llm.agent_config(provider),
        "system_prompt": SYSTEM_PROMPT + PI_TOOL_GUIDE,
        "prompt": prompt,
    }


def _json_body(resp) -> dict:
    return resp.json() if resp.headers.get("content-type", "").startswith("application/json") else {}


def generate_pi_stream(
    question: str,
    context: list[str],
    history: list,
    budget: int | None = None,
    over_budget: bool = False,
    min_price: int | None = None,
    provider: dict | None = None,
):
    """โหมด Pi Agent แบบทยอยส่ง: yield event ของหน้าเว็บ (ดู relay_pi_events) ระหว่าง agent ทำงาน
    event สุดท้ายคือ {"type": "done", "message": คำตอบ, "tools": [...]} — พังเมื่อไหร่ raise LLMError"""
    label = f"{provider['label']} (Pi Agent)"
    try:
        # timeout คู่ (เชื่อมต่อ, รอ byte ถัดไป) — ระหว่างสตรีมนับใหม่ทุกครั้งที่มี event เข้ามา
        resp = requests.post(
            f"{PI_AGENT_URL}/chat/stream",
            json=pi_request_body(question, context, history, budget, over_budget, min_price, provider),
            stream=True,
            timeout=(10, PI_AGENT_TIMEOUT),
        )
    except requests.RequestException as e:
        raise LLMError(f"ติดต่อ pi-agent ไม่ได้ ({e}) — เช็คว่า service pi-agent รันอยู่") from e

    # with: ผู้ใช้ปิดหน้ากลางคัน -> generator ถูกปิด -> ตัดสายกับ pi-agent ซึ่งจะสั่งหยุด agent ต่อ
    with resp:
        if not resp.ok:
            raise LLMError(f"{label} error: {_json_body(resp).get('error') or resp.text[:300]}")
        try:
            yield from relay_pi_events(resp.iter_lines(), label)
        except requests.RequestException as e:
            raise LLMError(f"{label}: การเชื่อมต่อกับ pi-agent ขาดกลางคัน ({e})") from e


# ---- Tool สำหรับ Pi Agent ----
# pi-agent เรียกผ่าน POST /internal/tools/<ชื่อ> แล้วส่งข้อความที่ได้ให้ LLM อ่านต่อ
# ข้อความสเปกใช้เอกสารชุดเดียวกับ vector index (ดึงจาก Chroma) เพื่อให้รูปแบบตรงกับโหมด classic
# และ budget_diff_table() อ่านราคาได้ ส่วนการกรองตามสเปกใช้ laptops.json ที่มีฟิลด์แยกครบ
TOOL_MAX_LIMIT = 15

_laptops: list[dict] | None = None
_spec_doc_by_name: dict[str, str] | None = None


def get_laptops() -> list[dict]:
    global _laptops
    if _laptops is None:
        _laptops = json.loads(LAPTOPS_PATH.read_text(encoding="utf-8"))
    return _laptops


def get_spec_doc_by_name() -> dict[str, str]:
    global _spec_doc_by_name
    if _spec_doc_by_name is None:
        res = get_collection().get(where={"type": {"$eq": "spec"}}, include=["documents", "metadatas"])
        _spec_doc_by_name = {m["name"]: d for m, d in zip(res["metadatas"], res["documents"])}
    return _spec_doc_by_name


def _int_arg(args: dict, key: str) -> int | None:
    """อ่านตัวเลขจาก argument ของ tool — LLM บางตัวส่งมาเป็น string "25,000" """
    raw = args.get(key)
    if raw in (None, ""):
        return None
    try:
        value = int(float(str(raw).replace(",", "")))
    except ValueError:
        raise ValueError(f"{key} ต้องเป็นตัวเลข (ได้ {raw!r})")
    return value if value > 0 else None


def _limit_arg(args: dict, default: int) -> int:
    return max(1, min(_int_arg(args, "limit") or default, TOOL_MAX_LIMIT))


def _ram_gb(ram: str) -> int:
    m = re.search(r"(\d+)\s*GB", ram or "", re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").lower()).strip()


def _price_note(budget: int | None, docs: list[str]) -> str:
    return budget_diff_table(docs, budget) if budget else ""


def tool_search_laptops(args: dict) -> str:
    """ค้นตามความหมาย (สเปก + รีวิว) — ใช้ retrieve() ตัวเดียวกับโหมด classic"""
    query = str(args.get("query") or "").strip()
    if not query:
        raise ValueError("ต้องระบุ query")
    max_price, min_price = _int_arg(args, "max_price"), _int_arg(args, "min_price")
    docs, over = retrieve(query, k_spec=_limit_arg(args, TOP_K_SPEC), k_review=TOP_K_REVIEW,
                          budget=max_price, min_price=min_price)
    docs = [d for d in docs if not is_summary_doc(d)]
    head = (
        "ไม่มีรุ่นไหนอยู่ในช่วงราคาที่ขอ — รายการด้านล่างคือรุ่นที่ใกล้เคียงที่สุด\n" if over else ""
    )
    return head + "\n".join(f"- {with_levels(d)}" for d in docs) + _price_note(max_price, docs)


_SORTS = {
    "price_asc": (lambda lp: lp["ราคา"], False, "ราคาจากถูกไปแพง"),
    "price_desc": (lambda lp: lp["ราคา"], True, "ราคาจากแพงไปถูก"),
    "gpu_score": (lambda lp: lp.get("benchmark", {}).get("GPU Score") or 0, True, "คะแนน GPU สูงสุดก่อน"),
    "cpu_score": (lambda lp: lp.get("benchmark", {}).get("CPU Score") or 0, True, "คะแนน CPU สูงสุดก่อน"),
}
_TEXT_FILTERS = ("cpu", "gpu", "display", "storage", "os")


def tool_filter_laptops(args: dict) -> str:
    """กรองตามสเปกจริงจาก laptops.json — ตอบคำถามแบบ "RTX 4060 มีรุ่นไหนบ้าง" ได้ครบทุกรุ่น
    (vector search คืนแค่ top-k ที่ความหมายใกล้ อาจตกหล่นรุ่นที่ตรงเงื่อนไข)"""
    max_price, min_price = _int_arg(args, "max_price"), _int_arg(args, "min_price")
    min_ram = _int_arg(args, "min_ram_gb")
    min_cpu, min_gpu = _int_arg(args, "min_cpu_score"), _int_arg(args, "min_gpu_score")
    sort_key, reverse, sort_label = _SORTS.get(str(args.get("sort_by") or "price_asc"), _SORTS["price_asc"])
    text_filters = {k: _norm(str(args[k])) for k in _TEXT_FILTERS if args.get(k)}

    def match(lp: dict) -> bool:
        spec, bench = lp.get("spec", {}), lp.get("benchmark", {})
        if max_price and lp["ราคา"] > max_price:
            return False
        if min_price and lp["ราคา"] < min_price:
            return False
        if min_ram and _ram_gb(spec.get("RAM", "")) < min_ram:
            return False
        if min_cpu and (bench.get("CPU Score") or 0) < min_cpu:
            return False
        if min_gpu and (bench.get("GPU Score") or 0) < min_gpu:
            return False
        return all(want in _norm(spec.get(k.upper(), "")) for k, want in text_filters.items())

    found = sorted((lp for lp in get_laptops() if match(lp)), key=sort_key, reverse=reverse)
    if not found:
        return "ไม่พบรุ่นที่ตรงทุกเงื่อนไขในข้อมูลที่มี (ลองผ่อนเงื่อนไขบางข้อ หรือใช้ search_laptops)"

    shown = found[:_limit_arg(args, 8)]
    by_name = get_spec_doc_by_name()
    docs = [by_name.get(lp["ชื่อสินค้า"], lp["ชื่อสินค้า"]) for lp in shown]
    return (
        f"พบ {len(found)} รุ่นที่ตรงเงื่อนไข แสดง {len(shown)} รุ่น เรียงตาม{sort_label}:\n"
        + "\n".join(f"- {with_levels(d)}" for d in docs)
        + _price_note(max_price, docs)
    )


def tool_get_laptop_details(args: dict) -> str:
    """สเปกเต็ม + รีวิวทั้งหมดของรุ่นที่ระบุชื่อ (จับคู่แบบไม่สนตัวพิมพ์ เอาเฉพาะชื่อที่มีทุกคำที่ค้น)"""
    name = _norm(str(args.get("name") or ""))
    if not name:
        raise ValueError("ต้องระบุ name")
    words = name.split()
    names = [lp["ชื่อสินค้า"] for lp in get_laptops()]
    hits = [n for n in names if all(w in _norm(n) for w in words)]
    # ชื่อที่มีคำค้นต่อกันเป็นก้อนเดียวมาก่อน ("legion 5" ต้องเจอ Legion 5 ก่อน Legion 9i ที่บังเอิญมีเลข 5)
    hits.sort(key=lambda n: (name not in _norm(n), len(n)))
    if not hits:
        # ไม่เจอตรง ๆ -> เสนอชื่อที่มีคำซ้ำมากที่สุดให้ AI เลือกถามลูกค้าต่อ
        scored = sorted(names, key=lambda n: -sum(w in _norm(n) for w in words))[:5]
        return f"ไม่พบรุ่นที่ชื่อตรงกับ \"{name}\" ในข้อมูล ชื่อที่ใกล้เคียง: " + " / ".join(scored)

    by_name = get_spec_doc_by_name()
    parts = []
    for n in hits[:3]:
        res = get_collection().get(
            where={"$and": [{"type": {"$eq": "review"}}, {"name": {"$eq": n}}]}, include=["documents"]
        )
        reviews = "\n".join(f"  - {r}" for r in res["documents"]) or "  - (ไม่มีรีวิว)"
        parts.append(f"- {with_levels(by_name.get(n, n))}\n  รีวิวจากผู้ใช้:\n{reviews}")
    more = f"\n(ยังมีอีก {len(hits) - 3} รุ่นที่ชื่อตรง ระบุชื่อให้ละเอียดขึ้นถ้าต้องการ)" if len(hits) > 3 else ""
    return "\n".join(parts) + more


TOOLS = {
    "search_laptops": tool_search_laptops,
    "filter_laptops": tool_filter_laptops,
    "get_laptop_details": tool_get_laptop_details,
}


# ---- สตรีมความคืบหน้าไปที่หน้าเว็บ (/chat/stream) ----
# โหมด Pi Agent ใช้เวลา 8-18 วิ (เรียก tool หลายรอบ) จึงทยอยส่ง event เป็น Server-Sent Events
# แต่ละ event คือบรรทัด "data: {json}" ตามด้วยบรรทัดว่าง:
#   {"type": "status",  "text": ...}   ระบบกำลังทำอะไร เช่น "กำลังกรองสเปก: RTX 4060 · ไม่เกิน 40,000 บาท…"
#   {"type": "delta",   "text": ...}   ข้อความคำตอบที่เพิ่งพิมพ์ต่อท้าย
#   {"type": "replace", "text": ...}   ทิ้งข้อความที่โชว์ไปแล้วแล้วใช้ค่านี้แทน (เจอ <think> / AI ขึ้นข้อความใหม่หลังเรียก tool)
#   {"type": "done",    "message": ..., "meta": ..., "commit": ...}   คำตอบสุดท้ายที่ต้องใช้จริง
#   {"type": "error",   "error": ...}
TOOL_STATUS = {
    "filter_laptops": "กำลังกรองสเปก",
    "search_laptops": "กำลังค้นสเปกและรีวิว",
    "get_laptop_details": "กำลังดูรายละเอียดรุ่น",
}
STATUS_SEARCHING = "กำลังค้นข้อมูลในร้าน…"
STATUS_AGENT = "AI กำลังอ่านข้อมูลที่ค้นมา…"
STATUS_WRITING = "กำลังเรียบเรียงคำตอบ…"


def _safe_int_arg(args: dict, key: str) -> int | None:
    try:
        return _int_arg(args, key)
    except ValueError:
        return None


def tool_status(name: str, args) -> str:
    """ข้อความสถานะตอน agent เรียก tool — มีเงื่อนไขที่ AI ส่งมาด้วย ผู้ใช้จะได้เห็นว่ามันค้นอะไรอยู่"""
    args = args if isinstance(args, dict) else {}
    label = TOOL_STATUS.get(name, f"กำลังเรียกเครื่องมือ {name}")
    parts = []
    query = str(args.get("query") or "").strip()
    if query:
        parts.append(f"“{query}”")
    for key in ("name", "gpu", "cpu", "display", "storage", "os"):
        value = str(args.get(key) or "").strip()
        if value:
            parts.append(value)
    ram = _safe_int_arg(args, "min_ram_gb")
    if ram:
        parts.append(f"RAM {ram}GB ขึ้นไป")
    lo, hi = _safe_int_arg(args, "min_price"), _safe_int_arg(args, "max_price")
    if lo and hi:
        parts.append(f"{lo:,}–{hi:,} บาท")
    elif hi:
        parts.append(f"ไม่เกิน {hi:,} บาท")
    elif lo:
        parts.append(f"ตั้งแต่ {lo:,} บาท")
    detail = " · ".join(parts)
    if len(detail) > 80:
        detail = detail[:79] + "…"
    return f"{label}: {detail}…" if detail else f"{label}…"


_THINK_OPEN = "<think>"


def visible_answer(raw: str) -> str:
    """ส่วนของคำตอบที่โชว์ระหว่างสตรีมได้ — ตัด <think> แบบเดียวกับคำตอบสุดท้าย (บล็อกที่ยังคิดไม่จบ
    ก็ตัด) และกั๊กท้ายที่อาจเป็นแท็กเปิดที่มายังไม่ครบ เช่น "<thi" ไม่งั้นแท็กจะโผล่ให้เห็นแวบหนึ่ง"""
    text = strip_think(raw)
    for n in range(len(_THINK_OPEN) - 1, 0, -1):
        if text.endswith(_THINK_OPEN[:n]):
            return text[:-n].rstrip()
    return text


def relay_pi_events(lines, label: str = "Pi Agent"):
    """แปลง NDJSON จาก pi-agent /chat/stream เป็น event ของหน้าเว็บ (รูปแบบอยู่ด้านบน)
    จบด้วย {"type": "done", "message": คำตอบที่ strip_think แล้ว, "tools": [...]}
    pi-agent แจ้ง error หรือสายขาดก่อนจบ -> raise LLMError"""
    raw = shown = ""
    running: dict[str, str] = {}   # tool ที่ยังทำงานอยู่ (id -> ข้อความสถานะ)
    for line in lines:
        if not line:
            continue
        try:
            ev = json.loads(line)
        except ValueError:
            continue
        kind = ev.get("type")

        if kind == "message_start":
            # AI ขึ้นข้อความใหม่ (เช่น หลังได้ผลจาก tool) — ข้อความรอบก่อนไม่ใช่คำตอบสุดท้าย
            raw = ""
            if shown:
                shown = ""
                yield {"type": "replace", "text": ""}
        elif kind == "text":
            raw += ev.get("delta") or ""
            text = visible_answer(raw)
            if text == shown:
                continue
            # strip_think อาจตัดของที่โชว์ไปแล้วทิ้ง (เพิ่งเจอ </think>) -> ส่งข้อความทั้งก้อนแทน
            if text.startswith(shown):
                yield {"type": "delta", "text": text[len(shown):]}
            else:
                yield {"type": "replace", "text": text}
            shown = text
        elif kind == "tool_start":
            status = tool_status(ev.get("name") or "", ev.get("args"))
            running[str(ev.get("id"))] = status
            yield {"type": "status", "text": status}
        elif kind == "tool_end":
            running.pop(str(ev.get("id")), None)
            yield {"type": "status", "text": next(reversed(running.values()), STATUS_WRITING)}
        elif kind == "done":
            yield {
                "type": "done",
                "message": strip_think(ev.get("message") or "") or EMPTY_ANSWER,
                "tools": ev.get("tools") or [],
            }
            return
        elif kind == "error":
            raise LLMError(f"{label} error: {ev.get('error') or 'ไม่ทราบสาเหตุ'}")
    raise LLMError(f"{label}: pi-agent ปิดการเชื่อมต่อก่อนตอบเสร็จ")


def sse(event: dict) -> str:
    # json.dumps escape การขึ้นบรรทัดใหม่ในข้อความให้แล้ว หนึ่ง event จึงเป็นบรรทัดเดียวเสมอ
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


# ---- บันทึกประวัติหลังสตรีมจบ ----
# session ของ Flask อยู่ใน cookie ซึ่งต้องส่งไปพร้อม header — แต่กว่า /chat/stream จะรู้คำตอบครบ
# header ถูกส่งไปนานแล้ว จึงแนบ "commit token" ไปกับ event done ให้หน้าเว็บยิงกลับมาที่ /chat/commit
# token เซ็นด้วย secret_key: ผู้ใช้แก้เนื้อหาคำตอบที่จะเข้าไปอยู่ใน context ของ LLM รอบหน้าไม่ได้
COMMIT_MAX_AGE = 600   # วินาที


def _commit_serializer() -> URLSafeTimedSerializer:
    return URLSafeTimedSerializer(app.secret_key, salt="chat-commit")


def commit_token(question: str, answer: str) -> str:
    return _commit_serializer().dumps({"q": question[:MAX_STORED_CHARS], "a": answer[:MAX_STORED_CHARS]})


def read_commit_token(token: str) -> tuple[str, str] | None:
    """(คำถาม, คำตอบ) จาก token — ถูกแก้ / หมดอายุ / รูปแบบผิด = None"""
    try:
        turn = _commit_serializer().loads(token, max_age=COMMIT_MAX_AGE)
    except BadSignature:   # SignatureExpired เป็นลูกของ BadSignature
        return None
    if not isinstance(turn, dict) or not isinstance(turn.get("q"), str) or not isinstance(turn.get("a"), str):
        return None
    return turn["q"], turn["a"]


def save_turn(history: list, question: str, answer: str) -> None:
    """บันทึกประวัติลง session — Flask เก็บ session ไว้ใน cookie ซึ่งเบราว์เซอร์ตัดทิ้งถ้าเกิน
    4096 bytes ดังนั้นต้องจำกัดทั้งจำนวนข้อความและความยาวต่อข้อความ ไม่งั้นประวัติจะหายเงียบ ๆ"""
    history.append({"role": "user", "content": question[:MAX_STORED_CHARS]})
    history.append({"role": "assistant", "content": answer[:MAX_STORED_CHARS]})
    session["history"] = history[-MAX_HISTORY_MSGS:]


@app.route("/internal/tools/<name>", methods=["POST"])
def internal_tool(name):
    """endpoint ให้ pi-agent เรียก tool — ต้องแนบ X-Internal-Token ที่ตรงกับ INTERNAL_TOKEN"""
    token = request.headers.get("X-Internal-Token", "")
    if not INTERNAL_TOKEN or not hmac.compare_digest(token, INTERNAL_TOKEN):
        return jsonify({"error": "forbidden"}), 403
    fn = TOOLS.get(name)
    if fn is None:
        return jsonify({"error": f"ไม่มี tool ชื่อ {name}"}), 404

    args = request.get_json(silent=True)
    try:
        return jsonify({"text": fn(args if isinstance(args, dict) else {})})
    except ValueError as e:
        return jsonify({"error": str(e)}), 400


@app.context_processor
def static_version():
    """เลขเวอร์ชันต่อท้าย URL ของ CSS/JS ในหน้าเว็บ (= เวลาแก้ไฟล์ล่าสุด)
    Cloudflare สั่งเบราว์เซอร์แคชไฟล์ static ไว้ 4 ชม. แม้แอปจะส่ง no-cache — ไม่มีเลขนี้
    คนที่เคยเปิดเว็บจะได้ JS/CSS ตัวเก่าไปอีกหลายชั่วโมงหลังอัปเดต"""
    def static_v(path: str) -> int:
        try:
            return int((BASE_DIR / "static" / path).stat().st_mtime)
        except OSError:
            return 0
    return {"static_v": static_v}


@app.route("/")
def index():
    # หน้าหลัก — ไม่ต้องเตรียม session ที่นี่ เพราะทุกจุดที่อ่านประวัติใช้ .get() ที่มีค่า default
    # อยู่แล้ว และหน้าเว็บจะยิง /new_chat ตอนโหลดเพื่อล้าง session ให้ตรงกับจอว่างอยู่แล้ว
    return render_template("index.html")


def _chat_request():
    """ตรวจ request ของ /chat และ /chat/stream (รูปแบบ, ความยาว, AI ที่เลือก, rate limit)
    แล้วอ่านงบจากข้อความ คืน (ข้อมูลที่พร้อมใช้, None) หรือ (None, response error)

    งบต้องเขียนลง session ที่นี่ ก่อนเริ่มตอบ — /chat/stream ส่ง header (รวม cookie) ไปตั้งแต่ event แรก
    """
    # silent=True: body ที่ไม่ใช่ JSON ต้องได้ error เป็น JSON ไม่ใช่หน้า HTML 400
    # (ไม่งั้น fetch ฝั่งหน้าเว็บจะ parse ไม่ได้แล้วขึ้นว่า "เชื่อมต่อเซิร์ฟเวอร์ไม่ได้" ซึ่งทำให้เข้าใจผิด)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return None, (jsonify({"error": "ส่งข้อมูลมาไม่ถูกรูปแบบ (ต้องเป็น JSON)"}), 400)

    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return None, (jsonify({"error": "empty message"}), 400)
    if len(user_msg) > MAX_MSG_LEN:
        # กันข้อความยาวผิดปกติมาถ่วงเธรด (gunicorn มีแค่ 1 worker)
        return None, (jsonify({"error": f"ข้อความยาวเกินไป (สูงสุด {MAX_MSG_LEN:,} ตัวอักษร)"}), 413)

    # AI และโหมดที่ผู้ใช้เลือกจากหน้าเว็บ (ไม่ส่งมา = ใช้ค่าเริ่มต้นใน .env)
    provider, engine, pick_error = llm.pick(data.get("provider"), data.get("engine"))
    if pick_error:
        return None, (jsonify({"error": pick_error}), 400)

    if too_many_requests(client_ip()):
        return None, (jsonify({
            "error": f"ถามถี่เกินไปครับ (จำกัด {RATE_LIMIT_PER_MIN} ข้อความต่อนาที) "
                     f"รอสักครู่แล้วลองใหม่นะครับ"
        }), 429)

    history = session.get("history", [])

    # หางบจากข้อความล่าสุดก่อน ถ้าไม่มีค่อยใช้งบที่เคยจำไว้ใน session
    # (เช่น รอบก่อนบอก "งบ 20000" รอบนี้ถามแค่ "มีรุ่นอื่นอีกไหม" ก็ยังต้องอยู่ในงบเดิม)
    # เก็บงบแยกจาก history เพราะ history ถูกตัดให้สั้นเพื่อคุมขนาด cookie
    budget = extract_budget(user_msg)
    min_price = extract_min_price(user_msg)
    budget_from = "message" if budget is not None else None
    if budget is None:
        # ข้อความนี้ไม่ได้บอกราคา -> ใช้ของเดิมทั้งคู่ (ขอบบนกับขอบล่างต้องไปด้วยกันเสมอ
        # ไม่งั้นถาม "อยากดูรุ่นอื่น" ต่อจาก "ราคา 30,000-40,000" จะหลุดช่วงไปทั้งหมด)
        budget = session.get("budget")
        min_price = session.get("budget_min")
        budget_from = "session" if budget is not None else None
    else:
        session["budget"] = budget
        if min_price is None:
            session.pop("budget_min", None)
        else:
            session["budget_min"] = min_price

    return {"message": user_msg, "provider": provider, "engine": engine, "history": history,
            "budget": budget, "min_price": min_price, "budget_from": budget_from,
            # ขอดูกระบวนการคิดได้เฉพาะตอนผู้ดูแลเปิด DEBUG_PANEL ไว้ (ปิดอยู่ = ไม่สนค่าที่ส่งมา)
            "debug": DEBUG_PANEL and data.get("debug") is True}, None


def new_trace(job: dict) -> dict | None:
    """จุดเริ่มของข้อมูล debug: ระบบตีความคำถามว่าอะไร ก่อนเริ่มค้น (None = ไม่ได้ขอ debug)"""
    if not job["debug"]:
        return None
    q = job["message"]
    return {
        "engine": job["engine"],
        "query": {
            "budget": job["budget"],
            "min_price": job["min_price"],
            "budget_from": job["budget_from"],
            "model_count": wanted_model_count(q),
            "cheapest": bool(_RE_CHEAPEST.search(q)),
            "priciest": bool(_RE_PRICIEST.search(q)),
            "gpu_terms": gpu_terms(q),
            "asked_models": asked_model_names(q),
            "history_msgs": min(len(job["history"]), HISTORY_FOR_LLM),
        },
    }


def chat_meta(job: dict, tools_used: list[dict], started: float) -> dict:
    # หน้าเว็บแสดงบรรทัดเล็ก ๆ ใต้คำตอบว่าใครตอบ ด้วยโหมดไหน ค้นเพิ่มกี่ครั้ง
    provider = job["provider"]
    return {
        "provider": provider["id"],
        "provider_label": provider.get("short") or provider["label"],
        "engine": job["engine"],
        "tools": [t.get("name") for t in tools_used],
        "elapsed": round(time.time() - started, 1),
    }


@app.route("/chat", methods=["POST"])
def chat():
    # endpoint หลัก: รับข้อความ -> ค้นหา (retrieve) -> แต่งคำตอบ (generate) -> ตอบเป็น JSON ก้อนเดียว
    # (หน้าเว็บใช้ /chat/stream ที่ทยอยส่งความคืบหน้า — อันนี้ไว้ให้สคริปต์ / ชุดทดสอบเรียกง่าย ๆ)
    job, error = _chat_request()
    if error:
        return error
    user_msg, provider, history = job["message"], job["provider"], job["history"]
    budget, min_price = job["budget"], job["min_price"]

    # จับเวลาตั้งแต่ก่อนค้น — เวลาใน meta ต้องตรงกับที่ผู้ใช้รอจริง (ค้น + LLM)
    started = time.time()
    trace = new_trace(job)
    debug_part = {"debug": trace} if trace is not None else {}
    try:
        context, over_budget = retrieve(user_msg, budget=budget, min_price=min_price, trace=trace)
    except Exception as e:
        return jsonify({"error": f"Vector search error: {e}", **debug_part}), 500

    tools_used: list[dict] = []
    try:
        if job["engine"] == llm.ENGINE_PI:
            answer, tools_used = generate_pi(user_msg, context, history, budget=budget,
                                             over_budget=over_budget, min_price=min_price,
                                             provider=provider)
        else:
            answer = generate(user_msg, context, history, budget=budget,
                              over_budget=over_budget, min_price=min_price, provider=provider,
                              trace=trace)
    except LLMError as e:
        return jsonify({"error": str(e), **debug_part}), 502
    except Exception as e:
        return jsonify({"error": f"LLM error: {e}", **debug_part}), 500

    save_turn(history, user_msg, answer)
    return jsonify({"message": answer, "meta": chat_meta(job, tools_used, started), **debug_part})


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """เหมือน /chat แต่ตอบเป็น Server-Sent Events: บอกสถานะระหว่างทาง (ค้นข้อมูล / เรียก tool)
    และทยอยส่งข้อความคำตอบของโหมด Pi Agent ตามที่ AI พิมพ์ (โหมด classic ส่งคำตอบทีเดียวตอนจบ)

    error ที่รู้ก่อนเริ่มตอบ (JSON พัง, ถามถี่เกิน ...) ตอบเป็น JSON + status code แบบเดียวกับ /chat
    ส่วน error ระหว่างสตรีมส่งเป็น event {"type": "error"} เพราะ status 200 ถูกส่งไปแล้ว
    ประวัติแชทบันทึกผ่าน /chat/commit (ดู commit_token)
    """
    job, error = _chat_request()
    if error:
        return error
    user_msg, provider, history = job["message"], job["provider"], job["history"]
    budget, min_price = job["budget"], job["min_price"]

    # ทุกอย่างที่ต้องใช้จาก request/session อ่านมาครบแล้ว — generator นี้ทำงานหลังส่ง header
    # จึงไม่แตะ session อีก (เขียนไปก็ไม่ถูกบันทึก)
    def events():
        started = time.time()
        # ขอดูกระบวนการคิด -> แนบไปกับ event สุดท้าย (done หรือ error) ทั้งก้อน
        trace = new_trace(job)
        debug_part = {"debug": trace} if trace is not None else {}
        yield sse({"type": "status", "text": STATUS_SEARCHING})
        try:
            context, over_budget = retrieve(user_msg, budget=budget, min_price=min_price, trace=trace)
        except Exception as e:
            yield sse({"type": "error", "error": f"Vector search error: {e}", **debug_part})
            return

        tools_used: list[dict] = []
        try:
            if job["engine"] == llm.ENGINE_PI:
                yield sse({"type": "status", "text": STATUS_AGENT})
                for ev in generate_pi_stream(user_msg, context, history, budget=budget,
                                             over_budget=over_budget, min_price=min_price,
                                             provider=provider):
                    if ev["type"] == "done":
                        answer, tools_used = ev["message"], ev["tools"]
                    else:
                        yield sse(ev)
            else:
                yield sse({"type": "status", "text": STATUS_WRITING})
                answer = generate(user_msg, context, history, budget=budget,
                                  over_budget=over_budget, min_price=min_price, provider=provider,
                                  trace=trace)
        except LLMError as e:
            yield sse({"type": "error", "error": str(e), **debug_part})
            return
        except Exception as e:
            yield sse({"type": "error", "error": f"LLM error: {e}", **debug_part})
            return

        if trace is not None and tools_used:
            # โหมดค้นละเอียด: prompt/เวลาของ AI อยู่ฝั่ง pi-agent — บอกได้แค่ว่าค้นเพิ่มอะไรไปบ้าง
            trace["tools"] = tools_used
        yield sse({
            "type": "done",
            "message": answer,
            "meta": chat_meta(job, tools_used, started),
            "commit": commit_token(user_msg, answer),
            **debug_part,
        })

    return Response(events(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # บอก nginx ไม่ให้กั๊ก response ไว้รอจนครบ
    })


@app.route("/chat/commit", methods=["POST"])
def chat_commit():
    """บันทึกคำถาม-คำตอบจาก /chat/stream ลง session (หน้าเว็บยิงมาทันทีหลังได้ event done)"""
    data = request.get_json(silent=True)
    turn = read_commit_token(data.get("token")) if isinstance(data, dict) and isinstance(
        data.get("token"), str) else None
    if turn is None:
        return jsonify({"error": "token ไม่ถูกต้องหรือหมดอายุ"}), 400

    question, answer = turn
    history = session.get("history", [])
    # ยิงซ้ำ (เช่น เน็ตกระตุกแล้วหน้าเว็บลองใหม่) ไม่ต้องบันทึกซ้ำ
    if history[-2:] != [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]:
        save_turn(history, question, answer)
    return jsonify({"ok": True})


@app.route("/providers")
def providers():
    """รายการ AI ที่เลือกได้ (ไม่มี API key หลุดออกไป) — หน้าเว็บใช้สร้างเมนูเปลี่ยน AI
    debug = เปิดแผงดูกระบวนการคิดไว้ไหม (หน้าเว็บโชว์ปุ่ม Debug เฉพาะตอนเปิด)"""
    return jsonify({**llm.public_list(), "debug": DEBUG_PANEL})


@app.route("/debug", methods=["POST"])
def debug():
    """ดู "ข้อมูลดิบ" ที่ระบบหามาได้ ก่อนถูกส่งไปให้ ThaiLLM แต่งเป็นคำตอบ
    ไม่เรียก LLM เลย จึงตอบทันทีและไม่เปลืองโควตา

    ใช้ตอบคำถามว่า "ที่บอทตอบแบบนี้ เพราะมันเห็นข้อมูลอะไร"
    เปิดใช้ด้วย ENABLE_DEBUG=true (ปิดไว้เป็นค่าเริ่มต้น เพราะ endpoint นี้เปิดเผย prompt ทั้งก้อน)
    """
    if not ENABLE_DEBUG:
        return jsonify({"error": "ปิดอยู่ — ตั้ง ENABLE_DEBUG=true ใน .env เพื่อเปิด"}), 403

    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "ส่งข้อมูลมาไม่ถูกรูปแบบ (ต้องเป็น JSON)"}), 400
    question = (data.get("message") or "").strip()
    if not question:
        return jsonify({"error": "empty message"}), 400

    budget = extract_budget(question)
    min_price = extract_min_price(question)
    if budget is None:
        budget = session.get("budget")
        min_price = session.get("budget_min")

    try:
        context, over_budget = retrieve(question, budget=budget, min_price=min_price)
    except Exception as e:
        return jsonify({"error": f"Vector search error: {e}"}), 500

    # แยกประเภทเอกสารให้ดูง่าย
    specs, reviews, summary = [], [], []
    for d in context:
        if is_summary_doc(d):
            summary.append(d)
        elif d.startswith("รีวิวจากผู้ใช้รุ่น"):
            reviews.append(d)
        else:
            specs.append(d)

    return jsonify({
        "คำถาม": question,
        "สิ่งที่ระบบตีความได้": {
            "งบที่อ่านได้ (เพดานบน)": budget,
            "ราคาขั้นต่ำ (ถ้าถามเป็นช่วง)": min_price,
            "ไม่มีรุ่นไหนในช่วงราคา": over_budget,
            "ขอกี่รุ่น": wanted_model_count(question),
            "ถามถึงรุ่นถูกสุด": bool(_RE_CHEAPEST.search(question)),
            "ถามถึงรุ่นแพงสุด": bool(_RE_PRICIEST.search(question)),
        },
        "เอกสารที่ดึงมา": {
            "รวม": len(context),
            "สเปก": specs,
            "รีวิว": reviews,
            "ภาพรวมร้าน": summary,
        },
        "ตารางส่วนต่างงบ": budget_diff_table(context, budget) if budget else None,
        "prompt_ที่จะส่งให้_LLM": build_prompt(
            question, context, session.get("history", []), budget, over_budget, min_price
        ),
    })


@app.route("/new_chat", methods=["POST"])
def new_chat():
    # ล้างประวัติบทสนทนาใน session เพื่อเริ่มสนทนาใหม่ (ล้างงบที่จำไว้ด้วย)
    session["history"] = []
    session.pop("budget", None)
    # ขอบล่างต้องล้างคู่กับงบเสมอ ไม่งั้นแชทใหม่ที่ไม่บอกงบจะโดนกรองด้วยราคาขั้นต่ำของแชทเก่า
    session.pop("budget_min", None)
    return jsonify({"ok": True})


@app.route("/restore", methods=["POST"])
def restore():
    """โหลดบทสนทนาเก่าที่หน้าเว็บเก็บไว้ใน localStorage กลับเข้า session ฝั่งเซิร์ฟเวอร์
    เพื่อให้กดดูแชทเก่าจาก sidebar แล้วคุยต่อได้จริง (ของเดิมกดแล้วไม่เกิดอะไรขึ้น)

    ข้อมูลมาจากฝั่งผู้ใช้ จึงต้องตรวจรูปแบบและตัดความยาวก่อนเสมอ —
    ไม่งั้นจะกลายเป็นช่องยัดข้อความปลอมในบท "ผู้ช่วย" เข้าไปใน context ของ LLM
    """
    data = request.get_json(silent=True)
    if not isinstance(data, dict) or not isinstance(data.get("history"), list):
        return jsonify({"error": "ส่งข้อมูลมาไม่ถูกรูปแบบ"}), 400

    clean = [
        {"role": m["role"], "content": str(m["content"])[:MAX_STORED_CHARS]}
        for m in data["history"][-MAX_HISTORY_MSGS:]
        if isinstance(m, dict) and m.get("role") in ("user", "assistant") and m.get("content")
    ]
    session["history"] = clean

    # ตั้งงบใหม่จากข้อความล่าสุดของผู้ใช้ในบทสนทนาที่โหลดกลับมา
    session.pop("budget", None)
    session.pop("budget_min", None)
    for m in reversed(clean):
        if m["role"] == "user":
            found = extract_budget(m["content"])
            if found is not None:
                session["budget"] = found
                min_price = extract_min_price(m["content"])
                if min_price is not None:
                    session["budget_min"] = min_price
                break

    return jsonify({"ok": True, "restored": len(clean)})


if __name__ == "__main__":
    if not CHROMA_DIR.exists():
        print("ERROR: ไม่พบ chroma_db — รัน `python build_index.py` ก่อนนะ")
    else:
        # debug=True เปิด Werkzeug debugger ซึ่งรันโค้ดจากเบราว์เซอร์ได้ — ห้ามติดไปถึง production
        app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", port=5000)
