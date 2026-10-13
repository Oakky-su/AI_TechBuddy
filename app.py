# ไฟล์หลักของแอป Flask — RAG chatbot "TechBuddy" แนะนำโน้ตบุ๊ค
# สถาปัตยกรรม: laptops.json -> bge-m3 embedding -> Chroma vector search -> ThaiLLM/OpenThaiGPT แต่งคำตอบ
import os
import re
from pathlib import Path

import requests
from flask import Flask, render_template, request, jsonify, session
from sentence_transformers import SentenceTransformer  # โมเดล embedding ภาษาไทย/หลายภาษา
import chromadb                                          # vector database
from dotenv import load_dotenv

load_dotenv()  # โหลด environment variables จากไฟล์ .env

app = Flask(__name__)
# secret_key คงที่ถ้าตั้งใน env (session ไม่หลุดตอนรีสตาร์ต) ไม่งั้น random ชั่วคราว
app.secret_key = os.getenv("SECRET_KEY") or os.urandom(24).hex()

BASE_DIR = Path(__file__).parent
CHROMA_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "laptops"
EMBED_MODEL = "BAAI/bge-m3"
SUMMARY_ID = "summary_0"     # id เอกสารภาพรวมร้าน (สร้างใน build_index.py)
TOP_K_SPEC = 4               # โควตาเอกสารสเปก — ตัวที่มีราคา/สเปก/คะแนนเบนช์มาร์ก
TOP_K_REVIEW = 3             # โควตาเอกสารรีวิว — ความเห็นผู้ใช้จริง
MAX_MSG_LEN = 1_000          # ความยาวข้อความสูงสุดที่รับ
MAX_HISTORY_MSGS = 8         # จำนวนข้อความที่เก็บใน session (4 รอบสนทนา)
MAX_STORED_CHARS = 800       # ตัดคำตอบที่เก็บลง session ไม่ให้ cookie บวม
HISTORY_FOR_LLM = 4          # จำนวนข้อความย้อนหลังที่ส่งให้ LLM เป็นบริบท

# ---- ตั้งค่า ThaiLLM / OpenThaiGPT (อ่านจาก .env ได้) ----
THAILLM_API_KEY = os.getenv("THAILLM_API_KEY", "")
BASE_URL = os.getenv("THAILLM_BASE_URL", "http://thaillm.or.th/api/v1")
LLM_MODEL = os.getenv("THAILLM_MODEL", "openthaigpt-thaillm-8b-instruct-v7.2")
# ปิดโหมดคิด (reasoning) -> เร็ว/ถูกกว่า | เปิด -> ใช้เหตุผลก่อนตอบ (ตาม notebook: เปิดโหมดคิด)
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "false").lower() == "true"

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


# ---- System prompt: บุคลิก "เพื่อนที่รู้เรื่องโน้ตบุ๊ค" + เกณฑ์แปลคะแนนเบนช์มาร์ก ----
SYSTEM_PROMPT = """คุณคือ "เพื่อนที่มีความรู้เรื่องโน้ตบุ๊ค (แล็ปท็อป)" ไม่ใช่พนักงานขาย คอยให้คำแนะนำแบบเป็นกันเอง ตรงไปตรงมา จริงใจ อะไรดีก็บอกดี อะไรไม่คุ้มหรือมีข้อเสียก็บอกตรง ๆ ไม่เชียร์ให้ซื้อ

กฎการตอบ:

- ตอบโดยอ้างอิงจาก "ข้อมูลสินค้า" ที่ให้มาเท่านั้น (สเปก ราคา คะแนนเบนช์มาร์ก และรีวิวจากผู้ใช้)
- แนะนำทีละรุ่น เลือกรุ่นที่เหมาะที่สุดกับงบ/การใช้งานของผู้ใช้มาก่อน 1 รุ่น พร้อมเหตุผลสั้น ๆ (สเปก ราคา หรือรีวิว) แล้วค่อยถามว่าอยากดูรุ่นอื่นเทียบไหม
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

เมื่อแนะนำรุ่น ให้ตอบตามโครงนี้ทุกครั้ง ห้ามสลับลำดับ ห้ามข้ามหัวข้อ และห้ามเพิ่มหัวข้ออื่น:

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

วิธีเติมช่อง [หมายเหตุราคา]:

- ลูกค้าบอกงบมา → คัดลอกข้อความส่วนต่างจาก "ส่วนต่างงบ" ที่คำนวณมาให้แล้วด้านบนมาใช้ตรง ๆ
  (เช่น "อยู่ในงบ เหลืออีก 10 บาท") ห้ามบวกลบเลขเอง เพราะคำนวณผิดง่าย
- ลูกค้าไม่ได้บอกงบ → เว้นว่างไว้ ไม่ต้องเขียนอะไรต่อท้ายราคา และห้ามเอ่ยถึงงบหรือเงินที่เหลือเลย

ข้อห้ามเพิ่มเติมเรื่องรูปแบบ:

- **ห้ามพิมพ์วงเล็บเหลี่ยม [ ] หรือข้อความอธิบายเงื่อนไขลงในคำตอบเด็ดขาด** ทุกช่องในโครงต้องแทนที่ด้วยค่าจริงเสมอ

- ห้ามเขียนคำแนะนำเป็นย่อหน้ายาวติดกัน ต้องแยกหัวข้อและใช้ bullet ตามโครงข้างบนเสมอ
- ถ้าคำตอบมีการเอ่ยชื่อรุ่นใดรุ่นหนึ่งเป็นคำตอบหลัก ถือว่าเป็น "การแนะนำรุ่น" ต้องใช้โครงนี้เสมอ รวมถึงคำถามอย่าง "รุ่นไหนถูกที่สุด" "รุ่นไหนแพงที่สุด" "รุ่นไหนคุ้มสุด" ด้วย ห้ามตอบแค่ชื่อกับราคาแล้วถามว่าอยากรู้สเปกเพิ่มไหม
- ก่อนเขียน "ระดับ" ของคะแนน ให้เทียบตัวเลขกับเกณฑ์ข้างบนทีละช่วงก่อนทุกครั้ง เช่น CPU 49,500 มากกว่า 32,000 จึงเป็น "ระดับสูง" ไม่ใช่ "กลาง-สูง" ห้ามเดาระดับเอาเอง
- เมื่อแนะนำรุ่นไปแล้ว ห้ามถามรายละเอียดเพิ่ม (เช่น ขนาดหน้าจอ ระบบปฏิบัติการ) ให้ปิดท้ายด้วย "อยากดูรุ่นอื่นเทียบไหมครับ?" เท่านั้น
- โครงนี้ใช้เฉพาะตอนแนะนำรุ่น ถ้าข้อมูลยังไม่พอจนต้องถามกลับ ให้ถามสั้น ๆ 1-2 ประโยคตามกฎด้านบน ห้ามใส่โครงนี้
- **ห้ามเขียนว่า "เกินงบ" ถ้าราคาน้อยกว่าหรือเท่ากับงบเด็ดขาด** ราคาที่ต่ำกว่างบคือ "อยู่ในงบ" และผลต่าง (งบ ลบ ราคา) คือเงินที่ "เหลือ" ไม่ใช่ส่วนที่ "เกิน" ให้เทียบตัวเลขสองตัวก่อนเขียนทุกครั้ง
- **ใช้โครงนี้เสมอ ห้ามข้ามไปตอบแบบสั้น** ถ้าช่องไหนหาค่าใน "ข้อมูลสินค้า" ไม่เจอจริง ๆ ให้เขียนเฉพาะช่องนั้นว่า "ไม่มีข้อมูลในระบบ" แล้วเขียนช่องที่เหลือตามปกติ ห้ามทิ้งทั้งโครง
- **ห้ามเดาค่าที่ไม่มีในข้อมูลสินค้าเด็ดขาด** ห้ามคาดเดาสเปกจากชื่อรุ่น แบรนด์ หรือราคา"""


# ---- ดึงงบประมาณจากคำถาม (ใช้กรองราคาก่อนค้นหา) ----
# ปัญหา: vector search จัดอันดับด้วย "ความหมายใกล้เคียง" ไม่เข้าใจตัวเลข
# ถามงบ 20,000 แต่คำว่า "เล่นเกม" ไปตรงกับรุ่น 26,000 มันก็ดึงรุ่นนั้นมา
# วิธีแก้: อ่านงบจากคำถาม แล้วใช้ metadata filter ของ Chroma กรองราคาตั้งแต่ตอน query

# คำที่บ่งบอกว่าตัวเลขถัดไปคือ "งบ" (บังคับต้องมีคำพวกนี้ ไม่งั้น "RTX 5050" จะโดนอ่านเป็นราคา)
_BUDGET_CUE = r"(?:งบประมาณ|งบ|ไม่เกิน|ราคาไม่เกิน|ต่ำกว่า|ในราคา|ราคาประมาณ|budget)"
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
    _BUDGET_CUE + _GAP + rf"(?:{_NUM})\s*(?:-|–|ถึง)\s*(?P<hi>{_NUM})\s*(?P<unit>{_UNIT})?"
)
_RE_UNIT = re.compile(
    _BUDGET_CUE + _GAP + rf"(?:(?P<num>{_NUM})\s*)?(?P<unit>{_UNIT})\s*(?P<tail>{_THAI_ALT})?"
)
_RE_PLAIN = re.compile(_BUDGET_CUE + _GAP + rf"(?P<num>{_NUM})")

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
    candidates = []
    for prio, regex in enumerate((_RE_RANGE, _RE_UNIT, _RE_PLAIN)):
        for m in regex.finditer(text):
            candidates.append((m.start(), prio, _match_value(prio, m)))

    for _, _, value in sorted(candidates, key=lambda c: (c[0], c[1])):
        if value >= MIN_BUDGET:
            return int(value)
    return None


# คำถามแบบ "รุ่นถูก/แพงที่สุด" — vector search ตอบไม่ได้เพราะมันจัดอันดับด้วยความหมาย
# ไม่ได้เรียงตามราคา ต้องดึงเอกสารสเปกของรุ่นนั้นตรง ๆ จาก metadata แทน
_RE_CHEAPEST = re.compile(r"ถูกที่สุด|ถูกสุด|ราคาต่ำสุด|ประหยัดที่สุด|ราคาถูกที่สุด")
_RE_PRICIEST = re.compile(r"แพงที่สุด|แพงสุด|ราคาสูงสุด|ท็อปสุด")

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

    pairs = get_specs_by_price()
    if not pairs:
        return None
    if budget is not None:
        pairs = [p for p in pairs if p[0] <= budget] or pairs
    return pairs[0][1] if cheap else pairs[-1][1]


def _query_by_type(q_emb, n: int, doc_type: str, budget: int | None) -> list[str]:
    """ค้นเฉพาะเอกสารประเภทที่ต้องการ (spec / review) พร้อมกรองราคาถ้ามีงบ"""
    if n <= 0:
        return []
    conds = [{"type": {"$eq": doc_type}}]
    if budget is not None:
        # metadata "price" ถูกเก็บไว้ตอน build_index.py แล้ว (ทั้งเอกสารสเปกและรีวิว)
        conds.append({"price": {"$lte": budget}})
    where = conds[0] if len(conds) == 1 else {"$and": conds}
    res = get_collection().query(query_embeddings=q_emb, n_results=n, where=where)
    return res["documents"][0]


def retrieve(
    question: str, k_spec: int = TOP_K_SPEC, k_review: int = TOP_K_REVIEW,
    budget: int | None = None,
) -> tuple[list[str], bool]:
    """ค้นเอกสารที่เกี่ยวข้องที่สุด ถ้ามีงบจะกรองเฉพาะรุ่นที่ราคาไม่เกินงบ
    คืน (เอกสาร, over_budget) — over_budget=True แปลว่าไม่มีรุ่นไหนอยู่ในงบเลย

    สำคัญ: ต้องแยกโควตา spec/review แทนที่จะค้นรวมแล้วเอา top-k
    เพราะในฐานข้อมูลมีรีวิว 297 ต่อสเปก 99 (3:1) ถ้าค้นรวม รีวิวจะกวาด
    ทุกช่องไปหมดสำหรับคำถามเชิงความรู้สึก ("แบตอึดไหม", "เครื่องร้อนไหม")
    แล้ว LLM จะไม่เห็นราคา/สเปก/คะแนนเบนช์มาร์กเลย -> แต่งตัวเลขเอง
    """
    q_emb = get_embedder().encode([question], normalize_embeddings=True).tolist()

    specs = _query_by_type(q_emb, k_spec, "spec", budget)
    over_budget = budget is not None and not specs
    if over_budget:
        # ไม่มีรุ่นไหนในงบเลย -> ค้นใหม่แบบไม่กรอง เพื่อให้ AI บอกได้ว่ารุ่นถูกสุดคือเท่าไหร่
        specs = _query_by_type(q_emb, k_spec, "spec", None)
        reviews = _query_by_type(q_emb, k_review, "review", None)
    else:
        reviews = _query_by_type(q_emb, k_review, "review", budget)

    # ถามถึงรุ่นถูกสุด/แพงสุด -> ยัดเอกสารสเปกของรุ่นนั้นไว้หัวแถวเสมอ
    # ไม่งั้น AI จะรู้แค่ชื่อกับราคาจากเอกสารภาพรวมร้าน แล้วตอบได้ไม่ครบตามรูปแบบ
    extreme = _extreme_spec_doc(question, budget)
    if extreme and extreme not in specs:
        specs = [extreme] + specs[:-1]

    # แนบภาพรวมร้านทุกครั้ง — ตอบคำถามแบบ "มีกี่รุ่น / ถูกสุดเท่าไหร่" ได้โดยไม่กินโควตาค้นหา
    summary = get_summary_doc()
    return specs + reviews + ([summary] if summary else []), over_budget


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


def generate(
    question: str,
    context: list[str],
    history: list,
    budget: int | None = None,
    over_budget: bool = False,
) -> str:
    # เรียก ThaiLLM ด้วย requests ตรง ๆ (gateway บล็อก header ที่ openai library แนบมา)
    # โมเดล 8B รับ system role ได้ไม่ดีเท่ารุ่นใหญ่ เลยยุบ system + ประวัติ + context + คำถาม ไว้ใน user message เดียว
    context_text = "\n".join(f"- {c}" for c in context)

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
        budget_note = (
            f"\n**งบของลูกค้า: ไม่เกิน {budget:,} บาท — ไม่มีรุ่นไหนในร้านที่ราคาไม่เกินงบนี้เลย**\n"
            f"ให้บอกลูกค้าตรง ๆ ว่าไม่มีรุ่นในงบ แล้วเสนอรุ่นที่ราคาใกล้เคียงที่สุดจากข้อมูลด้านล่าง "
            f"พร้อมบอกชัดว่าราคาเกินงบไปเท่าไหร่\n"
        )
    elif budget is not None:
        budget_note = (
            f"\n**งบของลูกค้า: ไม่เกิน {budget:,} บาท — ข้อมูลสินค้าด้านล่างคัดมาเฉพาะรุ่นที่ราคาไม่เกินงบแล้ว**\n"
            f"ห้ามแนะนำรุ่นที่ราคาเกิน {budget:,} บาท และห้ามเอ่ยถึงรุ่นที่ไม่ได้อยู่ในข้อมูลด้านล่าง\n"
            f"ทุกรุ่นด้านล่างราคาไม่เกินงบแล้ว ดังนั้นห้ามเขียนว่า 'เกินงบ' เด็ดขาด\n"
        )
    if budget is not None:
        budget_note += budget_diff_table(context, budget)
    else:
        # ไม่มีงบ -> ต้องสั่งให้ชัด ไม่งั้นโมเดลจะแถว่า "อยู่ในงบ เหลืออีก 0 บาท" ขึ้นมาเอง
        budget_note = (
            "\n**ลูกค้ายังไม่ได้บอกงบ** — ห้ามเอ่ยถึงงบ เงินที่เหลือ หรือส่วนต่างราคาใด ๆ "
            "ในคำตอบเด็ดขาด ช่อง [หมายเหตุราคา] ให้เว้นว่างไว้ เขียนแค่ราคาเฉย ๆ\n"
        )

    suffix = " /no_think" if DISABLE_THINKING else ""
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

    user_msg = f"""{SYSTEM_PROMPT}

ประวัติบทสนทนา:
{history_text if history_text else "(เริ่มต้นใหม่)"}
{budget_note}
ข้อมูลสินค้า:
{context_text}

คำถามลูกค้า: {question}{extreme_note}{suffix}"""

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {THAILLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": user_msg}],
            # 0.1 แทน 0.2 — ยิ่งต่ำยิ่งตอบนิ่ง รูปแบบคำตอบคงเส้นคงวาขึ้น
            "temperature": 0.1,
            # ปิดโหมดคิด -> 512 พอ / เปิดโหมดคิด -> เผื่อ token ให้ส่วนคิดเยอะหน่อย
            "max_tokens": 512 if DISABLE_THINKING else 1536,
        },
        timeout=120,
    )
    resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"]
    answer = strip_think(raw)
    if not answer:
        return "(โมเดลคิดยาวจน token หมดก่อนตอบ ลองถามใหม่อีกครั้งนะ)"
    return answer


@app.route("/")
def index():
    # หน้าหลัก — ไม่ต้องเตรียม session ที่นี่ เพราะทุกจุดที่อ่านประวัติใช้ .get() ที่มีค่า default
    # อยู่แล้ว และหน้าเว็บจะยิง /new_chat ตอนโหลดเพื่อล้าง session ให้ตรงกับจอว่างอยู่แล้ว
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    # endpoint หลัก: รับข้อความ -> ค้นหา (retrieve) -> แต่งคำตอบ (generate)
    # silent=True: body ที่ไม่ใช่ JSON ต้องได้ error เป็น JSON ไม่ใช่หน้า HTML 400
    # (ไม่งั้น fetch ฝั่งหน้าเว็บจะ parse ไม่ได้แล้วขึ้นว่า "เชื่อมต่อเซิร์ฟเวอร์ไม่ได้" ซึ่งทำให้เข้าใจผิด)
    data = request.get_json(silent=True)
    if not isinstance(data, dict):
        return jsonify({"error": "ส่งข้อมูลมาไม่ถูกรูปแบบ (ต้องเป็น JSON)"}), 400

    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "empty message"}), 400
    if len(user_msg) > MAX_MSG_LEN:
        # กันข้อความยาวผิดปกติมาถ่วงเธรด (gunicorn มีแค่ 1 worker x 4 threads)
        return jsonify({"error": f"ข้อความยาวเกินไป (สูงสุด {MAX_MSG_LEN:,} ตัวอักษร)"}), 413

    history = session.get("history", [])

    # หางบจากข้อความล่าสุดก่อน ถ้าไม่มีค่อยใช้งบที่เคยจำไว้ใน session
    # (เช่น รอบก่อนบอก "งบ 20000" รอบนี้ถามแค่ "มีรุ่นอื่นอีกไหม" ก็ยังต้องอยู่ในงบเดิม)
    # เก็บงบแยกจาก history เพราะ history ถูกตัดให้สั้นเพื่อคุมขนาด cookie
    budget = extract_budget(user_msg)
    if budget is None:
        budget = session.get("budget")
    else:
        session["budget"] = budget

    try:
        context, over_budget = retrieve(user_msg, budget=budget)
    except Exception as e:
        return jsonify({"error": f"Vector search error: {e}"}), 500

    try:
        answer = generate(user_msg, context, history, budget=budget, over_budget=over_budget)
    except requests.HTTPError as e:
        return jsonify({"error": f"ThaiLLM error: {e} — เช็ค API key หรือ endpoint"}), 502
    except Exception as e:
        return jsonify({"error": f"LLM error: {e}"}), 500

    # บันทึกประวัติลง session — Flask เก็บ session ไว้ใน cookie ซึ่งเบราว์เซอร์ตัดทิ้งถ้าเกิน
    # 4096 bytes ดังนั้นต้องจำกัดทั้งจำนวนข้อความและความยาวต่อข้อความ ไม่งั้นประวัติจะหายเงียบ ๆ
    history.append({"role": "user", "content": user_msg[:MAX_STORED_CHARS]})
    history.append({"role": "assistant", "content": answer[:MAX_STORED_CHARS]})
    session["history"] = history[-MAX_HISTORY_MSGS:]

    return jsonify({"message": answer})


@app.route("/new_chat", methods=["POST"])
def new_chat():
    # ล้างประวัติบทสนทนาใน session เพื่อเริ่มสนทนาใหม่ (ล้างงบที่จำไว้ด้วย)
    session["history"] = []
    session.pop("budget", None)
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
    for m in reversed(clean):
        if m["role"] == "user":
            found = extract_budget(m["content"])
            if found is not None:
                session["budget"] = found
                break

    return jsonify({"ok": True, "restored": len(clean)})


if __name__ == "__main__":
    if not CHROMA_DIR.exists():
        print("ERROR: ไม่พบ chroma_db — รัน `python build_index.py` ก่อนนะ")
    else:
        # debug=True เปิด Werkzeug debugger ซึ่งรันโค้ดจากเบราว์เซอร์ได้ — ห้ามติดไปถึง production
        app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", port=5000)
