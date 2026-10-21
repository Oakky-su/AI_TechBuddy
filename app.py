"""Tech Buddy — RAG chatbot แนะนำโน้ตบุ๊ค (Flask) ไฟล์เดียวทั้งระบบ เรียงโค้ดตามลำดับการทำงานจากบนลงล่าง

ตอนแอปเริ่ม
  1. ข้อมูลร้าน: อ่าน laptops.json -> 1 รุ่น = 1 เอกสาร (สเปก + คะแนน + ระดับ + รีวิว) พร้อมค่าตัวเลขให้กรองได้
  2. embedding: โหลด bge-m3 + embed เอกสาร 99 รุ่นไว้ล่วงหน้าใน thread พื้นหลัง

ทุกคำถาม (เรียก LLM 2 ครั้ง — เป็น RAG ไม่ใช่ agent: ไม่มีการวนเรียกเครื่องมือ)
  3. รับคำถาม: ตรวจรูปแบบ ความยาว AI ที่เลือก และ rate limit
  4. (ตัวช่วยเรียก AI แบบ OpenAI-compatible ที่ขั้น 5 และ 7 ใช้)
  5. ตีความ: LLM แปลงคำถาม + ประวัติแชท + เงื่อนไขเดิม เป็น JSON (intent, filters, sort, limit, usage)
     แล้วโค้ดตรวจ/เติมด้วยตัวอ่านตัวเลข (งบ แรม การ์ดจอ จอ SSD Windows) ที่ทดสอบแล้ว
  6. คัดรุ่น: กรอง/เรียงจากข้อมูลจริงทั้ง 99 รุ่นด้วยโค้ด แล้วใช้ embedding จัดลำดับตามการใช้งาน
  7. ตอบ: ส่งรายการที่คัดแล้วให้ LLM เขียนตามรูปแบบของ intent (ทยอยสตรีมไปหน้าเว็บ)
     คำถามนอกเรื่อง/ทักทาย/ความรู้ทั่วไป/ยี่ห้อที่ไม่มี ใช้ข้อความสำเร็จรูป ไม่เรียก LLM รอบสอง
  8. จำบทสนทนา: ประวัติแชท + เงื่อนไขล่าสุดเก็บใน session (ใช้ตีความคำถามถัดไป)
  9. endpoint ของหน้าเว็บ (/chat/stream, /chat, /chat/commit, ...)

LLM เลือกได้ใน .env / providers.json (ดู llm_providers.py)
"""
import json
import os
import re
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import requests
from dotenv import load_dotenv
from flask import Flask, Response, render_template, request, jsonify, session
from itsdangerous import BadSignature, URLSafeTimedSerializer
from sentence_transformers import SentenceTransformer  # โมเดล embedding ภาษาไทย/หลายภาษา

import llm_providers as llm

load_dotenv()  # โหลด environment variables จากไฟล์ .env

app = Flask(__name__)
# secret_key คงที่ถ้าตั้งใน env (session ไม่หลุดตอนรีสตาร์ต) ไม่งั้น random ชั่วคราว
app.secret_key = os.getenv("SECRET_KEY") or os.urandom(24).hex()

BASE_DIR = Path(__file__).parent
LAPTOPS_PATH = BASE_DIR / "laptops.json"
EMBED_MODEL = "BAAI/bge-m3"
# แผง "ดูกระบวนการคิด" ใต้คำตอบบนหน้าเว็บ: ตีความคำถามยังไง คัดรุ่นอะไรมา คะแนนเท่าไหร่
# ส่ง prompt อะไรให้ AI และ AI ใช้เวลา/โทเคนเท่าไหร่ — ไว้ทดสอบกับลูกค้า
# เปิดแล้วทุกคนที่เข้าเว็บกดดูได้ รวม prompt ทั้งก้อน -> ปิดก่อนเปิดให้คนทั่วไปใช้จริง
DEBUG_PANEL = os.getenv("DEBUG_PANEL", "false").lower() == "true"
RATE_LIMIT_PER_MIN = int(os.getenv("RATE_LIMIT_PER_MIN", "10"))  # ข้อความต่อนาทีต่อ IP
MAX_MSG_LEN = 1_000          # ความยาวข้อความสูงสุดที่รับ
MAX_HISTORY_MSGS = 8         # จำนวนข้อความที่เก็บใน session (4 รอบสนทนา)
MAX_STORED_CHARS = 800       # ตัดคำตอบที่เก็บลง session ไม่ให้ cookie บวม
HISTORY_TURNS = 4            # ข้อความย้อนหลังที่ส่งให้ตัวตีความ/ตัวตอบ


# ═════════════════════════ 1. ข้อมูลร้าน: 1 รุ่น = 1 เอกสาร + ค่าตัวเลขให้กรองได้ ═════════════════════════
_laptops: list[dict] | None = None
_catalog: list[dict] | None = None

# ระดับคะแนนเบนช์มาร์ก — คำนวณด้วยโค้ดแล้วแนบไปกับเอกสาร ไม่ปล่อยให้ LLM เทียบตัวเลขกับช่วงเอง
# (ทดสอบกับ Gemma: GPU 10,500 ถูกเขียนว่า "เริ่มต้น-กลาง" ทั้งที่อยู่ช่วง 8,000-14,000 = "กลาง-สูง")
CPU_LEVELS = ((18_000, "เริ่มต้น"), (25_000, "กลาง"), (32_000, "กลาง-สูง"))
GPU_LEVELS = ((4_000, "เริ่มต้น"), (8_000, "เริ่มต้น-กลาง"), (14_000, "กลาง-สูง"))


def get_laptops() -> list[dict]:
    global _laptops
    if _laptops is None:
        _laptops = json.loads(LAPTOPS_PATH.read_text(encoding="utf-8"))
    return _laptops


def score_level(score: int, levels: tuple) -> str:
    for upper, name in levels:
        if score < upper:
            return name
    # ขอบบนของช่วงสุดท้ายนับเป็นช่วงนั้น เช่น GPU 14,000 พอดี = กลาง-สูง (เกณฑ์เขียน "8,000–14,000")
    return levels[-1][1] if score == levels[-1][0] else "สูง"


def _ram_gb(ram: str) -> int:
    m = re.search(r"(\d+)\s*GB", ram or "", re.IGNORECASE)
    return int(m.group(1)) if m else 0


def _series(name: str) -> str:
    """"HP Victus 16-r0356TX" -> "HP Victus 16" / "Lenovo LOQ 15IRX9-83DV01A0TA" -> "Lenovo LOQ" """
    out = []
    for tok in name.split():
        if "-" in tok:
            head = tok.split("-")[0]
            if head.isdigit() and len(head) <= 3:
                out.append(head)
            break
        out.append(tok)
    return " ".join(out)


def _gpu_family(gpu: str) -> str:
    g = gpu.lower()
    if "rtx" in g:
        return "rtx"
    if "gtx" in g:
        return "gtx"
    if re.search(r"\brx\s*\d", g):
        return "radeon rx"
    if "arc" in g and re.search(r"arc\s*[a-z]?\d", g):
        return "arc"
    return "integrated"


def _gb(num: str, unit: str) -> int:
    """ความจุเป็น GB: "512", "GB" -> 512 / "1", "TB" -> 1000 (ใช้หน่วยเดียวกันทั้งข้อมูลและคำถาม)"""
    return round(float(num) * (1000 if unit.lower() == "tb" else 1))


def catalog() -> list[dict]:
    global _catalog
    if _catalog is None:
        out = []
        for lp in get_laptops():
            spec, bench = lp["spec"], lp["benchmark"]
            cs, gs = bench["CPU Score"], bench["GPU Score"]
            screen = re.match(r"([\d.]+)\s*inch", spec["DISPLAY"])
            res = re.search(r"\((\d{3,4})\s*x\s*(\d{3,4})\)", spec["DISPLAY"])   # "16.0 inch (2560x1600) QHD+"
            gen = re.search(r"gen\s*(\d)(?!\d)", spec["STORAGE"], re.I)            # "512 GB SSD PCIe M.2 Gen 4"
            size = re.match(r"\s*(\d+(?:\.\d+)?)\s*(tb|gb)", spec["STORAGE"], re.I)
            rec = {
                "name": lp["ชื่อสินค้า"], "brand": lp["ชื่อสินค้า"].split()[0], "series": _series(lp["ชื่อสินค้า"]),
                "price": lp["ราคา"], "cpu": spec["CPU"], "gpu": spec["GPU"], "gpu_family": _gpu_family(spec["GPU"]),
                "ram": spec["RAM"], "ram_gb": _ram_gb(spec["RAM"]), "display": spec["DISPLAY"],
                "screen": float(screen.group(1)) if screen else None, "storage": spec["STORAGE"], "os": spec["OS"],
                "res_w": int(res.group(1)) if res else None, "res_h": int(res.group(2)) if res else None,
                "ssd_gen": int(gen.group(1)) if gen else None,
                "storage_gb": _gb(size.group(1), size.group(2)) if size else None,
                "cpu_score": cs, "gpu_score": gs,
                "cpu_level": score_level(cs, CPU_LEVELS), "gpu_level": score_level(gs, GPU_LEVELS),
                "reviews": lp.get("reviews", []),
            }
            rec["doc"] = (
                f"{rec['name']} ราคา {rec['price']:,} บาท | CPU {rec['cpu']} คะแนน {cs:,} (ระดับ{rec['cpu_level']}) | "
                f"GPU {rec['gpu']} คะแนน {gs:,} (ระดับ{rec['gpu_level']}) | แรม {rec['ram']} | จอ {rec['display']} | "
                f"ความจุ {rec['storage']} | {rec['os']} | รีวิวผู้ใช้: " + " / ".join(rec["reviews"])
            )
            out.append(rec)
        _catalog = out
    return _catalog


def store_stats() -> str:
    """ภาพรวมร้าน (ตอบ "มีกี่รุ่น / ราคาเริ่มต้น / มียี่ห้ออะไร") — คำนวณจากข้อมูลจริง"""
    cat = catalog()
    cheap, top = min(cat, key=lambda x: x["price"]), max(cat, key=lambda x: x["price"])
    brands = {}
    for lp in cat:
        brands.setdefault(lp["brand"], []).append(lp["price"])
    rams = {}
    for lp in cat:
        rams[lp["ram_gb"]] = rams.get(lp["ram_gb"], 0) + 1
    fams = {}
    for lp in cat:
        fams[lp["gpu_family"]] = fams.get(lp["gpu_family"], 0) + 1
    return (
        f"ทั้งหมด {len(cat)} รุ่น ราคา {cheap['price']:,}–{top['price']:,} บาท | "
        f"ถูกที่สุด {cheap['name']} ({cheap['price']:,} บาท) | แพงที่สุด {top['name']} ({top['price']:,} บาท) | "
        "ยี่ห้อ: " + ", ".join(f"{b} {len(p)} รุ่น ({min(p):,}–{max(p):,} บาท)"
                               for b, p in sorted(brands.items(), key=lambda kv: -len(kv[1]))) + " | "
        "แรม: " + ", ".join(f"{g}GB {n} รุ่น" for g, n in sorted(rams.items())) + " | "
        "การ์ดจอ: " + ", ".join(f"{f} {n} รุ่น" for f, n in sorted(fams.items(), key=lambda kv: -kv[1]))
    )


# ═════════════════════════ 2. embedding: โหลดโมเดล + เวกเตอร์ 99 รุ่นไว้ล่วงหน้า ═════════════════════════
# cache โมเดลและเวกเตอร์ไว้ใน memory ป้องกันการโหลดซ้ำทุก request
_embedder: SentenceTransformer = None
_embedder_lock = threading.Lock()
_doc_vecs: np.ndarray | None = None
_doc_vecs_lock = threading.Lock()   # warmup ตอนแอปเริ่มกับคำถามที่เข้ามาระหว่างนั้นใช้ผลเดียวกัน ไม่ embed ซ้ำ

# คำถามแรกหลังรีสตาร์ตที่ต้องเรียงตามการใช้งาน (เช่น "เล่นเกม งบ 30000") ต้องโหลด bge-m3 (~2GB)
# แล้ว embed เอกสารทั้ง 99 รุ่นก่อน บนเว็บจริงเคยรวมกันเกิน 100 วินาที -> Cloudflare ตัดเป็น 524
# จึงเริ่มโหลดใน thread พื้นหลังทันทีที่แอปขึ้น เซิร์ฟเวอร์เปิดรับคำถามได้เลยไม่ต้องรอ
# false = โหลดตอนมีคำถามแรกแบบเดิม (เช่น เครื่องแรมน้อยที่ไม่อยากโหลดโมเดลจนกว่าจะมีคนใช้)
WARMUP_EMBEDDINGS = os.getenv("WARMUP_EMBEDDINGS", "true").lower() == "true"


def get_embedder():
    # โหลด bge-m3 ครั้งแรกแล้ว cache ไว้ (โหลดช้า + กินแรม ~2GB)
    # lock: คำถามที่เข้ามาระหว่าง warmup กำลังโหลด ต้องรอตัวเดียวกัน ไม่โหลดซ้ำอีกก้อน
    global _embedder
    if _embedder is None:
        with _embedder_lock:
            if _embedder is None:
                _embedder = SentenceTransformer(EMBED_MODEL)
    return _embedder


def doc_vectors() -> np.ndarray:
    """เวกเตอร์ของเอกสารรุ่นละ 1 ชิ้น (99 ชิ้น) — embed ครั้งแรกที่ใช้ (หรือตอนแอปเริ่ม ดู WARMUP_EMBEDDINGS)
    แล้วเก็บไว้ในหน่วยความจำ"""
    global _doc_vecs
    with _doc_vecs_lock:
        if _doc_vecs is None:
            docs = [lp["doc"] for lp in catalog()]
            _doc_vecs = np.asarray(get_embedder().encode(docs, normalize_embeddings=True, batch_size=16))
    return _doc_vecs


def warmup_embeddings() -> bool:
    """โหลดโมเดล embedding + เวกเตอร์ 99 รุ่นไว้ก่อนมีคำถาม คืน True ถ้าสำเร็จ
    พังก็แค่ log แล้วจบ แอปทำงานต่อตามปกติ แค่คำถามแรกจะกลับไปโหลดเองแบบเดิม"""
    started = time.perf_counter()
    print("[warmup] กำลังโหลดโมเดล embedding ล่วงหน้า — ระหว่างนี้แอปรับคำถามได้ตามปกติ", flush=True)
    try:
        doc_vectors()
    except Exception as e:
        print(f"[warmup] โหลดล่วงหน้าไม่สำเร็จ ({type(e).__name__}: {e}) "
              "— แอปทำงานต่อ จะโหลดตอนมีคำถามแรกแทน", flush=True)
        return False
    print(f"[warmup] โมเดล embedding พร้อมแล้ว ใช้เวลา {time.perf_counter() - started:.1f} วินาที", flush=True)
    return True


def start_warmup() -> threading.Thread | None:
    """เริ่ม warmup_embeddings ใน thread พื้นหลัง (daemon = ไม่ค้างตอนปิดแอป) — ปิดไว้ใน .env คืน None"""
    if not WARMUP_EMBEDDINGS:
        return None
    thread = threading.Thread(target=warmup_embeddings, name="warmup-embeddings", daemon=True)
    thread.start()
    return thread


# ═════════════════════════ 3. รับคำถาม: ตรวจ request + จำกัดจำนวนคำขอต่อ IP ═════════════════════════
# ทุก /chat = 2 คำขอไปยัง LLM ซึ่งมีโควตาจำกัด
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


def _chat_request():
    """ตรวจ request ของ /chat และ /chat/stream (รูปแบบ, ความยาว, AI ที่เลือก, rate limit)
    คืน (ข้อมูลที่พร้อมใช้, None) หรือ (None, response error)"""
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

    # AI ที่ผู้ใช้เลือกจากหน้าเว็บ (ไม่ส่งมา = ใช้ค่าเริ่มต้นใน .env)
    provider, pick_error = llm.pick(data.get("provider"))
    if pick_error:
        return None, (jsonify({"error": pick_error}), 400)

    if too_many_requests(client_ip()):
        return None, (jsonify({
            "error": f"ถามถี่เกินไปครับ (จำกัด {RATE_LIMIT_PER_MIN} ข้อความต่อนาที) "
                     f"รอสักครู่แล้วลองใหม่นะครับ"
        }), 429)

    return {"message": user_msg, "provider": provider, "history": session.get("history", []),
            # ขอดูกระบวนการคิดได้เฉพาะตอนผู้ดูแลเปิด DEBUG_PANEL ไว้ (ปิดอยู่ = ไม่สนค่าที่ส่งมา)
            "debug": DEBUG_PANEL and data.get("debug") is True}, None


def new_trace(job: dict) -> dict | None:
    """ที่เก็บข้อมูล debug ของคำถามนี้ — ขั้น 5-7 เขียนต่อเอง (None = ไม่ได้ขอ debug)"""
    return {} if job["debug"] else None


def chat_meta(job: dict, started: float) -> dict:
    # หน้าเว็บแสดงบรรทัดเล็ก ๆ ใต้คำตอบว่า AI ตัวไหนตอบ ใช้เวลาเท่าไหร่
    provider = job["provider"]
    return {
        "provider": provider["id"],
        "provider_label": provider.get("short") or provider["label"],
        "elapsed": round(time.time() - started, 1),
    }


# ═════════════════════════ 4. เรียก AI (endpoint แบบ OpenAI — ใช้ทั้งขั้นตีความและขั้นตอบ) ═════════════════════════
class LLMError(Exception):
    """เรียก LLM ไม่สำเร็จ — ข้อความใช้แสดงให้ผู้ใช้เห็นได้เลย"""


def _llm_headers(provider: dict) -> dict:
    headers = {"Content-Type": "application/json"}
    if provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"
    return headers


def complete(provider: dict, messages: list, max_tokens: int, temperature: float = 0.1,
             json_mode: bool = False) -> tuple[str, dict, float]:
    """เรียก endpoint แบบ OpenAI ของ provider — คืน (ข้อความ, response JSON, วินาที)
    เรียกด้วย requests ตรง ๆ (gateway ของ ThaiLLM บล็อก header ที่ openai library แนบมา)"""
    body = {"model": provider["model"], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = _llm_headers(provider)
    started = time.perf_counter()
    try:
        resp = requests.post(f"{provider['base_url']}/chat/completions", headers=headers, json=body, timeout=120)
        if json_mode and resp.status_code == 400:   # endpoint บางตัวไม่รองรับ response_format
            body.pop("response_format")
            resp = requests.post(f"{provider['base_url']}/chat/completions", headers=headers, json=body, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise LLMError(f"{provider['label']} error: {e} — เช็ค API key หรือ endpoint") from e
    data = resp.json()
    return data["choices"][0]["message"].get("content") or "", data, time.perf_counter() - started


def complete_stream(provider: dict, messages: list, max_tokens: int, temperature: float = 0.1,
                    meta: dict | None = None):
    """เหมือน complete() แต่ทยอยคืนข้อความทีละท่อนระหว่างที่ AI พิมพ์ (stream)
    meta เก็บ usage / finish_reason / ความคิด / วินาที ไว้ให้โหมด debug หลังจบ"""
    body = {"model": provider["model"], "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    headers = _llm_headers(provider)
    url = f"{provider['base_url']}/chat/completions"
    started = time.perf_counter()
    try:
        resp = requests.post(url, headers=headers, json=body, timeout=120, stream=True)
        if resp.status_code == 400:   # endpoint บางตัวไม่รองรับ stream_options
            resp.close()
            body.pop("stream_options")
            resp = requests.post(url, headers=headers, json=body, timeout=120, stream=True)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise LLMError(f"{provider['label']} error: {e} — เช็ค API key หรือ endpoint") from e
    usage, finish, thinking = None, None, []
    with resp:   # ลูกค้ากดหยุด -> generator ถูกปิด -> ตัดสายกับ AI ทันที
        try:
            for line in resp.iter_lines():
                line = line.decode("utf-8", "replace")
                if not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if payload == "[DONE]":
                    break
                chunk = json.loads(payload)
                usage = chunk.get("usage") or usage
                for ch in chunk.get("choices") or []:
                    finish = ch.get("finish_reason") or finish
                    delta = ch.get("delta") or {}
                    thinking += [delta[k] for k in ("reasoning", "reasoning_content") if delta.get(k)]
                    if delta.get("content"):
                        yield delta["content"]
        except requests.RequestException as e:
            raise LLMError(f"{provider['label']} error: {e}") from e
    if meta is not None:
        meta.update({"seconds": time.perf_counter() - started, "usage": usage,
                     "finish_reason": finish, "thinking": "".join(thinking)})


# ═════════════════════════ 5. ตีความคำถามเป็นเงื่อนไข (เรียก LLM ครั้งที่ 1) ═════════════════════════
INTENTS = ("recommend", "list", "rank", "lookup", "compare", "overview",
           "not_in_store", "general", "greeting", "offtopic")
TEMPLATED = ("not_in_store", "general", "greeting", "offtopic")
SORT_FIELDS = {"price": "price", "cpu_score": "cpu_score", "gpu_score": "gpu_score", "ram": "ram_gb"}
RAM_SIZES = {4, 8, 12, 16, 24, 32, 48, 64, 96, 128}
LIST_CAP = 40          # ไล่รายชื่อได้สูงสุด (ทั้งร้านมี 99 รุ่น)
NAMED_CAP = 8

UNDERSTAND_RULES = """คุณแปลงข้อความลูกค้าร้านโน้ตบุ๊คให้เป็นเงื่อนไขค้นหา ตอบเป็น JSON อย่างเดียว ห้ามมีข้อความอื่น
รูปแบบ: {"intent": "...", "models": [], "filters": {"price_max": null, "price_min": null, "price_target": null,
"ram_gb": null, "ram_min": null, "gpu": null, "gpu_family": null, "cpu": null, "brand": null, "series": null,
"screen": null}, "sort": null, "limit": null, "usage": null, "detail": false}

intent (เลือก 1):
- recommend = ขอคำแนะนำตามการใช้งานหรืองบ เช่น "เล่นเกม งบ 25000", "แบตอึด พกพา", "ขอ 3 รุ่น ตัดต่อ", "อยากได้ราคาประมาณ 24000"
- list = ขอรายชื่อรุ่นที่ตรงเงื่อนไข เช่น "แรม 24 มีตัวไหนบ้าง", "LOQ มีกี่รุ่น", "ขอสเปคทุกตัว", "เรียงราคาให้หน่อย"
- rank = ถามรุ่นที่ดีที่สุด/ถูกที่สุด/แพงที่สุดตามเกณฑ์ เช่น "CPU แรงสุด", "การ์ดจอแรงสุดในงบ", "รุ่นไหนถูกที่สุด", "RTX ที่ถูกที่สุด"
- lookup = ถามสเปก ราคา หรือคะแนนของรุ่นที่ระบุ เช่น "HP AI 15 สเปคเป็นยังไง", "ตัวที่ถูกกว่าใช้ CPU อะไร"
- compare = เทียบรุ่นที่ระบุตั้งแต่ 2 รุ่นขึ้นไป หรือถามส่วนต่างราคาของรุ่นที่ระบุ
- overview = ถามภาพรวมร้าน เช่น มีกี่รุ่น ราคาเริ่มต้น/สูงสุดในร้าน มียี่ห้ออะไรบ้าง
- not_in_store = ถามถึงยี่ห้อหรือรุ่นที่ไม่มีในร้าน เช่น MacBook, Dell
- general = ถามความรู้ทั่วไปที่ไม่ได้หาสินค้า เช่น "TDP คืออะไร", "ควรรอรุ่นใหม่ไหม"
- greeting = ทักทายหรือขอบคุณ
- offtopic = เรื่องที่ไม่เกี่ยวกับโน้ตบุ๊ค เช่น อากาศ อาหาร ข่าว

filters (ใส่เฉพาะที่ลูกค้าบอก ไม่บอกให้เป็น null):
- price_max = "งบ X" หรือ "ไม่เกิน X" / price_min = ขอบล่าง ("ระหว่าง A ถึง B" = price_min A, price_max B)
- price_target = "ราคาประมาณ X" หรือ "ราคา X" ที่ไม่ได้บอกว่าไม่เกิน
- ram_gb = แรมเท่ากับ X พอดี ("แรม 24") / ram_min = "แรม X ขึ้นไป" หรือ "อย่างน้อย"
- gpu = ชื่อการ์ดจอเจาะจง เช่น "RTX 4060" / gpu_family = "rtx" | "gtx" | "integrated" เมื่อพูดกว้าง ๆ
- cpu = คำในชื่อซีพียู เช่น "Ryzen 7", "Core i5", "Core Ultra 7"
- brand = ยี่ห้อในร้าน / series = ชื่อซีรีส์ เช่น "Lenovo LOQ", "HP Victus 16"
- screen = ขนาดจอเป็นนิ้ว เช่น 16
sort = {"by": "price" | "cpu_score" | "gpu_score", "order": "asc" | "desc"} เช่น "ถูกที่สุด" = price asc,
  "แพงไปถูก" = price desc, "CPU แรงสุด" = cpu_score desc, "การ์ดจอแรงสุด" = gpu_score desc
limit = จำนวนรุ่นที่ลูกค้าขอ ("ขอ 3 รุ่น" = 3, "ตัวไหนถูกสุด"/"รุ่นไหนแรงสุด" = 1)
models = ชื่อรุ่นหรือซีรีส์ที่ลูกค้าเอ่ยถึง ถ้าลูกค้าอ้างถึงรุ่นจากคำตอบก่อนหน้า ("ตัวนั้น", "ตัวที่ถูกกว่า") ให้ใส่ชื่อรุ่นเหล่านั้น
usage = การใช้งานที่ลูกค้าพูดเอง เช่น "เล่นเกม", "แบตอึด พกพา", "ตัดต่อวิดีโอ" (ไม่พูด = null ห้ามเดาจากงบหรือสเปก)
detail = true ถ้าลูกค้าขอสเปกละเอียด

ถามต่อ: ถ้าข้อความใหม่ต่อจากเรื่องเดิม (เช่น "แล้วราคา 24000 ล่ะ", "อันไหนถูกสุด", "มีรุ่นอื่นอีกไหม", "ตัวที่ถูกกว่า...")
ให้คงเงื่อนไขเดิมที่ยังเกี่ยวข้อง แล้วเพิ่มหรือแก้ตามข้อความใหม่ ถ้าเป็นเรื่องใหม่ไม่ต้องใช้เงื่อนไขเดิม

ตัวอย่าง:
"ขอรุ่นที่ใช้ Core i7 งบไม่เกิน 35k" -> {"intent":"list","filters":{"cpu":"Core i7","price_max":35000},"models":[],"sort":null,"limit":null,"usage":null,"detail":false}
"แรม 24 มีตัวไหนบ้าง" -> {"intent":"list","filters":{"ram_gb":24},"models":[],"sort":null,"limit":null,"usage":null,"detail":false}
(เงื่อนไขเดิม ram_gb 24) "แล้วราคา 24000 บาทหละ" -> {"intent":"list","filters":{"ram_gb":24,"price_target":24000},"models":[],"sort":null,"limit":null,"usage":null,"detail":false}
"งบ 30,000 รุ่นไหน CPU แรงที่สุด" -> {"intent":"rank","filters":{"price_max":30000},"sort":{"by":"cpu_score","order":"desc"},"limit":1,"models":[],"usage":null,"detail":false}
(คุยเรื่อง Acer Nitro V 15 กับ HP Victus 16) "ตัวที่ถูกกว่าใช้ CPU อะไร" -> {"intent":"lookup","models":["Acer Nitro V 15 ANV15-51-56QE","HP Victus 16-r0356TX"],"sort":{"by":"price","order":"asc"},"limit":1,"filters":{},"usage":null,"detail":false}
"""


def _series_list() -> str:
    seen = []
    for lp in catalog():
        s = " ".join(lp["series"].split()[:3])
        if s not in seen:
            seen.append(s)
    return ", ".join(seen)


def _history_text(history: list) -> str:
    recent = history[-HISTORY_TURNS:]
    return "\n".join(f"{'ลูกค้า' if h['role'] == 'user' else 'ผู้ช่วย'}: {h['content'][:600]}" for h in recent) or "(ไม่มี)"


# ---- 5.1 อ่าน JSON จาก LLM แล้วจัดรูป ----
def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("ไม่มี JSON")
    return json.loads(text[start:end + 1])


def _int(v) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"\d[\d,]*", str(v))
    return int(m.group(0).replace(",", "")) if m else None


def _screen(v) -> float | None:
    """ขนาดจอ: รับทั้ง 16, "16", "16 นิ้ว" — นอกช่วง 11-19 นิ้วทิ้ง"""
    if v is None or isinstance(v, bool):
        return None
    m = re.search(r"\d+(?:\.\d+)?", str(v))
    size = float(m.group(0)) if m else None
    return size if size and 11 <= size <= 19 else None


def normalize(raw: dict) -> dict:
    """ตรวจและจัดรูป JSON จาก LLM — ค่าที่ผิดรูปแบบทิ้งเป็น null แทนที่จะพัง"""
    f = raw.get("filters") or {}
    plan = {
        "intent": raw.get("intent") if raw.get("intent") in INTENTS else "recommend",
        "models": [str(m).strip() for m in (raw.get("models") or []) if str(m).strip()][:NAMED_CAP],
        "filters": {
            "price_max": _int(f.get("price_max")), "price_min": _int(f.get("price_min")),
            "price_target": _int(f.get("price_target")),
            "ram_gb": _int(f.get("ram_gb")), "ram_min": _int(f.get("ram_min")),
            "gpu": (str(f["gpu"]).strip() or None) if f.get("gpu") else None,
            "gpu_family": str(f["gpu_family"]).lower().strip() if f.get("gpu_family") else None,
            "cpu": (str(f["cpu"]).strip() or None) if f.get("cpu") else None,
            "brand": (str(f["brand"]).strip() or None) if f.get("brand") else None,
            "series": (str(f["series"]).strip() or None) if f.get("series") else None,
            "screen": _screen(f.get("screen")),
            # ความละเอียดจอ / SSD / Windows: โค้ดอ่านจากคำถามเอง (spec_extras) ไม่รับจาก LLM
            "res_w": None, "res_h": None, "res_min": None, "res_gt": None,
            "ssd_gen": None, "storage_gb": None, "storage_min": None, "os": None,
        },
        "sort": None,
        "limit": _int(raw.get("limit")),
        "usage": (str(raw["usage"]).strip() or None) if raw.get("usage") else None,
        "detail": bool(raw.get("detail")),
        "unsupported": [],   # โค้ดเป็นคนใส่ (unsupported_terms) ไม่รับจาก LLM
    }
    fl = plan["filters"]
    for k in ("ram_gb", "ram_min"):
        if fl[k] is not None and fl[k] not in RAM_SIZES:
            fl[k] = None
    for k in ("price_max", "price_min", "price_target"):
        if fl[k] is not None and fl[k] < MIN_BUDGET:   # "ไม่เกิน 15 นิ้ว" ไม่ใช่ราคา
            fl[k] = None
    if fl["gpu_family"] not in (None, "rtx", "gtx", "radeon rx", "arc", "integrated"):
        fl["gpu_family"] = None
    s = raw.get("sort")
    if isinstance(s, dict) and s.get("by") in SORT_FIELDS:
        plan["sort"] = {"by": s["by"], "order": "desc" if s.get("order") == "desc" else "asc"}
    if plan["limit"] is not None and not 1 <= plan["limit"] <= LIST_CAP:
        plan["limit"] = None
    return plan


# ---- 5.2 ตัวอ่านตัวเลขด้วยโค้ด — เติมสิ่งที่ LLM ลืมใส่ (งบ / แรม / การ์ดจอ / ขนาดจอ) ----
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
    คืน None ถ้าไม่พบงบ

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


_RE_RAM_ASK = re.compile(
    r"(?:แรม|ram|หน่วยความจำ)\s*(?:ขนาด\s*)?(\d{1,3})(?!\d)|(?<!\d)(\d{1,3})\s*(?:gb|กิ๊ก|กิก|จิ๊ก)\s*(?:แรม|ram)",
    re.I,
)
_RE_RAM_AT_LEAST = re.compile(r"ขึ้นไป|อย่างน้อย|ไม่ต่ำกว่า|อย่างต่ำ|หรือมากกว่า|\d\s*(?:gb)?\s*\+", re.I)


def ram_condition(question: str) -> tuple[int, bool] | None:
    """(แรมกี่ GB, ขึ้นไปไหม) — "แรม 24" = 24 GB พอดี / "แรม 24GB ขึ้นไป" = ตั้งแต่ 24 GB"""
    m = _RE_RAM_ASK.search(question)
    if not m:
        return None
    gb = int(m.group(1) or m.group(2))
    if gb not in RAM_SIZES:
        return None
    around = question[max(0, m.start() - 15): m.end() + 15]
    return gb, _RE_RAM_AT_LEAST.search(around) is not None


_RE_GPU_ASK = re.compile(
    r"(?<![a-z0-9])(?:(?:rtx|gtx)\s*\d{4}(?:\s*ti)?|rx\s*\d{4}m?|radeon\s*\d{3}m|arc\s*[a-z]?\d{3}[a-z]?)(?![a-z0-9])",
    re.I,
)


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


_RE_SCREEN = re.compile(r"(?:จอ\s*)?(1[1-9](?:\.\d)?)\s*(?:นิ้ว|\"|inch|in\b)|จอ\s*(1[1-9](?:\.\d)?)(?!\d)", re.I)


def _safety_net(plan: dict, question: str) -> list[str]:
    """เติมตัวเลขที่ตัวอ่านด้วยโค้ด (ทดสอบแล้ว) จับได้แต่ LLM ไม่ได้ใส่ — คืนรายการที่เติม"""
    fl, added = plan["filters"], []
    budget, low = extract_budget(question), extract_min_price(question)
    if budget and not any(fl[k] for k in ("price_max", "price_target")):
        if re.search(r"ประมาณ|ราวๆ|ราว ๆ|แถว", question):
            fl["price_target"] = budget
        else:
            fl["price_max"] = budget
        added.append(f"งบ {budget:,}")
    if low and not fl["price_min"] and fl["price_max"]:
        fl["price_min"] = low
    ram = ram_condition(question)
    if ram and not (fl["ram_gb"] or fl["ram_min"]):
        gb, at_least = ram
        fl["ram_min" if at_least else "ram_gb"] = gb
        added.append(f"แรม {gb}")
    terms = gpu_terms(question)
    if terms and not fl["gpu"]:
        fl["gpu"] = terms[0].upper()
        added.append(terms[0].upper())
    m = _RE_SCREEN.search(question)
    if m and not fl["screen"]:
        fl["screen"] = float(m.group(1) or m.group(2))
        added.append(f"จอ {fl['screen']:g} นิ้ว")
    if plan["intent"] in TEMPLATED and (budget or ram or terms or m):
        plan["intent"] = "recommend"   # มีเงื่อนไขสินค้าชัดเจน ไม่ใช่นอกเรื่อง
    return added


# ---- 5.3 ความละเอียดจอ / SSD / Windows — มีในข้อความสเปกอยู่แล้ว โค้ดอ่านเงื่อนไขจากคำถามเอง (ไม่ฝาก LLM) ----
# รองรับแบบพื้นฐาน: ค่าตรงตัว, "ขึ้นไป/อย่างน้อย", "สูงกว่า/มากกว่า" — ไม่จำข้ามข้อความ
RES_WORDS = (   # คำเรียกความละเอียด -> ความกว้างพิกเซลในข้อมูลร้าน
    (re.compile(r"(?<![a-z\d.])4\s*k(?![a-z0-9])|(?<![a-z])uhd(?![a-z]|\s*graphics|\s*\d)|2160p"), 3840),
    (re.compile(r"(?<![a-z\d.])(?:3|2\.8)\s*k(?![a-z0-9])"), 2880),
    (re.compile(r"(?<![a-z\d.])(?:2|2\.5)\s*k(?![a-z0-9])|(?<![a-z])(?:qhd|wqxga)(?![a-z])|1440p|1600p"), 2560),
    (re.compile(r"(?<![a-z])(?:fhd|full\s*hd|wuxga)(?![a-z])|1080p|1200p"), 1920),
)
_RE_ABOVE = re.compile(r"(?<!ไม่)(?:สูงกว่า|มากกว่า|เกินกว่า)")
_RE_AT_LEAST = re.compile(r"ขึ้นไป|อย่างน้อย|ไม่ต่ำกว่า|เป็นต้นไป")
_RE_SSD_WORD = re.compile(r"ssd|pcie|nvme|m\.2")
_RE_STORAGE = (
    re.compile(r"(\d+(?:\.\d+)?)\s*(tb)(?![a-z])"),
    re.compile(r"(?:ssd|ความจุ|storage|hdd|ฮาร์ดดิส|ที่เก็บข้อมูล)\D{0,12}?(\d+)\s*(gb|tb|กิ๊ก)"),
    re.compile(r"(?<!\d)(256|512)\s*(gb|กิ๊ก)"),
)
OS_WORDS = {"single language": "Windows 11 Home Single Language", "pro": "Windows Pro",
            "windows 10": "Windows 10", "home": "Windows 11 Home", "windows": "Windows"}


def _compare(q: str, start: int, end: int) -> str | None:
    """คำเปรียบเทียบที่อยู่ติดกับค่าที่เจอ: "gt" = สูงกว่า, "min" = ขึ้นไป (ดูแค่รอบ ๆ ไม่ให้ "งบไม่เกิน" ปน)"""
    near = q[max(0, start - 16):start] + " " + q[end:end + 14]
    if _RE_ABOVE.search(near):
        return "gt"
    if _RE_AT_LEAST.search(near):
        return "min"
    return None


def spec_extras(question: str) -> tuple[dict, list[str]]:
    """อ่านเงื่อนไขความละเอียดจอ / SSD Gen / ความจุ / รุ่น Windows จากคำถาม
    คืน (ค่าตัวกรอง, ชื่อเรื่องใน UNSUPPORTED ที่คัดได้แล้ว)"""
    q, fl, covered = question.lower(), {}, []
    m = re.search(r"(\d{4})\s*[x×]\s*(\d{3,4})", q)
    width = int(m.group(1)) if m else None
    if not m:
        for rx, w in RES_WORDS:
            m = rx.search(q)
            if m:
                width = w
                break
    if width:
        how = _compare(q, m.start(), m.end())
        if how == "gt":
            fl["res_gt"] = width
        elif how == "min":
            fl["res_min"] = width
        else:
            fl["res_w"] = width
            if m.re.pattern.startswith(r"(\d{4})"):
                fl["res_h"] = int(m.group(2))
        covered.append("ความละเอียดจอ")
    if _RE_SSD_WORD.search(q):
        gens = sorted({int(g) for g in re.findall(r"gen\s*(\d)(?!\d)", q)})
        if gens:
            g = re.search(r"gen\s*\d(?!\d)", q)
            fl["ssd_gen"] = list(range(gens[0], 6)) if _compare(q, g.start(), g.end()) else gens
            covered.append("Gen ของ SSD")
    for rx in _RE_STORAGE:
        m = rx.search(q)
        if m:
            gb = _gb(m.group(1), "tb" if m.group(2) == "tb" else "gb")
            fl["storage_min" if _compare(q, m.start(), m.end()) else "storage_gb"] = gb
            covered.append("ความจุ SSD")
            break
    if re.search(r"windows(?!\s*hello)|วินโดว์|(?<![a-z])win\s*1[01]|single\s*language", q):
        fl["os"] = ("single language" if re.search(r"single\s*language", q) else
                    "pro" if re.search(r"(?:windows|win)\s*(?:1[01]\s*)?pro(?![a-z])", q) else
                    "windows 10" if re.search(r"(?:windows|win)\s*10", q) else
                    "home" if "home" in q else "windows")
        covered.append("รุ่นของ Windows")
    return fl, covered


# ---- 5.4 เงื่อนไขที่ยังไม่มีตัวกรอง ----
# บางเรื่องอยู่ในข้อความสเปก (ความละเอียดจอ, Gen ของ SSD, รุ่น Windows — spec_extras คัดได้แล้ว)
# บางเรื่องไม่มีในข้อมูลเลย (Hz, น้ำหนัก, พอร์ต) ถ้าไม่จับไว้ ตัวตีความจะทิ้งเงื่อนไขเงียบ ๆ
# แล้วระบบส่งรุ่นที่ไม่ได้กรองไปให้ AI ซึ่งจะเขียนว่า "มี 40 รุ่นตรงเงื่อนไข" (ชุดคำถามลูกค้า 7 ข้อ)
UNSUPPORTED = (
    ("ความถี่จอ (Hz)", re.compile(r"\d{2,3}\s*(?:hz|เฮิร์?ตซ์)|refresh|รีเฟรช")),
    ("ความละเอียดจอ", re.compile(
        r"(?<![\d.,])[2-8](?:\.5)?\s*k(?![a-z0-9])|(?<![a-z])(?:qhd|fhd|wuxga|wqxga|full\s*hd)(?![a-z])"
        r"|(?<![a-z])uhd(?![a-z]|\s*graphics|\s*\d)|(?<!\d)(?:1080|1440|1600|2160)p|\d{4}\s*[x×]\s*\d{3,4}"
        r"|ความละเอียด|resolution")),
    ("Gen ของ SSD", re.compile(r"(?:ssd|pcie|nvme|m\.2).{0,20}?gen\s*\d|gen\s*\d.{0,20}?(?:ssd|pcie|nvme)")),
    ("ความจุ SSD", re.compile(
        r"\d+(?:\.\d+)?\s*tb(?![a-z])|(?:ssd|ความจุ|storage|hdd|ฮาร์ดดิส|ที่เก็บข้อมูล)\D{0,12}\d+\s*(?:gb|tb|กิ๊ก)"
        r"|(?<!\d)(?:256|512)\s*(?:gb|กิ๊ก)")),
    ("รุ่นของ Windows", re.compile(r"windows(?!\s*hello)|วินโดว์|(?<![a-z])win\s*1[01]|single\s*language|ระบบปฏิบัติการ")),
    ("น้ำหนักเครื่อง", re.compile(r"น้ำหนัก|\d(?:\.\d+)?\s*(?:kg|กก\.?|กิโล)(?![a-z])")),
    ("พอร์ตและช่องเชื่อมต่อ", re.compile(
        r"usb[\s-]*c|type[\s-]*c|thunderbolt|hdmi|(?<![a-z])lan(?![a-z])|rj[\s-]*45|พอร์ต|ช่องเสียบ|ช่องต่อ"
        r"|sd\s*card|การ์ดรีดเดอร์")),
    ("แป้นตัวเลข (Number pad)", re.compile(r"num\s*pad|number\s*pad|แป้นตัวเลข|ปุ่มตัวเลข")),
    ("ไฟคีย์บอร์ด", re.compile(r"ไฟคีย์บอร์ด|ไฟที่คีย์บอร์ด|คีย์บอร์ดมีไฟ|backlit|backlight|(?<![a-z])rgb(?![a-z])")),
    ("มาตรฐาน MIL-STD", re.compile(r"mil[\s-]*std|มาตรฐานทหาร|มาตรฐานกองทัพ")),
    ("การสแกนนิ้วหรือสแกนหน้า", re.compile(r"สแกนนิ้ว|ลายนิ้วมือ|fingerprint|สแกนหน้า|windows\s*hello")),
    ("Wi-Fi หรือ Bluetooth", re.compile(r"wi[\s-]*fi\s*(?:6e?|7)|bluetooth\s*\d")),
    ("ชั่วโมงใช้งานแบต", re.compile(
        r"\d+\s*wh(?![a-z])|แบต\S*\s*(?:อยู่ได้|ใช้ได้)?\s*(?:นาน\s*)?\d+(?:\s*-\s*\d+)?\s*(?:ชั่วโมง|ชม)")),
    ("จอสัมผัส", re.compile(r"จอสัมผัส|จอทัช|ทัชสกรีน|touch\s*screen")),
)
SEARCHABLE = "ยี่ห้อ ซีรีส์ ราคา CPU การ์ดจอ แรม ขนาดจอ ความละเอียดจอ SSD และรุ่น Windows"


def unsupported_terms(question: str) -> list[str]:
    """เรื่องที่ลูกค้าขอแต่ระบบยังคัดรุ่นให้ไม่ได้ (คืนชื่อเรื่องภาษาไทย ไม่ซ้ำ ตามลำดับในตาราง)"""
    q = question.lower()
    return [label for label, rx in UNSUPPORTED if rx.search(q)]


# ---- 5.5 กฎหลัง LLM: การใช้งานที่ลูกค้าไม่ได้พูด / "แนะนำ" ที่จริงคือขอรายชื่อ ----
USAGE_HINTS = {
    "เกม": ("เกม", "game", "gaming", "valorant", "rov", "pubg", "gta", "fps", "esport", "สตรีม", "stream"),
    "ตัดต่อ": ("ตัดต่อ", "edit", "premiere", "davinci", "render", "เรนเดอร์", "กราฟิก", "graphic", "3d", "blender",
               "photoshop", "ออกแบบ", "design", "autocad", "วิดีโอ", "video", "คลิป"),
    "ทำงาน": ("ทำงาน", "office", "ออฟฟิศ", "เอกสาร", "excel", "word", "powerpoint"),
    "เรียน": ("เรียน", "นักศึกษา", "นักเรียน", "มหาลัย", "study"),
    "โปรแกรม": ("โปรแกรม", "โค้ด", "code", "coding", "dev"),
    "พกพา": ("พกพา", "เบา", "บาง", "แบต", "battery", "portable"),
    "ai": ("ai", "machine learning", "deep learning", "llm"),
}
_RE_ADVICE = re.compile(r"แนะนำ|ตัวไหนดี|รุ่นไหนดี|อันไหนดี|คุ้ม|เหมาะ|ควรซื้อ|ช่วยเลือก")
SPEC_FILTERS = ("ram_gb", "ram_min", "gpu", "gpu_family", "cpu", "brand", "series", "screen",
                "res_w", "res_min", "res_gt", "ssd_gen", "storage_gb", "storage_min", "os")
_RE_LIST_Q = re.compile(r"(?:รุ่น|ตัว|เครื่อง|อะไร|ไหน)\s*(?:ไหน|อะไร)?\s*บ้าง|ทุกรุ่น|ทุกตัว|ทั้งหมด")


def _ground_usage(plan: dict, question: str, history: list, last_plan: dict | None) -> str | None:
    """ทิ้ง usage ที่ลูกค้าไม่ได้พูดถึง (ในข้อความนี้ ข้อความก่อน ๆ หรือเงื่อนไขเดิม) — คืนค่าที่ทิ้ง"""
    usage = plan["usage"]
    if not usage:
        return None
    said = " ".join([question] + [h["content"] for h in history[-HISTORY_TURNS:] if h["role"] == "user"]).lower()
    if last_plan and last_plan.get("usage"):
        said += " " + str(last_plan["usage"]).lower()
    u = usage.lower()
    cats = [kws for kws in USAGE_HINTS.values() if any(k in u for k in kws)]
    grounded = any(k in said for kws in cats for k in kws) if cats else u in said
    if grounded:
        return None
    plan["usage"] = None
    return usage


def check_plan(plan: dict, question: str, history: list, last_plan: dict | None) -> list[str]:
    """กฎหลัง LLM แปลงคำถาม: ทิ้ง usage ที่ลูกค้าไม่ได้พูด และแก้ "แนะนำ" ที่จริงคือขอรายชื่อ — คืนสิ่งที่แก้"""
    fixed = []
    dropped = _ground_usage(plan, question, history, last_plan)
    if dropped:
        fixed.append(f"ตัดการใช้งาน «{dropped}» ออก (ลูกค้าไม่ได้พูด)")
    if plan["intent"] == "recommend" and not plan["usage"] and not plan["limit"] and not _RE_ADVICE.search(question):
        # "มีรุ่นไหนบ้าง" / "ขอรุ่นที่ใช้ Core i7" ไม่ได้บอกการใช้งานและไม่ได้ขอคำแนะนำ
        # = ขอรายชื่อทุกรุ่นที่ตรง ไม่ใช่ 3 รุ่นแนะนำ
        if _RE_LIST_Q.search(question):
            plan["intent"] = "list"
            fixed.append("เปลี่ยนเป็นขอรายชื่อ (ถามว่ามีรุ่นไหนบ้าง)")
        elif any(plan["filters"][k] for k in SPEC_FILTERS):
            plan["intent"] = "list"
            fixed.append("เปลี่ยนเป็นขอรายชื่อ (ระบุสเปกแต่ไม่ได้ขอคำแนะนำ)")
    return fixed


# ---- 5.6 รวมทุกขั้นของการตีความ ----
def understand(question: str, history: list, last_plan: dict | None, provider: dict,
               trace: dict | None = None) -> dict:
    started = time.perf_counter()
    user = (f"ยี่ห้อในร้าน: Lenovo, Asus, MSI, Acer, HP, HyperX, Gigabyte\nซีรีส์ในร้าน: {_series_list()}\n\n"
            f"เงื่อนไขเดิม: {json.dumps(last_plan, ensure_ascii=False) if last_plan else '(ไม่มี)'}\n"
            f"บทสนทนาล่าสุด:\n{_history_text(history)}\n\nข้อความใหม่ของลูกค้า: {question}\nJSON:")
    raw_text, err = "", None
    try:
        raw_text, _, _ = complete(provider, [{"role": "system", "content": UNDERSTAND_RULES},
                                             {"role": "user", "content": user}],
                                  max_tokens=400, temperature=0, json_mode=True)
        plan = normalize(_parse_json(raw_text))
    except (ValueError, json.JSONDecodeError) as e:
        err = f"อ่าน JSON ไม่ได้ ({e}) — ใช้ตัวอ่านตัวเลขเดิมแทน"
        plan = normalize({})
    added = _safety_net(plan, question)
    extras, covered = spec_extras(question)
    if extras:
        plan["filters"].update(extras)
        added.append("อ่านจากคำถาม: " + describe({**normalize({})["filters"], **extras}))
        if plan["intent"] in ("general", "offtopic", "greeting", "overview"):
            plan["intent"] = "list"   # "มีกี่รุ่นที่ใช้ SSD Gen 5" ต้องคัดรุ่น ไม่ใช่ภาพรวมร้าน
        if plan["intent"] == "rank" and not plan["sort"]:
            # LLM ไม่มีช่องให้ใส่ความละเอียด/SSD เลยเดาเป็น "จัดอันดับ 1 รุ่น" — ไม่มีเกณฑ์เรียง = ขอรายชื่อ
            plan["intent"], plan["limit"] = "list", None
    added += check_plan(plan, question, history, last_plan)
    plan["unsupported"] = [t for t in unsupported_terms(question) if t not in covered]
    if plan["unsupported"]:
        added.append("ยังคัดตามเรื่องนี้ไม่ได้: " + ", ".join(plan["unsupported"]))
    if trace is not None:
        trace["understand"] = {"ms": round((time.perf_counter() - started) * 1000), "raw": raw_text[:1500],
                               "plan": plan, "added": added, "error": err}
    return plan


# ═════════════════════════ 6. คัดรุ่นจากข้อมูลจริง ═════════════════════════
RECOMMEND_POOL = 3     # รุ่นที่ส่งให้ AI เลือกแนะนำ (ถ้าลูกค้าไม่ได้ขอจำนวน)
RANK_TOP = 5           # อันดับที่แนบไปให้ AI เล่าต่อในคำถามจัดอันดับ
PRICE_BAND = 0.15      # "ราคาประมาณ X" = X ±15%


def resolve_models(mentions: list[str]) -> tuple[list[dict], list[str]]:
    """ชื่อรุ่น/ซีรีส์ที่ลูกค้าเอ่ย -> รุ่นในร้าน (คืน รุ่นที่เจอ, คำที่หาไม่เจอ)"""
    cat, found, missing = catalog(), [], []
    for m in mentions:
        q = re.sub(r"\s+", " ", m.lower()).strip()
        hits = [lp for lp in cat if q in lp["name"].lower()]
        if not hits:   # ชื่อย่อ เช่น "Legion 5" / "Victus 16" -> เทียบกับซีรีส์
            hits = [lp for lp in cat if q in lp["series"].lower() or lp["series"].lower() in q]
        if not hits:
            hits = [lp for lp in cat if any(len(t) >= 6 and t.lower() in q for t in lp["name"].split() if "-" in t)]
        if hits:
            found += [lp for lp in hits if lp not in found]
        else:
            missing.append(m)
    return found[:NAMED_CAP * 2], missing


def screen_ok(size: float | None, want: float) -> bool:
    """"จอ 16" = รุ่นจอ 16.x ทั้งหมด (ไม่รวม 15.6) / "จอ 15.6" = 15.6 พอดี"""
    if size is None:
        return False
    if want == int(want):
        return int(size) == int(want)
    return abs(size - want) < 0.05


def _match(lp: dict, fl: dict) -> bool:
    if fl["price_max"] and lp["price"] > fl["price_max"]:
        return False
    if fl["price_min"] and lp["price"] < fl["price_min"]:
        return False
    if fl["ram_gb"] and lp["ram_gb"] != fl["ram_gb"]:
        return False
    if fl["ram_min"] and lp["ram_gb"] < fl["ram_min"]:
        return False
    if fl["gpu"] and not gpu_matches(lp["gpu"], re.sub(r"([a-z])(\d)", r"\1 \2", fl["gpu"].lower())):
        return False
    if fl["gpu_family"] and lp["gpu_family"] != fl["gpu_family"]:
        return False
    if fl["cpu"] and fl["cpu"].lower() not in lp["cpu"].lower():
        return False
    if fl["brand"] and fl["brand"].lower() != lp["brand"].lower():
        return False
    if fl["series"] and fl["series"].lower() not in lp["name"].lower():
        return False
    if fl["screen"] and not screen_ok(lp["screen"], fl["screen"]):
        return False
    if fl.get("res_w") and lp["res_w"] != fl["res_w"]:
        return False
    if fl.get("res_h") and lp["res_h"] != fl["res_h"]:
        return False
    if fl.get("res_min") and (lp["res_w"] or 0) < fl["res_min"]:
        return False
    if fl.get("res_gt") and (lp["res_w"] or 0) <= fl["res_gt"]:
        return False
    if fl.get("ssd_gen") and lp["ssd_gen"] not in fl["ssd_gen"]:
        return False
    if fl.get("storage_gb") and lp["storage_gb"] != fl["storage_gb"]:
        return False
    if fl.get("storage_min") and (lp["storage_gb"] or 0) < fl["storage_min"]:
        return False
    if fl.get("os") and fl["os"] not in lp["os"].lower():
        return False
    return True


def _size(gb: int) -> str:
    return f"{gb / 1000:g}TB" if gb >= 1000 else f"{gb}GB"


def describe(fl: dict) -> str:
    """เงื่อนไขเป็นภาษาคน เช่น "แรม 24GB · ราคาไม่เกิน 30,000 บาท" (ว่าง = ไม่ได้กรอง)"""
    parts = []
    if fl["brand"]:
        parts.append(f"ยี่ห้อ {fl['brand']}")
    if fl["series"]:
        parts.append(f"ซีรีส์ {fl['series']}")
    if fl["gpu"]:
        parts.append(f"การ์ดจอ {fl['gpu']}")
    if fl["gpu_family"]:
        parts.append(f"การ์ดจอกลุ่ม {fl['gpu_family'].upper()}")
    if fl["cpu"]:
        parts.append(f"CPU {fl['cpu']}")
    if fl["ram_gb"]:
        parts.append(f"แรม {fl['ram_gb']}GB")
    if fl["ram_min"]:
        parts.append(f"แรม {fl['ram_min']}GB ขึ้นไป")
    if fl["screen"]:
        parts.append(f"จอ {fl['screen']:g} นิ้ว")
    if fl.get("res_w"):
        parts.append(f"ความละเอียดจอ {fl['res_w']}x{fl['res_h']}" if fl.get("res_h")
                     else f"ความละเอียดจอแนวนอน {fl['res_w']} พิกเซล")
    if fl.get("res_min"):
        parts.append(f"ความละเอียดจอแนวนอน {fl['res_min']} พิกเซลขึ้นไป")
    if fl.get("res_gt"):
        parts.append(f"ความละเอียดจอแนวนอนมากกว่า {fl['res_gt']} พิกเซล")
    if fl.get("ssd_gen"):
        parts.append("SSD Gen " + " หรือ ".join(str(g) for g in fl["ssd_gen"]))
    if fl.get("storage_gb"):
        parts.append(f"ความจุ {_size(fl['storage_gb'])}")
    if fl.get("storage_min"):
        parts.append(f"ความจุ {_size(fl['storage_min'])} ขึ้นไป")
    if fl.get("os"):
        parts.append(OS_WORDS.get(fl["os"], fl["os"]))
    if fl["price_min"] and fl["price_max"]:
        parts.append(f"ราคา {fl['price_min']:,}–{fl['price_max']:,} บาท")
    elif fl["price_max"]:
        parts.append(f"ราคาไม่เกิน {fl['price_max']:,} บาท")
    elif fl["price_min"]:
        parts.append(f"ราคาตั้งแต่ {fl['price_min']:,} บาท")
    if fl["price_target"]:
        parts.append(f"ราคาใกล้ {fl['price_target']:,} บาท")
    return " · ".join(parts)


def select(plan: dict, question: str, trace: dict | None = None) -> dict:
    """คืน {"items": [...], "notes": [...], "named_missing": [...]} เรียงตามลำดับที่ควรนำเสนอ"""
    started = time.perf_counter()
    cat, fl, intent = catalog(), plan["filters"], plan["intent"]
    named, missing = resolve_models(plan["models"])
    notes, source = [], "filter"

    if intent in ("lookup", "compare") and named:
        items, source = list(named), "named"
        if fl["price_max"] or fl["price_min"]:
            over = [lp for lp in items if not _match(lp, fl)]
            if over:
                notes.append("รุ่นที่ลูกค้าเอ่ยถึงมีในร้าน แต่ราคาอยู่นอกงบ: "
                             + ", ".join(f"{lp['name']} ({lp['price']:,} บาท)" for lp in over[:6])
                             + " — ให้บอกว่ามีแต่เกินงบ ห้ามบอกว่าไม่มีในข้อมูล")
    else:
        items = [lp for lp in cat if _match(lp, fl)]
        if named and intent in ("list", "rank", "recommend"):
            narrowed = [lp for lp in items if lp in named]
            items = narrowed or items
            over = [lp for lp in named if lp not in items]
            if over and (fl["price_max"] or fl["price_min"]):
                notes.append("รุ่นที่ลูกค้าเอ่ยถึงมีในร้าน แต่ราคาอยู่นอกงบ: "
                             + ", ".join(f"{lp['name']} ({lp['price']:,} บาท)" for lp in over[:6]))
        if items and fl["price_target"]:
            # "ราคาประมาณ X" = ช่วง ±15% ไม่ใช่ทุกรุ่นเรียงตามความใกล้ (ไล่ 13 รุ่นตั้งแต่ 25k ถึง 54k ไม่ตอบโจทย์)
            t = fl["price_target"]
            band = [lp for lp in items if t * (1 - PRICE_BAND) <= lp["price"] <= t * (1 + PRICE_BAND)]
            if band:
                items = band
            else:
                items = sorted(items, key=lambda lp: (abs(lp["price"] - t), lp["price"]))[:RECOMMEND_POOL]
                source = "nearest"
                notes.append(f"ไม่มีรุ่นที่ราคาใกล้ {t:,} บาท (±{PRICE_BAND:.0%}) — รายการคือรุ่นที่ราคาใกล้ที่สุด "
                             "ให้บอกลูกค้าตรง ๆ แล้วเสนอรุ่นเหล่านี้")
        if not items and any(fl[k] for k in ("price_max", "price_min", "price_target")):
            # ไม่มีรุ่นในช่วงราคาเลย -> ตัดเงื่อนไขราคาออก แล้วเสนอรุ่นที่ราคาใกล้ที่สุด
            loose = [lp for lp in cat if _match(lp, {**fl, "price_max": None, "price_min": None})]
            target = fl["price_target"] or fl["price_max"] or fl["price_min"]
            items = sorted(loose, key=lambda lp: (abs(lp["price"] - target), lp["price"]))[:RECOMMEND_POOL]
            source = "nearest"
            notes.append(f"ไม่มีรุ่นที่ตรงทุกเงื่อนไข ({describe(fl)}) — รายการด้านล่างคือรุ่นที่ราคาใกล้ที่สุด "
                         f"ให้บอกลูกค้าตรง ๆ ว่าไม่มีรุ่นในช่วงราคานี้ แล้วเสนอรุ่นเหล่านี้แทน")

    # ลำดับ
    sims = {}
    if plan["sort"]:
        key = SORT_FIELDS[plan["sort"]["by"]]
        items.sort(key=lambda lp: (lp[key], lp["price"]) if plan["sort"]["order"] == "asc"
                   else (-lp[key], lp["price"]))
    elif source != "nearest" and fl["price_target"]:
        items.sort(key=lambda lp: (abs(lp["price"] - fl["price_target"]), lp["price"]))
    elif source != "nearest" and (plan["usage"] or intent == "recommend") and items:
        # เรียงตามความใกล้เคียงระหว่าง "การใช้งาน" (หรือทั้งคำถาม) กับเอกสารของแต่ละรุ่น
        t0 = time.perf_counter()
        q_vec = np.asarray(get_embedder().encode([plan["usage"] or question], normalize_embeddings=True))[0]
        vecs = doc_vectors()
        index = {lp["name"]: i for i, lp in enumerate(cat)}
        sims = {lp["name"]: float(vecs[index[lp["name"]]] @ q_vec) for lp in items}
        items.sort(key=lambda lp: -sims[lp["name"]])
        if trace is not None:
            trace["embed"] = {"model": EMBED_MODEL, "dims": int(q_vec.shape[0]),
                              "ms": round((time.perf_counter() - t0) * 1000), "load_ms": 0,
                              "preview": [round(float(v), 4) for v in q_vec[:8]]}
    elif source not in ("named",):
        items.sort(key=lambda lp: lp["price"])

    total = len(items)
    if intent == "list":
        keep = LIST_CAP
    elif intent == "rank":
        keep = max(plan["limit"] or 1, RANK_TOP)
    elif intent == "recommend":
        keep = max(plan["limit"] or 1, RECOMMEND_POOL)
    elif intent in ("lookup", "compare"):
        keep = plan["limit"] or NAMED_CAP
    else:
        keep = 0
    items = items[:keep]
    if intent == "list" and total > LIST_CAP:
        notes.append(f"ตรงเงื่อนไขทั้งหมด {total} รุ่น แสดง {LIST_CAP} รุ่นแรก")

    if trace is not None:
        trace["search_ms"] = round((time.perf_counter() - started) * 1000)
        trace["adjust"] = ([f"เงื่อนไข: {describe(fl)}"] if describe(fl) else []) + notes \
            + ([f"หาชื่อรุ่นไม่เจอ: {', '.join(missing)}"] if missing else [])
        if sims:
            trace["searches"] = [{"type": "spec", "want": len(items), "filter": describe(fl) or "ไม่กรอง",
                                  "hits": [{"name": lp["name"], "price": lp["price"], "similarity": round(sims[lp["name"]], 4)}
                                           for lp in items]}]
        trace["context"] = [{"source": source, "kind": "spec", "text": lp["doc"]} for lp in items]
    return {"items": items, "total": total, "notes": notes, "missing": missing, "source": source}


# ═════════════════════════ 7. ตอบ (เรียก LLM ครั้งที่ 2 แบบสตรีม) ═════════════════════════
OUTSIDE_BRANDS = ("macbook", "apple", "dell", "thinkpad", "surface", "razer", "alienware", "samsung", "huawei", "microsoft")
BRAND_CASE = {"macbook": "MacBook", "thinkpad": "ThinkPad", "alienware": "Alienware"}
EMPTY_ANSWER = "(โมเดลคิดยาวจน token หมดก่อนตอบ ลองถามใหม่อีกครั้งนะ)"

ANSWER_RULES = """คุณคือ "Tech Buddy" พนักงานแนะนำโน้ตบุ๊คของร้าน (ข้อมูล 99 รุ่น) พูดสุภาพเป็นกันเองแบบเพื่อนที่รู้เรื่องโน้ตบุ๊ค
ลงท้ายด้วย "ครับ" (คำเดียว ไม่ซ้ำเป็น "ครับครับ") เรียกลูกค้าว่า "คุณ" ช่วยลูกค้าตัดสินใจอย่างจริงใจ ไม่ยัดเยียด ไม่กดดัน
ไม่ต้องทักทายหรือขึ้นต้นด้วย "สวัสดี" ให้เริ่มตอบคำถามเลย
กฎ:
1. ใช้เฉพาะข้อมูลใน "รายการที่ระบบคัดมา" และ "ข้อมูลร้าน" ห้ามแต่งรุ่น ราคา สเปก หรือคะแนน และห้ามใช้ความรู้นอกข้อมูล
2. ระดับคะแนน (เริ่มต้น / กลาง / กลาง-สูง / สูง) และส่วนต่างงบ ใช้ตามที่ระบบเขียนไว้ ห้ามคำนวณหรือเดาใหม่
3. เขียนชื่อรุ่นเต็มตามรายการทุกครั้ง
4. ถ้ามีหมายเหตุจากระบบ ให้ทำตามหมายเหตุนั้น
5. ตอบตามรูปแบบใน "งานของคุณ" ใช้ markdown (หัวข้อด้วย **ตัวหนา** และ bullet "-") ห้ามใช้ตาราง
6. ใส่รายละเอียดให้ครบตามที่ "งานของคุณ" บอก ไม่ต้องย่อ ลูกค้าอยากเห็นข้อมูลก่อนตัดสินใจ
7. ถ้ารายการมีรุ่นสินค้า (ยกเว้นคำถามภาพรวมร้าน) ปิดท้ายคำตอบด้วยหัวข้อ **คำแนะนำจาก Tech Buddy** 2-3 ประโยค:
   ถ้าให้เลือก รุ่นไหนในรายการน่าซื้อที่สุดสำหรับลูกค้าคนนี้ เพราะอะไร (อ้างสเปก ราคา หรือรีวิว)
   แล้วชวนคุยต่อแบบเป็นกันเอง เช่น ให้ช่วยเทียบรุ่น ดูสเปกเพิ่ม หรือบอกการใช้งานเพื่อเลือกให้ตรงขึ้น
   - เสนอได้เฉพาะรุ่นในรายการ ห้ามพูดถึงรุ่นอื่น
   - ห้ามพูดถึงโปรโมชั่น ส่วนลด ของแถม สต็อก การผ่อน หรือการรับประกัน เพราะไม่มีในข้อมูล
   - ห้ามเร่งให้ซื้อ เช่น บอกให้รีบตัดสินใจ หรือบอกว่าของใกล้หมด"""

# รูปแบบรายละเอียดของ 1 รุ่น (ใช้ซ้ำในหลายประเภทคำถาม)
FULL_SPEC = ("'- **สเปก:** CPU ... (คะแนน X ระดับ...) / GPU ... (คะแนน X ระดับ...) / แรม ... / ความจุ ... / จอ ... / OS' "
             "'- **เหมาะกับ:** งานที่สเปกและรีวิวบอกว่าทำได้ดี' '- **ข้อดี:** 2 ข้อจากรีวิว' '- **ข้อควรรู้:** 1-2 ข้อจากรีวิว'")
LIST_FULL_MAX = 15   # รายชื่อไม่เกินนี้ ใส่สเปกหลักทุกรุ่น / ยาวกว่านี้ใส่เฉพาะสเปกที่ลูกค้าถาม (คำตอบจะยาวเกินไป)

TASKS = {
    "list": ("ไล่ชื่อ **ครบทุกรุ่นในรายการ ({n} รุ่น)** ตามลำดับเดิม ห้ามตัดหรือข้าม ห้ามเพิ่มรุ่นนอกรายการ "
             "เริ่มด้วย 1 ประโยคบอกว่ามีทั้งหมดกี่รุ่น แล้วรุ่นละ 1 bullet: {list_line}"),
    "rank": ("รุ่นแรกในรายการคือคำตอบ (เรียงตามเกณฑ์ที่ลูกค้าถามแล้ว) ถ้ารุ่นต้น ๆ มีค่าเท่ากันให้บอกว่าเท่ากัน "
             "เขียนหัวข้อ **ชื่อรุ่นเต็ม — ราคา X บาท** บอกค่าตามเกณฑ์ (คะแนนหรือราคา) แล้วตามด้วย " + FULL_SPEC +
             " จากนั้นหัวข้อ **อันดับถัดไป** อีก 2 รุ่น บรรทัดละรุ่น: ชื่อรุ่นเต็ม — ราคา — ค่าตามเกณฑ์ — CPU / GPU"),
    "recommend": ("แนะนำ {n} รุ่นแรกในรายการ (ระบบเรียงตามความเหมาะสมแล้ว) รุ่นละหัวข้อ "
                  "**ชื่อรุ่นเต็ม — ราคา X บาท** (+ ส่วนต่างงบตามที่ระบบเขียน) แล้วตามด้วย " + FULL_SPEC +
                  " ถ้าแนะนำมากกว่า 1 รุ่น ตามด้วย **สรุปเปรียบเทียบ** ว่าแต่ละรุ่นเด่นเรื่องไหน "
                  "ถ้าลูกค้ายังไม่บอกการใช้งาน ให้ถามการใช้งานในส่วนคำแนะนำท้ายคำตอบ"),
    "lookup": ("ตอบสิ่งที่ลูกค้าถามให้ตรงคำถามก่อนในประโยคแรก (ถ้าคำถามเลือกรุ่นตามเกณฑ์ เช่น ตัวที่ถูกกว่า ให้ตอบรุ่นแรกในรายการ) "
               "แล้วให้รายละเอียดของรุ่นนั้นใต้หัวข้อ **ชื่อรุ่นเต็ม — ราคา X บาท** ตามด้วย " + FULL_SPEC),
    "compare": ("เทียบทุกรุ่นในรายการ รุ่นละหัวข้อ **ชื่อรุ่นเต็ม — ราคา X บาท** ตามด้วย " + FULL_SPEC +
                " แล้วปิดด้วย **สรุปเปรียบเทียบ**: ต่างกันตรงไหน (CPU GPU แรม จอ) และแต่ละรุ่นเหมาะกับใคร{diff}"),
    "overview": ("ตอบตรงคำถามจาก 'ข้อมูลร้าน' ในประโยคแรก แล้วขยายให้น่าอ่าน: "
                 "ถ้าถามราคาเริ่มต้น/ถูกสุด หรือราคาสูงสุด/แพงสุด ให้เขียนหัวข้อ **ชื่อรุ่นเต็ม — ราคา X บาท** ของรุ่นนั้นจากรายการ "
                 "ตามด้วย '- **สเปก:** ...' และ '- **จุดเด่น:** 1-2 ข้อจากรีวิว' / "
                 "ถ้าถามยี่ห้อ ให้ไล่ทุกยี่ห้อพร้อมจำนวนรุ่นและช่วงราคา / ถ้าถามจำนวนรุ่น ให้บอกจำนวนแยกยี่ห้อด้วย "
                 "แล้วปิดด้วย 1-2 ประโยคชวนบอกการใช้งานหรืองบ เพื่อช่วยเลือกรุ่นที่ใช่ (ไม่ต้องมีหัวข้อคำแนะนำ)"),
}

TEMPLATES = {
    "offtopic": "เรื่องนี้ผมตอบไม่ได้ครับ ผมช่วยได้เฉพาะเรื่องเลือกโน้ตบุ๊ค ตอนนี้มองหาโน้ตบุ๊คไว้ใช้ทำอะไร งบประมาณประมาณเท่าไหร่ครับ?",
    "greeting": "สวัสดีครับ ผม Tech Buddy ช่วยเลือกโน้ตบุ๊คจาก 99 รุ่นในร้านครับ ตอนนี้มองหาโน้ตบุ๊คไว้ใช้ทำอะไร งบประมาณประมาณเท่าไหร่ครับ?",
    "general": "ขอโทษครับ ข้อมูลที่ผมมีไม่ครอบคลุมเรื่องนี้ครับ ถ้าอยากให้ช่วยเลือกโน้ตบุ๊ค บอกงบและการใช้งานมาได้เลยครับ",
    "not_in_store": ("ขอโทษครับ ไม่มี{thing}ในข้อมูลที่ผมมีครับ ในร้านมียี่ห้อ Lenovo, Asus, MSI, Acer, HP, HyperX และ Gigabyte "
                     "ถ้าบอกงบและการใช้งาน ผมช่วยเลือกรุ่นที่ใกล้เคียงให้ได้ครับ"),
}


# ---- 7.1 เงื่อนไขที่ยังคัดไม่ได้ -> บอกลูกค้าตรง ๆ (ข้อความสำเร็จรูป หรือประโยคนำต้นคำตอบ) ----
def _thai_join(words: list[str]) -> str:
    if len(words) == 1:
        return words[0]
    return ", ".join(words[:-1]) + " และ" + (" " if words[-1][:1].isascii() else "") + words[-1]


def _pad(text: str) -> str:
    """เว้นวรรคระหว่างคำไทยกับคำอังกฤษ เช่น "ตาม" + "Gen ของ SSD" + "ให้" -> "ตาม Gen ของ SSD ให้" """
    return (" " if text[:1].isascii() else "") + text + (" " if text[-1:].isascii() else "")


def unsupported_reply(plan: dict) -> str | None:
    """ลูกค้าถามแต่เรื่องที่ยังคัดไม่ได้ (ไม่มีเงื่อนไขอื่น ไม่ระบุรุ่น ไม่บอกการใช้งาน) -> ตอบตรง ๆ แทนการส่งรุ่นที่ไม่ได้กรอง"""
    labels, intent = plan.get("unsupported") or [], plan["intent"]
    if not labels or plan["models"]:
        return None
    if intent == "overview" or (intent in ("list", "recommend", "rank")
                                and not plan["usage"] and not describe(plan["filters"])):
        return (f"ตอนนี้ผมยังคัดรุ่นตาม{_pad(_thai_join(labels))}ให้ไม่ได้ครับ ระบบค้นได้ตาม{SEARCHABLE} "
                "ถ้าสนใจรุ่นไหน ถามสเปกของรุ่นนั้นได้เลย หรือบอกงบและการใช้งาน ผมช่วยเลือกให้ได้ครับ")
    return None


def unsupported_lead(plan: dict) -> str:
    """มีเงื่อนไขอื่นที่คัดได้ด้วย -> โค้ดใส่ประโยคนี้ไว้ต้นคำตอบเอง (ไม่ฝากให้ AI จำ)"""
    labels = plan.get("unsupported") or []
    if not labels or plan["intent"] not in ("list", "recommend", "rank") or unsupported_reply(plan):
        return ""
    used = describe(plan["filters"]).split(" · ") if describe(plan["filters"]) else []
    if plan["models"]:
        used.append("รุ่นที่คุณถามถึง")
    if plan["usage"]:
        used.append("การใช้งานที่คุณบอก")
    return (f"ผมยังคัดรุ่นตาม{_pad(_thai_join(labels))}ให้ไม่ได้ครับ "
            f"รายการด้านล่างจึงคัดตาม{_pad(_thai_join(used))}อย่างเดียว ถ้าสนใจรุ่นไหน ถามสเปกของรุ่นนั้นต่อได้เลย\n\n")


# ---- 7.2 ประกอบ prompt ของขั้นตอบ ----
def _price_note(lp: dict, fl: dict) -> str:
    """ส่วนต่างงบ คำนวณด้วยโค้ด (LLM บวกลบเลขพลาดได้ เช่น 25,000 - 24,990 ตอบเป็น 1 แทนที่จะเป็น 10)"""
    if fl["price_max"]:
        diff = fl["price_max"] - lp["price"]
        return f"อยู่ในงบ เหลือ {diff:,} บาท" if diff >= 0 else f"เกินงบ {-diff:,} บาท"
    if fl["price_target"]:
        diff = lp["price"] - fl["price_target"]
        return "ตรงราคาที่ต้องการ" if diff == 0 else (f"แพงกว่าที่ต้องการ {diff:,} บาท" if diff > 0
                                                    else f"ถูกกว่าที่ต้องการ {-diff:,} บาท")
    return ""


def _spec_hint(plan: dict) -> str:
    if plan["detail"]:
        return "CPU, GPU, แรม, จอ, ความจุ และคะแนน CPU/GPU"
    fl, bits = plan["filters"], []
    if fl["ram_gb"] or fl["ram_min"]:
        bits.append("แรม")
    if fl["gpu"] or fl["gpu_family"]:
        bits.append("การ์ดจอ")
    if fl["cpu"]:
        bits.append("CPU")
    if fl["screen"] or fl.get("res_w") or fl.get("res_min") or fl.get("res_gt"):
        bits.append("จอ")
    if fl.get("ssd_gen") or fl.get("storage_gb") or fl.get("storage_min"):
        bits.append("ความจุ")
    if fl.get("os"):
        bits.append("OS")
    return ", ".join(bits or ["CPU", "การ์ดจอ", "แรม"])


def _list_line(plan: dict, n: int) -> str:
    """รูปแบบบรรทัดของคำตอบแบบรายชื่อ — รายชื่อสั้นใส่สเปกหลักครบ รายชื่อยาวใส่เฉพาะที่ลูกค้าถาม"""
    if plan["detail"] or n <= LIST_FULL_MAX:
        line = ("'- **ชื่อรุ่นเต็ม** — X บาท — CPU ... (ระดับ...) · GPU ... (ระดับ...) · แรม ... · ความจุ ... · จอ ...'")
        if plan["detail"]:
            line += " ต่อด้วยคะแนน CPU/GPU และจุดเด่นจากรีวิว 1 ข้อ"
        return line
    return f"'- **ชื่อรุ่นเต็ม** — X บาท — {_spec_hint(plan)}'"


def _understood(plan: dict) -> str:
    fl = plan["filters"]
    bits = [f"ประเภทคำถาม: {plan['intent']}"]
    if describe(fl):
        bits.append(f"เงื่อนไข: {describe(fl)}")
    if plan["sort"]:
        bits.append(f"เรียงตาม {plan['sort']['by']} ({'มากไปน้อย' if plan['sort']['order'] == 'desc' else 'น้อยไปมาก'})")
    if plan["usage"]:
        bits.append(f"การใช้งาน: {plan['usage']}")
    if plan["limit"]:
        bits.append(f"จำนวนที่ขอ: {plan['limit']}")
    return " | ".join(bits)


def build_answer_prompt(plan: dict, picked: dict, question: str, history: list) -> str:
    items, fl, intent = picked["items"], plan["filters"], plan["intent"]
    lines = []
    for i, lp in enumerate(items, 1):
        note = _price_note(lp, fl)
        lines.append(f"{i}. {lp['doc']}" + (f" | {note}" if note else ""))
    diff = ""
    if intent == "compare" and len(items) == 2:
        d = abs(items[0]["price"] - items[1]["price"])
        diff = f" และบอกส่วนต่างราคา {d:,} บาท (ระบบคำนวณแล้ว)"
    n = plan["limit"] or 1
    task = TASKS[intent].format(n=len(items) if intent == "list" else n, diff=diff,
                                list_line=_list_line(plan, len(items)))
    notes = list(picked["notes"])
    scoped = describe(fl) or plan["models"] or picked["source"] in ("named", "overview")
    if scoped:
        head = f"{len(items)} รุ่น จากที่ตรงเงื่อนไข {picked['total']} รุ่น"
    else:
        # ไม่ได้กรองอะไรเลย: ถ้าลูกค้าถามเงื่อนไขที่ตัวจับไม่รู้จัก ห้าม AI อ้างว่ารายการนี้ตรงเงื่อนไข
        head = f"{len(items)} รุ่น จากทั้งร้าน {picked['total']} รุ่น ไม่ได้กรองตามสเปก"
        if intent == "list":
            notes.append("ระบบไม่ได้กรองตามสเปก ถ้าคำถามของลูกค้ามีเงื่อนไขที่ไม่อยู่ใน 'ระบบเข้าใจว่า' "
                         "ห้ามบอกว่ารุ่นในรายการตรงเงื่อนไขนั้น ให้บอกตรง ๆ ว่ายังคัดตามเรื่องนั้นไม่ได้")
    notes = "\n".join(f"- {x}" for x in notes) or "-"
    return (f"บทสนทนาก่อนหน้า:\n{_history_text(history)}\n\n"
            f"คำถามลูกค้า: {question}\n"
            f"ระบบเข้าใจว่า: {_understood(plan)}\n"
            f"ข้อมูลร้าน: {store_stats()}\n"
            f"หมายเหตุจากระบบ:\n{notes}\n\n"
            f"รายการที่ระบบคัดมา ({head} เรียงตามลำดับที่ควรนำเสนอ):\n"
            + ("\n".join(lines) or "(ไม่มี)") +
            f"\n\nงานของคุณ: {task}")


# ---- 7.3 เก็บกวาดข้อความที่ AI พิมพ์ (ตัด <think>, ครับครับ) ----
_THINK_OPEN = "<think>"
_RE_DOUBLE_KRAB = re.compile(r"ครับ(?:[ \t]*ครับ)+")


def strip_think(text: str) -> str:
    """ตัดบล็อก <think>...</think> ทิ้ง เหลือเฉพาะคำตอบจริง (โมเดลบางตัวเป็น reasoning model)"""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # เผื่อกรณีแท็กเปิดหายไป เหลือแต่แท็กปิด
    cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL)
    # เผื่อกรณี token หมดกลางคันจนแท็กปิดไม่ครบ — ถ้าไม่ตัด ผู้ใช้จะเห็นความคิดดิบของโมเดล
    cleaned = re.sub(r"<think>.*$", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


def visible_answer(raw: str) -> str:
    """ส่วนของคำตอบที่โชว์ระหว่างสตรีมได้ — ตัด <think> แบบเดียวกับคำตอบสุดท้าย (บล็อกที่ยังคิดไม่จบ
    ก็ตัด) และกั๊กท้ายที่อาจเป็นแท็กเปิดที่มายังไม่ครบ เช่น "<thi" ไม่งั้นแท็กจะโผล่ให้เห็นแวบหนึ่ง"""
    text = strip_think(raw)
    for n in range(len(_THINK_OPEN) - 1, 0, -1):
        if text.endswith(_THINK_OPEN[:n]):
            return text[:-n].rstrip()
    return text


def tidy(text: str) -> str:
    """เก็บกวาดคำตอบ: "ครับครับ" -> "ครับ" """
    return _RE_DOUBLE_KRAB.sub("ครับ", text)


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


# ---- 7.4 คัดรุ่น + ตอบ ----
def respond_stream(plan: dict, question: str, history: list, provider: dict, trace: dict | None = None,
                   result: dict | None = None):
    """ทยอยส่งคำตอบ: yield {"type": "delta" | "replace", "text"} ระหว่างที่ AI พิมพ์
    แล้วใส่คำตอบสุดท้ายไว้ที่ result["answer"]"""
    result = {} if result is None else result
    intent = plan["intent"]
    canned = unsupported_reply(plan)
    if intent in TEMPLATED or canned:
        thing = "รุ่นนั้น"
        if intent == "not_in_store":
            words = [w for w in OUTSIDE_BRANDS if w in question.lower()]
            if words:
                thing = f" {BRAND_CASE.get(words[0], words[0].capitalize())} "
        text = canned or TEMPLATES[intent].format(thing=thing)
        if trace is not None:
            trace["llm"] = {"provider": "ข้อความสำเร็จรูป (ไม่เรียก AI รอบสอง)",
                            "model": "unsupported" if canned else intent,
                            "temperature": 0, "max_tokens": 0, "ms": 0, "usage": None,
                            "finish_reason": "stop", "thinking": "", "raw_chars": len(text)}
        result["answer"] = text
        return

    if intent == "overview":
        cat = catalog()
        ends = [min(cat, key=lambda x: x["price"]), max(cat, key=lambda x: x["price"])]
        picked = {"items": ends, "total": len(cat), "missing": [], "source": "overview",
                  "notes": ["รายการคือรุ่นที่ถูกที่สุดและแพงที่สุดในร้าน — ใช้เมื่อคำถามเกี่ยวกับราคาเริ่มต้นหรือราคาสูงสุด"]}
        if trace is not None:
            trace["context"] = [{"source": "overview", "kind": "spec", "text": lp["doc"]} for lp in ends]
    else:
        picked = select(plan, question, trace)
    lead = unsupported_lead(plan)
    if lead:
        picked["notes"].append(f"ลูกค้าถามเรื่อง{_pad(_thai_join(plan['unsupported']))}ด้วย แต่ระบบยังคัดตามเรื่องนี้ไม่ได้ "
                               "รายการคัดตามเงื่อนไขอื่นเท่านั้น ห้ามสรุปว่ารุ่นในรายการตรงหรือไม่ตรงเรื่องนี้ "
                               "ระบบแจ้งลูกค้าไว้ต้นคำตอบแล้ว ไม่ต้องพูดซ้ำ")
    prompt = build_answer_prompt(plan, picked, question, history)
    max_tokens = 6000 if intent == "list" else 3000
    meta, raw, shown = {}, "", ""
    if lead:
        yield {"type": "delta", "text": lead}
        shown = lead
    for piece in complete_stream(provider, [{"role": "system", "content": ANSWER_RULES},
                                            {"role": "user", "content": prompt}], max_tokens=max_tokens, meta=meta):
        raw += piece
        text = lead + tidy(visible_answer(raw))
        if text == shown:
            continue
        # ตัด <think> อาจทำให้ข้อความที่โชว์ไปแล้วหายไป -> ส่งทั้งก้อนแทน
        yield ({"type": "delta", "text": text[len(shown):]} if text.startswith(shown)
               else {"type": "replace", "text": text})
        shown = text
    if trace is not None:
        trace["prompt"] = ANSWER_RULES + "\n\n" + prompt
        data = {"choices": [{"message": {"reasoning": meta.get("thinking") or None},
                             "finish_reason": meta.get("finish_reason")}], "usage": meta.get("usage")}
        trace["llm"] = {"provider": provider["label"], "model": provider["model"], "temperature": 0.1,
                        "max_tokens": max_tokens, **llm_trace(data, raw, meta.get("seconds", 0))}
    result["answer"] = lead + (tidy(strip_think(raw)) or EMPTY_ANSWER)


def respond(plan: dict, question: str, history: list, provider: dict, trace: dict | None = None) -> str:
    """ตอบแบบรอทั้งก้อน (/chat) — ใช้ขั้นตอนเดียวกับแบบสตรีม"""
    result = {}
    for _ in respond_stream(plan, question, history, provider, trace, result):
        pass
    return result["answer"]


# ═════════════════════════ 8. จำบทสนทนา (session cookie) ═════════════════════════
# session ของ Flask อยู่ใน cookie ซึ่งต้องส่งไปพร้อม header — แต่กว่า /chat/stream จะรู้คำตอบครบ
# header ถูกส่งไปนานแล้ว จึงแนบ "commit token" ไปกับ event done ให้หน้าเว็บยิงกลับมาที่ /chat/commit
# token เซ็นด้วย secret_key: ผู้ใช้แก้เนื้อหาคำตอบที่จะเข้าไปอยู่ใน context ของ LLM รอบหน้าไม่ได้
COMMIT_MAX_AGE = 600   # วินาที


def memory(plan: dict) -> dict:
    """เงื่อนไขที่จำไว้ใช้ตีความข้อความถัดไป (เก็บใน session cookie — ต้องเล็ก)"""
    if plan["intent"] in TEMPLATED or plan["intent"] == "overview":
        return {}
    return {"intent": plan["intent"], "filters": {k: v for k, v in plan["filters"].items() if v},
            "models": plan["models"][:4], "usage": plan["usage"]}


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


# ═════════════════════════ 9. endpoint ของหน้าเว็บ ═════════════════════════
# /chat/stream ทยอยส่ง event เป็น Server-Sent Events แต่ละ event คือบรรทัด "data: {json}" ตามด้วยบรรทัดว่าง:
#   {"type": "status",  "text": ...}   ระบบกำลังทำอะไร
#   {"type": "delta",   "text": ...}   ข้อความคำตอบที่เพิ่งพิมพ์ต่อท้าย
#   {"type": "replace", "text": ...}   ทิ้งข้อความที่โชว์ไปแล้วแล้วใช้ค่านี้แทน (เจอ <think>)
#   {"type": "done",    "message": ..., "meta": ..., "commit": ...}   คำตอบสุดท้ายที่ต้องใช้จริง
#   {"type": "error",   "error": ...}
STATUS_WRITING = "กำลังเรียบเรียงคำตอบ…"


def sse(event: dict) -> str:
    # json.dumps escape การขึ้นบรรทัดใหม่ในข้อความให้แล้ว หนึ่ง event จึงเป็นบรรทัดเดียวเสมอ
    return f"data: {json.dumps(event, ensure_ascii=False)}\n\n"


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


@app.route("/chat/stream", methods=["POST"])
def chat_stream():
    """endpoint หลักของหน้าเว็บ: ตีความ -> คัดรุ่น -> ตอบ แบบ Server-Sent Events
    บอกสถานะระหว่างทาง แล้วทยอยส่งคำตอบตามที่ AI พิมพ์

    error ที่รู้ก่อนเริ่มตอบ (JSON พัง, ถามถี่เกิน, ตีความไม่ได้ ...) ตอบเป็น JSON + status code แบบเดียวกับ /chat
    ส่วน error ระหว่างสตรีมส่งเป็น event {"type": "error"} เพราะ status 200 ถูกส่งไปแล้ว
    ประวัติแชทบันทึกผ่าน /chat/commit (ดู commit_token)
    """
    job, error = _chat_request()
    if error:
        return error
    user_msg, provider, history = job["message"], job["provider"], job["history"]

    started = time.time()
    # ขอดูกระบวนการคิด -> แนบไปกับ event สุดท้าย (done หรือ error) ทั้งก้อน
    trace = new_trace(job)
    debug_part = {"debug": trace} if trace is not None else {}

    # ตีความคำถามก่อนเริ่มสตรีม — เงื่อนไขที่ได้ต้องเขียนลง session ซึ่งส่งไปกับ header
    try:
        plan = understand(user_msg, history, session.get("last_query"), provider, trace)
    except LLMError as e:
        return jsonify({"error": str(e), **debug_part}), 502
    session["last_query"] = memory(plan)

    # ทุกอย่างที่ต้องใช้จาก request/session อ่านมาครบแล้ว — generator นี้ทำงานหลังส่ง header
    # จึงไม่แตะ session อีก (เขียนไปก็ไม่ถูกบันทึก)
    def events():
        yield sse({"type": "status", "text": STATUS_WRITING})
        result = {}
        try:
            # ทยอยส่งคำตอบระหว่างที่ AI พิมพ์ คำตอบยาวก็เริ่มเห็นข้อความได้ทันที
            for ev in respond_stream(plan, user_msg, history, provider, trace, result):
                yield sse(ev)
            answer = result["answer"]
        except LLMError as e:
            yield sse({"type": "error", "error": str(e), **debug_part})
            return
        except Exception as e:
            yield sse({"type": "error", "error": f"RAG error: {e}", **debug_part})
            return
        yield sse({"type": "done", "message": answer, "meta": chat_meta(job, started),
                   "commit": commit_token(user_msg, answer), **debug_part})

    return Response(events(), mimetype="text/event-stream", headers={
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",   # บอก nginx ไม่ให้กั๊ก response ไว้รอจนครบ
    })


@app.route("/chat", methods=["POST"])
def chat():
    # เหมือน /chat/stream แต่ตอบเป็น JSON ก้อนเดียวตอนจบ — ไว้ให้สคริปต์ / ชุดทดสอบเรียกง่าย ๆ
    job, error = _chat_request()
    if error:
        return error
    user_msg, provider, history = job["message"], job["provider"], job["history"]

    # จับเวลาตั้งแต่ก่อนตีความ — เวลาใน meta ต้องตรงกับที่ผู้ใช้รอจริง (ตีความ + คัดรุ่น + ตอบ)
    started = time.time()
    trace = new_trace(job)
    debug_part = {"debug": trace} if trace is not None else {}
    try:
        plan = understand(user_msg, history, session.get("last_query"), provider, trace)
        session["last_query"] = memory(plan)
        answer = respond(plan, user_msg, history, provider, trace)
    except LLMError as e:
        return jsonify({"error": str(e), **debug_part}), 502
    except Exception as e:
        return jsonify({"error": f"RAG error: {e}", **debug_part}), 500
    save_turn(history, user_msg, answer)
    return jsonify({"message": answer, "meta": chat_meta(job, started), **debug_part})


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


@app.route("/new_chat", methods=["POST"])
def new_chat():
    # ล้างประวัติบทสนทนาและเงื่อนไขที่จำไว้ เพื่อเริ่มสนทนาใหม่
    session["history"] = []
    session.pop("last_query", None)
    return jsonify({"ok": True})


@app.route("/restore", methods=["POST"])
def restore():
    """โหลดบทสนทนาเก่าที่หน้าเว็บเก็บไว้ใน localStorage กลับเข้า session ฝั่งเซิร์ฟเวอร์
    เพื่อให้กดดูแชทเก่าจาก sidebar แล้วคุยต่อได้จริง

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
    # เงื่อนไขที่จำไว้เป็นของแชทที่เปิดอยู่ก่อนหน้า — ตัวตีความจะอ่านเงื่อนไขของแชทนี้จากประวัติเอง
    session.pop("last_query", None)
    return jsonify({"ok": True, "restored": len(clean)})


# เริ่มโหลดโมเดลล่วงหน้าตอน gunicorn import แอป — วางท้ายไฟล์ให้ทุกฟังก์ชัน/ค่าตั้งพร้อมก่อน thread เริ่ม
# (ถ้าวันหนึ่งใช้ gunicorn --preload ต้องย้ายไปเรียกใน hook post_fork แทน ไม่งั้น thread อยู่แต่ใน master)
start_warmup()


if __name__ == "__main__":
    # debug=True เปิด Werkzeug debugger ซึ่งรันโค้ดจากเบราว์เซอร์ได้ — ห้ามติดไปถึง production
    app.run(debug=os.getenv("FLASK_DEBUG", "false").lower() == "true", port=5000)
