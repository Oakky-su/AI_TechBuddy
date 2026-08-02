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
TOP_K = 5  # จำนวนเอกสารที่ดึงมาจาก vector search

# ---- ตั้งค่า ThaiLLM / OpenThaiGPT (อ่านจาก .env ได้) ----
THAILLM_API_KEY = os.getenv("THAILLM_API_KEY", "")
BASE_URL = os.getenv("THAILLM_BASE_URL", "http://thaillm.or.th/api/v1")
LLM_MODEL = os.getenv("THAILLM_MODEL", "openthaigpt-thaillm-8b-instruct-v7.2")
# ปิดโหมดคิด (reasoning) -> เร็ว/ถูกกว่า | เปิด -> ใช้เหตุผลก่อนตอบ (ตาม notebook: เปิดโหมดคิด)
DISABLE_THINKING = os.getenv("DISABLE_THINKING", "false").lower() == "true"

# cache โมเดลและ collection ไว้ใน memory ป้องกันการโหลดซ้ำทุก request
_embedder: SentenceTransformer = None
_collection = None


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

รูปแบบส่วนสรุปให้เขียนประมาณนี้:

สรุปเบนช์มาร์ก: คะแนน CPU [ตัวเลข] ถือว่าอยู่ในระดับ [ระดับ] ซึ่ง [อธิบายว่ารองรับงานแบบใดได้] ส่วนคะแนน GPU [ตัวเลข] อยู่ในระดับ [ระดับ] [อธิบายว่าเล่นเกม/ทำงานกราฟิกได้แค่ไหน]"""


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
_BUDGET_RE = re.compile(
    _BUDGET_CUE
    + r"\D{0,12}?"                                    # คำคั่น เช่น "ไม่เกิน", "ราวๆ"
    + r"(\d[\d,]*(?:\.\d+)?|" + "|".join(_THAI_NUM_WORD) + r")"  # ตัวเลข หรือคำไทย
    + r"\s*(หมื่น|พัน|k|K)?"                           # หน่วยคูณ (ถ้ามี)
)


def extract_budget(text: str) -> int | None:
    """อ่านงบสูงสุดจากข้อความ เช่น 'งบไม่เกิน 20000' / 'งบ 2 หมื่น' / 'งบ 20k' -> 20000
    คืน None ถ้าไม่พบงบ (จะได้ไม่กรองราคามั่วตอนผู้ใช้ไม่ได้พูดถึงงบ)"""
    m = _BUDGET_RE.search(text)
    if not m:
        return None

    raw, unit = m.group(1), m.group(2)
    value = float(_THAI_NUM_WORD[raw]) if raw in _THAI_NUM_WORD else float(raw.replace(",", ""))

    if unit in ("หมื่น",):
        value *= 10_000
    elif unit in ("พัน", "k", "K"):
        value *= 1_000

    # กันเคสอ่านผิด เช่น "งบ 15 นิ้ว" -> 15 (ไม่ใช่ราคา) หรือ "ไม่เกิน 2 ปี"
    if value < 1_000:
        return None
    return int(value)


def retrieve(question: str, k: int = TOP_K, budget: int | None = None) -> tuple[list[str], bool]:
    """ค้นเอกสารที่เกี่ยวข้องที่สุด k อันดับ ถ้ามีงบจะกรองเฉพาะรุ่นที่ราคาไม่เกินงบ
    คืน (เอกสาร, over_budget) — over_budget=True แปลว่าไม่มีรุ่นไหนอยู่ในงบเลย
    """
    q_emb = get_embedder().encode([question], normalize_embeddings=True).tolist()
    collection = get_collection()

    if budget is not None:
        # metadata "price" ถูกเก็บไว้ตอน build_index.py แล้ว (ทั้งเอกสารสเปกและรีวิว)
        res = collection.query(
            query_embeddings=q_emb, n_results=k, where={"price": {"$lte": budget}}
        )
        docs = res["documents"][0]
        # เอกสารสรุปภาพรวมร้านมี price=0 เลยผ่านฟิลเตอร์เสมอ — ไม่นับว่าเป็นรุ่นในงบ
        if any(not d.startswith("ร้านมีโน้ตบุ๊ค") for d in docs):
            return docs, False
        # ไม่มีรุ่นไหนในงบเลย -> ค้นใหม่แบบไม่กรอง เพื่อให้ AI บอกได้ว่ารุ่นถูกสุดคือเท่าไหร่
        res = collection.query(query_embeddings=q_emb, n_results=k)
        return res["documents"][0], True

    res = collection.query(query_embeddings=q_emb, n_results=k)
    return res["documents"][0], False


def strip_think(text: str) -> str:
    """ตัดบล็อก <think>...</think> ทิ้ง เหลือเฉพาะคำตอบจริง (โมเดลเป็น reasoning model)"""
    cleaned = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    # เผื่อกรณี token หมดกลางคันจน tag ปิดไม่ครบ
    cleaned = re.sub(r"^.*?</think>", "", cleaned, flags=re.DOTALL)
    return cleaned.strip()


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
        recent = history[-4:]  # เอาประวัติ 4 รอบล่าสุดมาให้ AI จำบริบท
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
        )

    suffix = " /no_think" if DISABLE_THINKING else ""
    user_msg = f"""{SYSTEM_PROMPT}

ประวัติบทสนทนา:
{history_text if history_text else "(เริ่มต้นใหม่)"}
{budget_note}
ข้อมูลสินค้า:
{context_text}

คำถามลูกค้า: {question}{suffix}"""

    resp = requests.post(
        f"{BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {THAILLM_API_KEY}",
            "Content-Type": "application/json",
        },
        json={
            "model": LLM_MODEL,
            "messages": [{"role": "user", "content": user_msg}],
            "temperature": 0.2,
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
    # หน้าหลัก — เริ่มต้น session history ถ้ายังไม่มี
    session.setdefault("history", [])
    return render_template("index.html")


@app.route("/chat", methods=["POST"])
def chat():
    # endpoint หลัก: รับข้อความ -> ค้นหา (retrieve) -> แต่งคำตอบ (generate)
    data = request.get_json()
    user_msg = (data.get("message") or "").strip()
    if not user_msg:
        return jsonify({"error": "empty message"}), 400

    history = session.get("history", [])

    # หางบจากข้อความล่าสุดก่อน ถ้าไม่มีค่อยย้อนดูที่ลูกค้าเคยบอกไว้ในบทสนทนา
    # (เช่น รอบก่อนบอก "งบ 20000" รอบนี้ถามแค่ "มีรุ่นอื่นอีกไหม" ก็ยังต้องอยู่ในงบเดิม)
    budget = extract_budget(user_msg)
    if budget is None:
        for h in reversed(history):
            if h["role"] == "user":
                budget = extract_budget(h["content"])
                if budget is not None:
                    break

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

    # บันทึกประวัติบทสนทนาใน session (เก็บไว้ 20 รอบล่าสุด)
    history.append({"role": "user", "content": user_msg})
    history.append({"role": "assistant", "content": answer})
    session["history"] = history[-20:]

    return jsonify({"message": answer})


@app.route("/new_chat", methods=["POST"])
def new_chat():
    # ล้างประวัติบทสนทนาใน session เพื่อเริ่มสนทนาใหม่
    session["history"] = []
    return jsonify({"ok": True})


if __name__ == "__main__":
    if not CHROMA_DIR.exists():
        print("ERROR: ไม่พบ chroma_db — รัน `python build_index.py` ก่อนนะ")
    else:
        app.run(debug=True, port=5000)
