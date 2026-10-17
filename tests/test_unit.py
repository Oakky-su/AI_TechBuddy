# ชุดทดสอบ logic ล้วน (ไม่ต้องใช้โมเดล/API) — รัน: python tests/test_unit.py
# ครอบคลุม: extract_budget(), strip_think(), laptops_to_documents(), ความสมบูรณ์ของ laptops.json,
#          รายการ AI (llm_providers) และ tool ของ Pi Agent ที่ไม่ต้องใช้ vector DB
import json
import re
import sys
import types
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# stub chromadb — ไม่ได้ติดตั้งบนเครื่อง host (อยู่ใน Docker image) และเทสต์ชุดนี้ไม่ได้ใช้
if "chromadb" not in sys.modules:
    stub = types.ModuleType("chromadb")
    stub.PersistentClient = lambda *a, **k: None
    stub.IS_STUB = True
    sys.modules["chromadb"] = stub

import os  # noqa: E402

# ไม่ต้องโหลดโมเดล embedding ล่วงหน้าตอน import แอป (test_warmup เรียกเองโดยใช้ตัวปลอม)
os.environ["WARMUP_EMBEDDINGS"] = "false"

import app  # noqa: E402
import llm_providers as llm  # noqa: E402
from app import extract_budget, strip_think  # noqa: E402
from build_index import laptops_to_documents  # noqa: E402

PASS, FAIL = [], []


def check(group, case, expected, actual, note=""):
    ok = expected == actual
    (PASS if ok else FAIL).append((group, case, expected, actual, note))
    return ok


# ─────────────────────────────────────────────────────────────
# 1. extract_budget() — อ่านงบจากคำถามภาษาไทย
# ─────────────────────────────────────────────────────────────
BUDGET_CASES = [
    # (คำถาม, ค่าที่ควรได้, หมายเหตุ)
    # --- รูปแบบตัวเลขพื้นฐาน ---
    ("งบ 20000", 20000, "ตัวเลขเปล่า"),
    ("งบ 20,000", 20000, "มี comma"),
    ("งบ 12,990 บาท", 12990, "comma + หน่วยบาท"),
    ("งบ 100000", 100000, "หกหลัก"),
    ("งบ  30000", 30000, "เว้นวรรคหลายตัว"),
    # --- หน่วยคูณ ---
    ("งบ 2 หมื่น", 20000, "หน่วย หมื่น"),
    ("งบ 5 พัน", 5000, "หน่วย พัน"),
    ("งบ 20k", 20000, "หน่วย k เล็ก"),
    ("งบ 50K", 50000, "หน่วย K ใหญ่"),
    ("งบ 1.5 หมื่น", 15000, "ทศนิยม + หมื่น"),
    # --- เลขไทยเป็นคำ ---
    ("งบสองหมื่น", 20000, "คำไทยติดกัน"),
    ("ไม่เกินสามหมื่น", 30000, "คำไทย + cue อื่น"),
    ("งบ สี่ หมื่น", 40000, "คำไทยเว้นวรรค"),
    # --- cue หลายแบบ ---
    ("งบประมาณ 35000 บาท", 35000, "cue งบประมาณ"),
    ("ราคาไม่เกิน 30000", 30000, "cue ราคาไม่เกิน"),
    ("ไม่เกิน 45000", 45000, "cue ไม่เกิน"),
    ("ต่ำกว่า 15000", 15000, "cue ต่ำกว่า"),
    ("ในราคา 25000", 25000, "cue ในราคา"),
    ("ราคาประมาณ 40000", 40000, "cue ราคาประมาณ"),
    ("budget 50000", 50000, "cue อังกฤษ"),
    # --- ต้องไม่จับ (ไม่มี cue / ไม่ใช่ราคา) ---
    ("อยากได้โน้ตบุ๊คเล่นเกม", None, "ไม่มีงบเลย"),
    ("RTX 5050 ดีไหม", None, "ตัวเลขรุ่นการ์ดจอ ไม่มี cue"),
    ("มี Core i7 13620H ไหม", None, "ตัวเลขรุ่น CPU"),
    ("จอไม่เกิน 15 นิ้ว", None, "15 ไม่ใช่ราคา (<1000)"),
    ("ใช้ได้ไม่เกิน 2 ปี", None, "2 ไม่ใช่ราคา"),
    ("ไม่เกิน 999", None, "ต่ำกว่าเกณฑ์ 1000"),
    ("งบ 0", None, "ศูนย์"),
    ("ไม่เกิน 1000", 1000, "ขอบเกณฑ์พอดี"),
    # --- เคสประโยคจริงที่ซับซ้อน ---
    ("โน้ตบุ๊คเล่นเกม งบ 35,000 ครับ", 35000, "งบอยู่กลางประโยค"),
    ("อยากได้โน้ตบุ๊คจอไม่เกิน 15 นิ้ว งบ 25000", 25000, "มีตัวเลขหลอกมาก่อนงบจริง"),
    ("ขอรุ่นที่ราคาไม่เกิน 30000 แต่ถ้าดีจริงจ่ายได้ถึง 35000", 30000, "เอาตัวแรก (เพดานที่ตั้งใจ)"),
    ("งบที่ตั้งไว้คือประมาณ 30000", 30000, "คำคั่นยาวเกิน 12 ตัวอักษร"),
    ("งบหมื่นห้า", 15000, "ภาษาพูด หมื่นห้า"),
    ("งบสามหมื่นห้า", 35000, "ภาษาพูด สามหมื่นห้า"),
    ("งบ 20000-25000", 25000, "ช่วงราคา ควรเอาเพดานบน"),
]


def test_budget():
    for q, expected, note in BUDGET_CASES:
        check("extract_budget", q, expected, extract_budget(q), note)


# ─────────────────────────────────────────────────────────────
# 2. strip_think() — ตัดบล็อก reasoning ของโมเดลออก
# ─────────────────────────────────────────────────────────────
THINK_CASES = [
    ("<think>คิดอยู่</think>คำตอบจริง", "คำตอบจริง", "บล็อกครบคู่"),
    ("<think>บรรทัด1\nบรรทัด2</think>\n\nคำตอบ", "คำตอบ", "บล็อกหลายบรรทัด"),
    ("คิดค้างมาจากก่อนหน้า</think>คำตอบ", "คำตอบ", "มีแต่แท็กปิด (token หมดตอนเริ่ม)"),
    ("<think>a</think>ตอบ1<think>b</think>ตอบ2", "ตอบ1ตอบ2", "สองบล็อก"),
    ("คำตอบธรรมดาไม่มีแท็ก", "คำตอบธรรมดาไม่มีแท็ก", "ไม่มีแท็กเลย"),
    ("", "", "สตริงว่าง"),
    ("<think>คิดยาวมากแล้ว token หมดก่อนปิดแท็ก", "", "แท็กเปิดแต่ไม่ปิด — ต้องไม่หลุดให้ผู้ใช้เห็น"),
]


def test_strip_think():
    for raw, expected, note in THINK_CASES:
        check("strip_think", repr(raw[:40]), expected, strip_think(raw), note)


# ─────────────────────────────────────────────────────────────
# 3. laptops_to_documents() — การแปลงข้อมูลเป็นเอกสารสำหรับ embed
# ─────────────────────────────────────────────────────────────
def test_documents():
    laptops = json.loads((BASE / "laptops.json").read_text(encoding="utf-8"))
    docs, metas, ids = laptops_to_documents(laptops)

    n_spec = sum(m["type"] == "spec" for m in metas)
    n_review = sum(m["type"] == "review" for m in metas)
    n_summary = sum(m["type"] == "summary" for m in metas)
    total_reviews = sum(len(lp.get("reviews", [])) for lp in laptops)

    check("documents", "3 ลิสต์ยาวเท่ากัน", True, len(docs) == len(metas) == len(ids), "")
    check("documents", "จำนวนเอกสารสเปก = จำนวนรุ่น", len(laptops), n_spec, "")
    check("documents", "จำนวนเอกสารรีวิว = รีวิวทั้งหมด", total_reviews, n_review, "")
    check("documents", "มีเอกสารสรุป 1 อัน", 1, n_summary, "")
    check("documents", "id ไม่ซ้ำ (ซ้ำ=Chroma ทับข้อมูล)", len(ids), len(set(ids)), "")
    check("documents", "ทุก meta มี price เป็น int", True,
          all(isinstance(m["price"], int) for m in metas), "ใช้กรองงบ")
    check("documents", "ไม่มีเอกสารว่าง", True, all(d.strip() for d in docs), "")

    # เอกสารสเปกต้องมีราคาอยู่ในข้อความ (LLM ต้องอ่านราคาได้จาก context)
    spec_docs = [d for d, m in zip(docs, metas) if m["type"] == "spec"]
    check("documents", "เอกสารสเปกมีคำว่า 'บาท'", len(spec_docs),
          sum("บาท" in d for d in spec_docs), "")
    # เอกสารรีวิวควรมีราคาด้วยไหม — ถ้า context ได้แต่รีวิว LLM จะไม่รู้ราคา
    review_docs = [d for d, m in zip(docs, metas) if m["type"] == "review"]
    check("documents", "เอกสารรีวิวมีราคาในข้อความ", len(review_docs),
          sum("บาท" in d for d in review_docs), "ถ้า 0 = LLM อ่านราคาจากรีวิวไม่ได้")
    # เอกสารสรุปต้องขึ้นต้นด้วยคำที่ retrieve() ใช้เช็ค
    summary_doc = [d for d, m in zip(docs, metas) if m["type"] == "summary"][0]
    # app.is_summary_doc() ใช้คำขึ้นต้นนี้แยกเอกสารสรุปออกจากเอกสารสเปก
    check("documents", "เอกสารสรุปขึ้นต้น 'ภาพรวมข้อมูลโน้ตบุ๊ค'", True,
          summary_doc.startswith("ภาพรวมข้อมูลโน้ตบุ๊ค"), "is_summary_doc() พึ่งพา prefix นี้")


# ─────────────────────────────────────────────────────────────
# 4. ความสมบูรณ์ของ laptops.json
# ─────────────────────────────────────────────────────────────
def test_data():
    laptops = json.loads((BASE / "laptops.json").read_text(encoding="utf-8"))
    nos = [lp.get("ลำดับ") for lp in laptops]
    names = [lp.get("ชื่อสินค้า") for lp in laptops]

    check("data", "ทุกรุ่นมี ลำดับ", 0, sum(1 for n in nos if n is None), "")
    check("data", "ลำดับ ไม่ซ้ำ", len(nos), len(set(nos)), "ซ้ำ = id ชนกันตอน build index")
    check("data", "ชื่อสินค้า ไม่ซ้ำ", len(names), len(set(names)), "")
    check("data", "ราคาเป็น int ทุกตัว", len(laptops),
          sum(isinstance(lp.get("ราคา"), int) for lp in laptops), "str = Chroma กรองงบไม่ได้")
    check("data", "ราคา > 0 ทุกตัว", len(laptops),
          sum(1 for lp in laptops if isinstance(lp.get("ราคา"), int) and lp["ราคา"] > 0), "")
    check("data", "ทุกรุ่นมีรีวิวอย่างน้อย 1", len(laptops),
          sum(1 for lp in laptops if lp.get("reviews")), "")
    for k in ["CPU", "GPU", "RAM", "DISPLAY", "STORAGE", "OS"]:
        check("data", f"ทุกรุ่นมี spec.{k}", len(laptops),
              sum(1 for lp in laptops if lp.get("spec", {}).get(k)), "")
    for k in ["CPU Score", "GPU Score"]:
        check("data", f"ทุกรุ่นมี benchmark['{k}'] เป็น int", len(laptops),
              sum(1 for lp in laptops if isinstance(lp.get("benchmark", {}).get(k), int)), "")


# ─────────────────────────────────────────────────────────────
# 5. llm_providers — รายการ AI ที่สลับได้จากหน้าเว็บ
# ─────────────────────────────────────────────────────────────
KEY_VARS = ("BMS_LLM_API_KEY", "THAILLM_API_KEY", "ANTHROPIC_API_KEY", "OPENAI_API_KEY",
            "GEMINI_API_KEY", "BMS_LLM_BASE_URL", "DEFAULT_PROVIDER", "DEFAULT_ENGINE",
            "VISIBLE_PROVIDERS", "OLLAMA_API_KEY", "ALLOWED_ENGINES")


def test_providers():
    saved = {k: os.environ.pop(k, None) for k in KEY_VARS}
    try:
        by_id = {p["id"]: p for p in llm.load_providers()}
        check("providers", "Wave AI ใช้ได้โดยไม่ต้องมี key", True, by_id["wave-ai"]["available"], "")
        check("providers", "ThaiLLM ไม่มี key = ใช้ไม่ได้", False, by_id["thaillm"]["available"], "")
        check("providers", "Claude ไม่มี key = ใช้ไม่ได้", False, by_id["claude-sonnet"]["available"], "")
        check("providers", "pi-builtin ใช้ได้เฉพาะโหมด pi", ["pi"], by_id["claude-sonnet"]["engines"], "")
        check("providers", "openai-compatible ใช้ได้ทั้ง 2 โหมด", ["pi", "classic"],
              by_id["wave-ai"]["engines"], "")
        check("providers", "ไม่ตั้ง DEFAULT_PROVIDER -> ตัวแรกที่ใช้ได้", "wave-ai",
              llm.public_list()["default_provider"], "")

        _, _, err = llm.pick("thaillm", "classic")
        check("providers", "เลือกตัวที่ยังไม่มี key -> error", True, bool(err) and "THAILLM_API_KEY" in err, "")
        _, _, err = llm.pick("no-such-ai", "pi")
        check("providers", "id ที่ไม่มี -> error", True, bool(err), "")

        os.environ["ANTHROPIC_API_KEY"] = "sk-test-secret"
        _, _, err = llm.pick("claude-sonnet", "classic")
        check("providers", "pi-builtin + classic -> error", True, bool(err) and "Pi Agent" in err, "")
        provider, engine, err = llm.pick("claude-sonnet", "pi")
        check("providers", "pi-builtin + pi -> ผ่าน", (None, "pi"), (err, engine), "")
        check("providers", "/providers ไม่มี API key หลุด", False,
              "sk-test-secret" in json.dumps(llm.public_list()), "สำคัญ: รายการนี้ส่งถึงเบราว์เซอร์")
        check("providers", "config ที่ส่ง pi-agent มี key", "sk-test-secret",
              llm.agent_config(provider).get("api_key"), "")

        os.environ["BMS_LLM_BASE_URL"] = "http://example.test/v1/"
        bms = next(p for p in llm.load_providers() if p["id"] == "wave-ai")
        check("providers", "env ทับ base_url + ตัด / ท้าย", "http://example.test/v1", bms["base_url"], "")

        os.environ["VISIBLE_PROVIDERS"] = "wave-ai"
        listed = [p["id"] for p in llm.public_list()["providers"]]
        check("providers", "VISIBLE_PROVIDERS ซ่อนตัวอื่นจากเมนู", ["wave-ai"], listed, "")
        _, _, err = llm.pick("claude-sonnet", "pi")
        check("providers", "ตัวที่ซ่อนเรียกผ่าน /chat ไม่ได้", True, bool(err) and "ไม่ได้เปิดให้ใช้" in err, "")
        os.environ.pop("VISIBLE_PROVIDERS")

        os.environ["ALLOWED_ENGINES"] = "classic"
        lst = llm.public_list()
        wave = next(p for p in lst["providers"] if p["id"] == "wave-ai")
        check("engines", "ALLOWED_ENGINES=classic -> เหลือโหมดเดียว", (["classic"], ["classic"], "classic"),
              (lst["engines"], wave["engines"], lst["default_engine"]), "")
        _, engine, err = llm.pick("wave-ai", "pi")
        check("engines", "ขอ pi ตอนปิดไว้ -> ใช้ classic แทนเงียบ ๆ", ("classic", None), (engine, err), "")
        os.environ["ANTHROPIC_API_KEY"] = "sk-test"
        claude = next(p for p in llm.load_providers() if p["id"] == "claude-sonnet")
        check("engines", "AI ที่ใช้ได้แค่ pi -> ใช้ไม่ได้ตอนปิด pi", False, claude["available"], "")
        os.environ["ALLOWED_ENGINES"] = "nonsense"
        check("engines", "ค่าแปลก -> เปิดทั้งสองโหมด", ["pi", "classic"], llm.allowed_engines(), "")
        os.environ.pop("ALLOWED_ENGINES")

        os.environ["DEFAULT_ENGINE"] = "classic"
        check("providers", "DEFAULT_ENGINE=classic", "classic", llm.default_engine(), "")
        os.environ["DEFAULT_ENGINE"] = "weird"
        check("providers", "DEFAULT_ENGINE แปลก -> pi", "pi", llm.default_engine(), "")
    finally:
        for k, v in saved.items():
            os.environ.pop(k, None)
            if v is not None:
                os.environ[k] = v


# ─────────────────────────────────────────────────────────────
# 6. tool ของ Pi Agent ที่ไม่ต้องใช้ vector DB
# ─────────────────────────────────────────────────────────────
def test_tools():
    laptops = json.loads((BASE / "laptops.json").read_text(encoding="utf-8"))
    app._spec_doc_by_name = {}   # ไม่มี Chroma บนเครื่อง host -> tool ใช้ชื่อรุ่นแทนเอกสารสเปก

    check("tools", "_int_arg รับ '25,000'", 25000, app._int_arg({"x": "25,000"}, "x"), "")
    check("tools", "_int_arg รับ 25000.0", 25000, app._int_arg({"x": 25000.0}, "x"), "")
    check("tools", "_int_arg ค่าว่าง = None", None, app._int_arg({"x": ""}, "x"), "")
    check("tools", "_ram_gb('16 GB DDR5')", 16, app._ram_gb("16 GB DDR5 5600 MHz"), "")
    check("tools", "_limit_arg เพดาน 15", 15, app._limit_arg({"limit": 99}, 8), "")

    # กรอง RTX 4060 งบ 40,000 ต้องได้ "ทุก" รุ่นที่ตรง (vector search ทำแบบนี้ไม่ได้)
    want = [lp for lp in laptops if "rtx 4060" in lp["spec"]["GPU"].lower() and lp["ราคา"] <= 40000]
    text = app.tool_filter_laptops({"gpu": "RTX 4060", "max_price": 40000, "limit": 15})
    if want:
        check("tools", "filter: นับจำนวนรุ่นถูก", True, text.startswith(f"พบ {len(want)} รุ่น"), text[:60])
        check("tools", "filter: มีทุกรุ่นที่ตรง", True,
              all(lp["ชื่อสินค้า"] in text for lp in want[:15]), "")
    else:
        check("tools", "filter: ไม่เจอ -> บอกตรง ๆ", True, text.startswith("ไม่พบรุ่น"), text[:60])

    cheapest = min(laptops, key=lambda lp: lp["ราคา"])["ชื่อสินค้า"]
    text = app.tool_filter_laptops({"sort_by": "price_asc", "limit": 1})
    check("tools", "filter: เรียงถูกสุดก่อน", True, cheapest in text, "")
    text = app.tool_filter_laptops({"max_price": 1})
    check("tools", "filter: ไม่มีรุ่นในงบ", True, text.startswith("ไม่พบรุ่น"), "")

    try:
        app.tool_get_laptop_details({"name": ""})
        check("tools", "details: ไม่ใส่ชื่อ -> ValueError", True, False, "")
    except ValueError:
        check("tools", "details: ไม่ใส่ชื่อ -> ValueError", True, True, "")
    text = app.tool_get_laptop_details({"name": "MacBook Air M9"})
    check("tools", "details: ชื่อที่ไม่มี -> ไม่พบ + เสนอชื่อใกล้เคียง", True, text.startswith("ไม่พบรุ่น"), "")

    # ระดับคะแนนต้องตรงเกณฑ์ใน SYSTEM_PROMPT รวมขอบช่วง
    for score, want in ((17_999, "เริ่มต้น"), (18_000, "กลาง"), (25_000, "กลาง-สูง"),
                        (32_000, "กลาง-สูง"), (32_001, "สูง"), (49_500, "สูง")):
        check("levels", f"CPU {score:,}", want, app.score_level(score, app.CPU_LEVELS), "")
    for score, want in ((3_999, "เริ่มต้น"), (6_500, "เริ่มต้น-กลาง"), (10_500, "กลาง-สูง"),
                        (14_000, "กลาง-สูง"), (14_500, "สูง")):
        check("levels", f"GPU {score:,}", want, app.score_level(score, app.GPU_LEVELS), "")
    spec = "X ราคา 30,990 บาท | ซีพียู (CPU): i5 | คะแนนเบนช์มาร์ก CPU 18776 / GPU 10500"
    check("levels", "with_levels ต่อท้ายเอกสารสเปก", True,
          app.with_levels(spec).endswith("CPU = กลาง / GPU = กลาง-สูง"), "")
    review = "รีวิวจากผู้ใช้รุ่น X ราคา 30,990 บาท (คะแนนเบนช์มาร์ก CPU 18776 / GPU 10500): ดี"
    check("levels", "with_levels ไม่แตะเอกสารรีวิว", review, app.with_levels(review), "")

    # ค้นตรงตามสเปก/ชื่อรุ่น (เสริม vector search ทั้งสองโหมด)
    check("exact", "gpu_terms รับ 'rtx4060' ติดกัน", ["rtx 4060"], app.gpu_terms("rtx4060 งบ 40k"), "")
    check("exact", "gpu_terms ไม่จับเลขรุ่นอื่น", [], app.gpu_terms("Ryzen 7 7735HS งบ 30000"), "")
    check("exact", "RTX 5070 ไม่นับ 5070 Ti", False, app.gpu_matches("NVIDIA GeForce RTX 5070 Ti", "rtx 5070"), "")
    check("exact", "RTX 5070 Ti ตรงตัว", True, app.gpu_matches("NVIDIA GeForce RTX 5070 Ti", "rtx 5070 ti"), "")
    legion5 = {lp["ชื่อสินค้า"] for lp in laptops if lp["ชื่อสินค้า"].startswith("Lenovo Legion 5 ")}
    victus16 = {lp["ชื่อสินค้า"] for lp in laptops if lp["ชื่อสินค้า"].startswith("HP Victus 16")}
    asked = set(app.asked_model_names("เทียบ Legion 5 กับ HP Victus 16 ให้หน่อย"))
    check("exact", "เอ่ย Legion 5 + Victus 16 -> ได้ครบทุกรุ่น", True, bool(legion5) and (legion5 | victus16) <= asked, "")
    check("exact", "ไม่ปน Legion 5a / Victus 15", set(), {n for n in asked if "Legion 5a" in n or "Victus 15" in n}, "")
    check("exact", "รุ่นที่ไม่มีในข้อมูล -> ไม่จับอะไร", [], app.asked_model_names("MacBook Air M4 ดีไหม"), "")
    rtx4060 = [lp for lp in laptops if "rtx 4060" in lp["spec"]["GPU"].lower() and lp["ราคา"] <= 40_000]
    notes, _ = app.exact_match("RTX 4060 มีรุ่นไหนบ้าง งบไม่เกิน 40,000", budget=40_000)
    check("exact", "บอกจำนวนรุ่น RTX 4060 ในงบตรงกับข้อมูล", True,
          bool(notes) and f"มีทั้งหมด {len(rtx4060)} รุ่น" in notes[0], notes[0][:80] if notes else "")
    check("exact", "คำถามทั่วไป -> ไม่ยุ่ง", ([], []), app.exact_match("โน้ตบุ๊คเล่นเกม งบ 25,000", budget=25_000), "")

    # เงื่อนไขตัวเลข (hybrid retrieval): แรม / จัดอันดับคะแนน / ถูกสุดในกลุ่ม / รุ่นที่เกินงบ
    H = "hybrid"
    cpu = lambda lp: lp["benchmark"]["CPU Score"]
    notes = lambda docs: " ".join(d for d in docs if d.startswith(app.EXACT_NOTE))
    check(H, "แรม 24 = 24GB พอดี", (24, False), app.ram_condition("แรม 24 มีตัวไหนบ้าง"), "")
    check(H, "แรม 32GB ขึ้นไป = ตั้งแต่ 32", (32, True), app.ram_condition("แรม 32GB ขึ้นไปมีรุ่นไหนบ้าง"), "")
    check(H, "16GB แรม (ตัวเลขนำหน้า)", (16, False), app.ram_condition("อยากได้ 16GB แรม"), "")
    check(H, "SSD 512GB ไม่ใช่แรม", None, app.ram_condition("SSD 512GB งบ 30,000"), "")
    check(H, "\"RTX 4060 รุ่นไหน\" ไม่ใช่ขอ 4060 รุ่น", None, app.wanted_model_count("RTX 4060 รุ่นไหนคุ้มสุด"), "")
    ram24 = [lp for lp in laptops if app._ram_gb(lp["spec"]["RAM"]) == 24]
    docs, _ = app.exact_match("แรม 24 มีตัวไหนบ้าง")
    check(H, "แรม 24 นับครบทุกรุ่น", True, f"มีทั้งหมด {len(ram24)} รุ่น" in notes(docs), notes(docs)[:80])
    docs, _ = app.exact_match("เรียงจากราคามากไปน้อยให้หน่อยตัว ram 24")
    priciest24 = max(ram24, key=lambda lp: lp["ราคา"])["ชื่อสินค้า"]
    check(H, "เรียงแพงไปถูก: รุ่นแพงสุดขึ้นก่อน", True, f"(เรียงราคาจากแพงไปถูก): {priciest24} " in notes(docs), "")
    top = max(laptops, key=cpu)["ชื่อสินค้า"]
    docs, _ = app.exact_match("รุ่นไหนคะแนน CPU สูงที่สุด")
    check(H, "CPU สูงสุดของทั้งร้านเป็นอันดับ 1", True, f"1. {top} " in notes(docs), "")
    top30 = max((lp for lp in laptops if lp["ราคา"] <= 30_000), key=cpu)["ชื่อสินค้า"]
    docs, _ = app.exact_match("งบ 30,000 รุ่นไหน CPU แรงที่สุด", budget=30_000)
    check(H, "CPU แรงสุดในงบเป็นอันดับ 1", True, f"1. {top30} " in notes(docs), "")
    rtx = min((lp for lp in laptops if "rtx" in lp["spec"]["GPU"].lower()), key=lambda lp: lp["ราคา"])
    q = "รุ่นที่มีการ์ดจอ RTX ที่ถูกที่สุดคือรุ่นไหน"
    docs, _ = app.exact_match(q)
    check(H, "RTX ที่ถูกที่สุด = ถูกสุดในกลุ่ม", True, f"คือ {rtx['ชื่อสินค้า']} " in notes(docs), "")
    check(H, "ถูกสุดในกลุ่ม -> ไม่ยัดรุ่นถูกสุดทั้งร้าน", None, app._extreme_spec_doc(q, None), "")
    docs, _ = app.exact_match("เทียบ Legion 5 กับ HP Victus 16 งบ 45000", budget=45_000)
    check(H, "Legion 5 เกินงบ -> บอกว่ามีแต่เกินงบ", True,
          "Legion 5" in notes(docs) and "ห้ามตอบว่าไม่มีในข้อมูล" in notes(docs), "")

    names = {t for t in app.TOOLS}
    guide = all(n in app.PI_TOOL_GUIDE for n in names)
    check("tools", "PI_TOOL_GUIDE เอ่ยถึง tool ครบ", True, guide, "ชื่อ tool ต้องตรงกับ pi-agent/server.mjs")


# ─────────────────────────────────────────────────────────────
# 7. สตรีมความคืบหน้าของ Pi Agent (/chat/stream + /chat/commit)
# ─────────────────────────────────────────────────────────────
def _ndjson(*events):
    """จำลองบรรทัดที่ requests.iter_lines() อ่านได้จาก pi-agent /chat/stream"""
    return [json.dumps(e, ensure_ascii=False).encode() for e in events]


def _screen_text(events):
    """ข้อความที่หน้าเว็บจะโชว์หลังได้ event ชุดนี้ (delta = ต่อท้าย, replace = แทนทั้งก้อน)"""
    text = ""
    for ev in events:
        if ev["type"] == "delta":
            text += ev["text"]
        elif ev["type"] == "replace":
            text = ev["text"]
    return text


def test_stream():
    S = "stream"
    check(S, "tool_status: filter + งบ", "กำลังกรองสเปก: RTX 4060 · ไม่เกิน 40,000 บาท…",
          app.tool_status("filter_laptops", {"gpu": "RTX 4060", "max_price": 40000}), "")
    check(S, "tool_status: ช่วงราคา + RAM", "กำลังกรองสเปก: RAM 16GB ขึ้นไป · 30,000–40,000 บาท…",
          app.tool_status("filter_laptops", {"min_ram_gb": 16, "min_price": "30,000", "max_price": 40000.0}), "")
    check(S, "tool_status: search มีคำค้น", "กำลังค้นสเปกและรีวิว: “แบตอึด”…",
          app.tool_status("search_laptops", {"query": "แบตอึด"}), "")
    check(S, "tool_status: ไม่มี argument", "กำลังดูรายละเอียดรุ่น…", app.tool_status("get_laptop_details", None), "")
    check(S, "tool_status: ราคาไม่ใช่ตัวเลขไม่พัง", "กำลังกรองสเปก…",
          app.tool_status("filter_laptops", {"max_price": "ถูก ๆ"}), "")
    check(S, "tool_status: ตัดข้อความยาว", True,
          len(app.tool_status("search_laptops", {"query": "ก" * 300})) < 110, "")

    check(S, "visible_answer: กั๊กแท็กเปิดที่มาไม่ครบ", "คำตอบ", app.visible_answer("คำตอบ <thi"), "")
    check(S, "visible_answer: ซ่อนส่วนที่ยังคิดไม่จบ", "ก่อน", app.visible_answer("ก่อน<think>กำลังคิด"), "")
    check(S, "visible_answer: < ท้ายข้อความถูกกั๊กไว้ก่อน", "ราคา", app.visible_answer("ราคา <"),
          "ไม่เสียหาย: คำตอบสุดท้ายที่โชว์มาจาก event done ไม่ได้มาจากตรงนี้")

    # รอบแรก AI พูดก่อนเรียก tool -> ข้อความนั้นต้องถูกล้าง แล้วรอบสองมี <think> ถูกหั่นกลางแท็ก
    events = list(app.relay_pi_events(_ndjson(
        {"type": "message_start"},
        {"type": "text", "delta": "ขอค้นเพิ่ม"},
        {"type": "tool_start", "id": "t1", "name": "filter_laptops", "args": {"gpu": "RTX 4060"}},
        {"type": "tool_end", "id": "t1", "name": "filter_laptops", "ok": True, "ms": 120},
        {"type": "message_start"},
        {"type": "text", "delta": "<thi"},
        {"type": "text", "delta": "nk>ลองคิดดู</think>"},
        {"type": "text", "delta": "มี **2 รุ่น**"},
        {"type": "text", "delta": " ครับ"},
        {"type": "done", "message": "<think>ลองคิดดู</think>มี **2 รุ่น** ครับ",
         "tools": [{"name": "filter_laptops", "ok": True, "ms": 120}]},
    )))
    statuses = [e["text"] for e in events if e["type"] == "status"]
    check(S, "relay: สถานะตอนเรียก tool", ["กำลังกรองสเปก: RTX 4060…", app.STATUS_WRITING], statuses, "")
    check(S, "relay: ล้างข้อความรอบก่อนเรียก tool", True,
          {"type": "replace", "text": ""} in events, "")
    check(S, "relay: จอสุดท้ายตรงกับคำตอบ", "มี **2 รุ่น** ครับ", _screen_text(events), "")
    check(S, "relay: ไม่มี <think> หลุดขึ้นจอ", False,
          any("think" in e.get("text", "") or "<" in e.get("text", "") for e in events), "")
    check(S, "relay: event สุดท้ายคือ done ที่ strip_think แล้ว",
          ("done", "มี **2 รุ่น** ครับ", 1), (events[-1]["type"], events[-1]["message"], len(events[-1]["tools"])), "")

    # tool สองตัวทำงานพร้อมกัน: ตัวแรกเสร็จ สถานะต้องยังเป็นของตัวที่ยังไม่เสร็จ
    events = list(app.relay_pi_events(_ndjson(
        {"type": "tool_start", "id": "a", "name": "filter_laptops", "args": {}},
        {"type": "tool_start", "id": "b", "name": "search_laptops", "args": {"query": "เบา"}},
        {"type": "tool_end", "id": "b", "ok": True},
        {"type": "done", "message": ""},
    )))
    check(S, "relay: tool ซ้อนกัน", "กำลังกรองสเปก…", events[2]["text"], "")
    check(S, "relay: คำตอบว่าง -> EMPTY_ANSWER", app.EMPTY_ANSWER, events[-1]["message"], "")

    for name, lines, word in (
        ("relay: pi-agent แจ้ง error", _ndjson({"type": "error", "status": 502, "error": "โมเดลล่ม"}), "โมเดลล่ม"),
        ("relay: สายขาดก่อน done", _ndjson({"type": "text", "delta": "ครึ่ง"}), "ก่อนตอบเสร็จ"),
    ):
        try:
            list(app.relay_pi_events(lines, "X"))
            check(S, name, "LLMError", "ไม่ raise", "")
        except app.LLMError as e:
            check(S, name, True, word in str(e), str(e))

    # commit token: ผู้ใช้แก้คำตอบที่จะเข้า context ของ LLM ไม่ได้
    token = app.commit_token("ถาม", "ตอบ")
    check(S, "commit token อ่านกลับได้", ("ถาม", "ตอบ"), app.read_commit_token(token), "")
    check(S, "commit token ถูกแก้ -> None", None, app.read_commit_token(token[:-2] + "xx"), "")
    check(S, "commit token มั่ว -> None", None, app.read_commit_token("hello"), "")

    client = app.app.test_client()
    r = client.post("/chat/commit", json={"token": token})
    r = client.post("/chat/commit", json={"token": token})   # ยิงซ้ำต้องไม่บันทึกซ้ำ
    with client.session_transaction() as sess:
        history = sess.get("history")
    check(S, "/chat/commit บันทึกประวัติครั้งเดียว",
          [{"role": "user", "content": "ถาม"}, {"role": "assistant", "content": "ตอบ"}], history, "")
    r = client.post("/chat/commit", json={"token": "x" + token})
    check(S, "/chat/commit token ปลอม -> 400", 400, r.status_code, "")
    r = client.post("/chat/stream", json={"message": "  "})
    check(S, "/chat/stream ข้อความว่าง -> JSON 400", (400, "application/json"),
          (r.status_code, r.content_type), "error ก่อนเริ่มสตรีมต้องเหมือน /chat")


# ─────────────────────────────────────────────────────────────
# 8. โหมด debug (DEBUG_PANEL) — แผงกระบวนการคิดใต้คำตอบ
# ─────────────────────────────────────────────────────────────
def test_debug():
    D = "debug"
    saved = app.DEBUG_PANEL
    client = app.app.test_client()
    try:
        # ผู้ดูแลปิดไว้ = ไม่สนค่า debug ที่หน้าเว็บส่งมา (กันคนนอกขอดู prompt ทั้งก้อน)
        for on, body, want in ((False, {"debug": True}, False), (True, {"debug": True}, True),
                               (True, {"debug": "yes"}, False), (True, {}, False)):
            app.DEBUG_PANEL = on
            with app.app.test_request_context("/chat", method="POST",
                                              json={"message": "เล่นเกม งบ 25000", **body}):
                job, err = app._chat_request()
            if err:
                check(D, "_chat_request ใช้ได้", True, False, "ไม่มี AI ที่ใช้ได้ในเครื่องนี้")
                break
            check(D, f"DEBUG_PANEL={on} ส่ง {body} -> debug={want}", want, job["debug"], "")
            check(D, f"new_trace ตาม debug={want}", want, app.new_trace(job) is not None, "")
            if want:
                trace = app.new_trace(job)
                check(D, "ตีความงบจากข้อความนี้", (25000, "message"),
                      (trace["query"]["budget"], trace["query"]["budget_from"]), "")
        for on in (False, True):
            app.DEBUG_PANEL = on
            check(D, f"/providers บอกสถานะ debug={on}", on, client.get("/providers").get_json().get("debug"), "")
    finally:
        app.DEBUG_PANEL = saved

    # ความคิดของโมเดลมาได้ทั้งฟิลด์ reasoning และบล็อก <think> (รวมบล็อกที่ token หมดก่อนปิด)
    info = app.llm_trace({"choices": [{"message": {"content": "<think>คิดก่อน</think>ตอบ<think>ค้าง",
                                                   "reasoning": "ขั้นแรก"}, "finish_reason": "stop"}],
                          "usage": {"total_tokens": 10}}, "<think>คิดก่อน</think>ตอบ<think>ค้าง", 1.2346)
    check(D, "llm_trace เก็บความคิดครบ 3 ส่วน", "ขั้นแรก\n\nคิดก่อน\n\nค้าง", info["thinking"], "")
    check(D, "llm_trace เวลา/โทเคน/เหตุที่จบ", (1235, 10, "stop"),
          (info["ms"], info["usage"]["total_tokens"], info["finish_reason"]), "")
    no_think = app.llm_trace({"choices": [{"message": {"content": "ตอบเลย"}}]}, "ตอบเลย", 0.5)
    check(D, "llm_trace ไม่มีความคิด = ว่าง", "", no_think["thinking"], "")

    spec = "Acer Nitro V 15 ANV15-51-56QE ราคา 30,990 บาท | ซีพียู (CPU): i5"
    review = "รีวิวจากผู้ใช้รุ่น Acer Nitro V 15 ANV15-51-56QE ราคา 30,990 บาท (คะแนน...): ดี"
    check(D, "doc_kind / doc_name เอกสารสเปก", ("spec", "Acer Nitro V 15 ANV15-51-56QE"),
          (app.doc_kind(spec), app.doc_name(spec)), "")
    check(D, "doc_kind / doc_name เอกสารรีวิว", ("review", "Acer Nitro V 15 ANV15-51-56QE"),
          (app.doc_kind(review), app.doc_name(review)), "")
    check(D, "doc_kind ผลค้นตรง", "note", app.doc_kind("ผลค้นตรงตามสเปก (นับจากข้อมูลทั้งหมด): ..."), "")

    # ค้นจริง (ต้องมี Chroma + โมเดล embed = รันใน container) — เครื่อง host ข้ามส่วนนี้
    if not app.CHROMA_DIR.exists() or getattr(sys.modules["chromadb"], "IS_STUB", False):
        return
    app._spec_doc_by_name = None   # test_tools ตั้งเป็น {} ไว้ (กรณีไม่มี Chroma)
    q = "RTX 4060 งบไม่เกิน 40,000"
    trace = {}
    ctx, over = app.retrieve(q, budget=40_000, trace=trace)
    plain, plain_over = app.retrieve(q, budget=40_000)
    check(D, "เปิด debug แล้วผลค้นต้องเหมือนเดิมทุกตัว", (plain, plain_over), (ctx, over), "")
    check(D, "embed: bge-m3 1024 มิติ + ตัวอย่าง 8 ค่า", (app.EMBED_MODEL, 1024, 8),
          (trace["embed"]["model"], trace["embed"]["dims"], len(trace["embed"]["preview"])), "")
    hits = [h for s in trace["searches"] for h in s["hits"]]
    check(D, "ผลค้นทุกตัวอยู่ในงบ", True, bool(hits) and all(h["price"] <= 40_000 for h in hits), "")
    check(D, "ความใกล้เคียงเรียงจากมากไปน้อยในแต่ละรอบ", True,
          all([h["similarity"] for h in s["hits"]] == sorted((h["similarity"] for h in s["hits"]), reverse=True)
              for s in trace["searches"]), "")
    check(D, "ความใกล้เคียงอยู่ในช่วง 0-1", True, all(0 <= h["similarity"] <= 1 for h in hits), "")
    check(D, "รายการเอกสารใน trace = ที่ส่งให้ AI จริง", len(ctx), len(trace["context"]), "")
    sources = {c["source"] for c in trace["context"]}
    check(D, "ที่มาของเอกสารเป็นชื่อที่หน้าเว็บรู้จัก", True,
          sources <= {"vector", "exact", "near_price", "extreme", "summary"}, str(sources))
    check(D, "RTX 4060 -> มีเอกสารจากการค้นตรง + บันทึกการปรับผลค้น", (True, True),
          ("exact" in sources, any("RTX 4060" in a for a in trace["adjust"])), "")


# ─────────────────────────────────────────────────────────────
# 8.5 โหลดโมเดล embedding ล่วงหน้าตอนแอปเริ่ม (WARMUP_EMBEDDINGS)
#     คำถามแรกหลังรีสตาร์ตไม่ต้องรอโหลด bge-m3 + embed 99 รุ่นเอง (เคยนานจน Cloudflare ตัดเป็น 524)
#     ใช้โมเดลปลอมทั้งหมด -> รันได้ทั้งบนเครื่อง host และใน container โดยไม่โหลดโมเดลจริง
# ─────────────────────────────────────────────────────────────
def test_warmup():
    import contextlib
    import io
    import threading
    import time

    import numpy as np
    import rag_v2

    W = "warmup"
    check(W, "ปิดได้ด้วย WARMUP_EMBEDDINGS=false (ชุดเทสต์ตั้งไว้)", False, app.WARMUP_EMBEDDINGS, "")
    saved = {k: getattr(app, k) for k in ("RAG_PIPELINE", "WARMUP_EMBEDDINGS", "_embedder",
                                          "SentenceTransformer", "get_embedder", "get_collection")}
    saved_v2 = {k: getattr(rag_v2, k) for k in ("doc_vectors", "_doc_vecs")}
    calls, built = [], []

    class FakeModel:
        def __init__(self, name=None):
            built.append(name)
            time.sleep(0.1)   # โหลดช้าพอให้ thread อื่นเข้ามาซ้อน

        def encode(self, texts, **kw):
            calls.append(("encode", len(texts)))
            time.sleep(0.05)
            return np.zeros((len(texts), 4))

    def warmup(**patch) -> tuple[bool, str]:
        for k, v in patch.items():
            setattr(app, k, v)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            ok = app.warmup_embeddings()
        return ok, out.getvalue()

    def together(fn, n=4) -> list:
        got = []
        threads = [threading.Thread(target=lambda: got.append(fn())) for _ in range(n)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(5)
        return got

    try:
        collection = lambda: calls.append("collection")
        rag_v2.doc_vectors = lambda: calls.append("doc_vectors")
        ok, log = warmup(RAG_PIPELINE="v2", get_collection=collection)
        check(W, "v2 = เตรียมเวกเตอร์ 99 รุ่น (โหลดโมเดลในตัว) + log ว่าพร้อม", (True, ["doc_vectors"], True),
              (ok, list(calls), "พร้อมแล้ว" in log), log)

        calls.clear()
        model = FakeModel()
        ok, log = warmup(RAG_PIPELINE="v1", get_collection=collection, get_embedder=lambda: model)
        check(W, "v1 = เปิด Chroma + โหลดโมเดล ไม่แตะ doc_vectors", (True, ["collection", ("encode", 1)]),
              (ok, list(calls)), log)

        def broken():
            raise RuntimeError("ไม่มีไฟล์โมเดล")
        rag_v2.doc_vectors = broken
        try:
            ok, log = warmup(RAG_PIPELINE="v2")
            check(W, "โหลดพัง = log สาเหตุแล้วไปต่อ (คืน False)", (False, True), (ok, "ไม่มีไฟล์โมเดล" in log), log)
        except Exception as e:
            check(W, "โหลดพัง = log สาเหตุแล้วไปต่อ (คืน False)", "ไม่โยน exception", repr(e), "")

        calls.clear()
        rag_v2.doc_vectors = lambda: calls.append("doc_vectors")
        app.WARMUP_EMBEDDINGS = False
        check(W, "ปิดไว้ = ไม่เริ่ม thread ไม่โหลดอะไร", (None, []), (app.start_warmup(), list(calls)), "")

        # เปิดไว้ = โหลดใน thread พื้นหลัง: start_warmup คืนทันทีทั้งที่การโหลดยังไม่จบ
        gate = threading.Event()
        rag_v2.doc_vectors = lambda: gate.wait(5)
        app.WARMUP_EMBEDDINGS = True
        with contextlib.redirect_stdout(io.StringIO()):
            t0 = time.perf_counter()
            thread = app.start_warmup()
            returned = time.perf_counter() - t0
            alive = thread is not None and thread.is_alive()
            gate.set()
            if thread is not None:
                thread.join(5)
        check(W, "เปิดไว้ = thread daemon ไม่บล็อกการเปิดเซิร์ฟเวอร์", (True, True, True),
              (alive, thread is not None and thread.daemon, returned < 1), f"{returned:.2f}s")

        # คำถามที่เข้ามาตอน warmup กำลังโหลด ต้องรอผลเดียวกัน ไม่โหลดโมเดล/embed 99 รุ่นซ้ำอีกรอบ
        app.get_embedder, app.SentenceTransformer, app._embedder = saved["get_embedder"], FakeModel, None
        built.clear()
        got = together(app.get_embedder)
        check(W, "get_embedder พร้อมกัน 4 thread = สร้างโมเดลครั้งเดียว", (1, 4, 1),
              (len(built), len(got), len({id(m) for m in got})), "")
        calls.clear()
        rag_v2.doc_vectors, rag_v2._doc_vecs = saved_v2["doc_vectors"], None
        got = together(rag_v2.doc_vectors)
        check(W, "doc_vectors พร้อมกัน 4 thread = embed 99 รุ่นครั้งเดียว", ([("encode", len(rag_v2.catalog()))], 4, True),
              (list(calls), len(got), all(v is got[0] for v in got)), "")
    finally:
        for k, v in saved.items():
            setattr(app, k, v)
        for k, v in saved_v2.items():
            setattr(rag_v2, k, v)


# ─────────────────────────────────────────────────────────────
# 9. RAG รุ่น 2 (rag_v2.py) — ตีความเป็นเงื่อนไข -> กรอง/เรียงจากข้อมูลจริง (ส่วนที่ไม่ต้องเรียก LLM)
# ─────────────────────────────────────────────────────────────
def test_rag_v2():
    try:
        import rag_v2 as v2
    except ImportError:
        return   # โค้ดรุ่นที่ยังไม่มี rag_v2
    R = "rag_v2"
    plan = lambda **kw: v2.normalize({"intent": kw.pop("intent", "list"), "filters": kw.pop("filters", {}), **kw})

    p = v2.normalize({"intent": "บ้า", "filters": {"ram_gb": 23, "price_max": 15, "gpu_family": "xyz"},
                      "sort": {"by": "speed"}, "limit": 999})
    check(R, "normalize: ค่าผิดรูปแบบทิ้งเป็น null", ("recommend", None, None, None, None, None),
          (p["intent"], p["filters"]["ram_gb"], p["filters"]["price_max"], p["filters"]["gpu_family"], p["sort"], p["limit"]), "")
    check(R, "อ่าน JSON ที่มีข้อความปน", {"intent": "list"}, v2._parse_json('ok ```json\n{"intent": "list"}\n```'), "")
    check(R, "ขนาดจอรับได้ทั้งตัวเลขและข้อความ", [16.0, 15.6, 16.0, None],
          [v2.normalize({"filters": {"screen": s}})["filters"]["screen"] for s in (16, "15.6", "16 นิ้ว", "165Hz")], "")
    for q, want in (("จอ 16 นิ้ว งบไม่เกิน 30,000", 16.0), ("จอ 15.6 มีรุ่นไหน", 15.6),
                    ("จอ 165Hz งบ 40000", None), ("แรม 16GB งบ 16000", None)):
        p = v2.normalize({})
        v2._safety_net(p, q)
        check(R, f"ตัวเติมจากข้อความ: ขนาดจอ «{q}»", want, p["filters"]["screen"], "")

    laptops = app.get_laptops()
    ram = lambda lp: app._ram_gb(lp["spec"]["RAM"])
    got = v2.select(plan(filters={"ram_gb": 24}), "แรม 24")["items"]
    check(R, "แรม 24 = ครบทุกรุ่น แรม 24 พอดี", (sum(ram(lp) == 24 for lp in laptops), True),
          (len(got), all(x["ram_gb"] == 24 for x in got)), "")
    screen = lambda lp: float(re.match(r"([\d.]+)", lp["spec"]["DISPLAY"]).group(1))
    got = v2.select(plan(filters={"screen": 16, "price_max": 30000}), "")["items"]
    check(R, "จอ 16 นิ้ว งบ 30,000 = ครบและไม่มีจอ 15.6 ปน",
          sum(int(screen(lp)) == 16 and lp["ราคา"] <= 30_000 for lp in laptops),
          len(got) if all(int(x["screen"]) == 16 for x in got) else -1, "")
    check(R, "จอ 15.6 = ขนาดนั้นพอดี / จอ 15 = 15.1-15.6", (True, True),
          (all(x["screen"] == 15.6 for x in v2.select(plan(filters={"screen": 15.6}), "")["items"]),
           [v2.screen_ok(x, 15) for x in (15.1, 15.3, 15.6, 16.0, 14.0)] == [True, True, True, False, False]), "")
    got = v2.select(plan(intent="list", filters={"ram_gb": 24}, sort={"by": "price", "order": "desc"}), "")["items"]
    check(R, "เรียงแพงไปถูก", True, [x["price"] for x in got] == sorted((x["price"] for x in got), reverse=True), "")
    top30 = max((lp for lp in laptops if lp["ราคา"] <= 30_000), key=lambda lp: lp["benchmark"]["CPU Score"])
    got = v2.select(plan(intent="rank", filters={"price_max": 30000}, sort={"by": "cpu_score", "order": "desc"}), "")["items"]
    check(R, "CPU แรงสุดในงบ = อันดับ 1", top30["ชื่อสินค้า"], got[0]["name"], "")
    got = v2.select(plan(filters={"ram_gb": 24, "price_target": 24000}), "")["items"]
    check(R, "ราคาประมาณ 24,000 = ช่วง ±15%", True,
          bool(got) and all(20_400 <= x["price"] <= 27_600 and x["ram_gb"] == 24 for x in got), str([x["price"] for x in got]))
    got = v2.select(plan(filters={"price_max": 10000}), "")
    check(R, "งบต่ำกว่ารุ่นถูกสุด -> เสนอรุ่นราคาใกล้ + หมายเหตุ", ("nearest", True),
          (got["source"], any("ไม่มีรุ่น" in n for n in got["notes"])), "")
    got = v2.select(plan(intent="compare", models=["Legion 5", "HP Victus 16"], filters={"price_max": 45000}), "")
    check(R, "เทียบรุ่นที่เกินงบ -> บอกว่ามีแต่เกินงบ", True,
          any("Legion 5" in x["name"] for x in got["items"]) and any("นอกงบ" in n for n in got["notes"]), "")
    found, missing = v2.resolve_models(["HP Victus 16", "MacBook Air"])
    check(R, "หาชื่อรุ่นจากชื่อย่อ / ชื่อที่ไม่มี", (True, ["MacBook Air"]),
          (bool(found) and all(x["name"].startswith("HP Victus 16") for x in found), missing), "")
    lookup = v2.select(plan(intent="lookup", models=["Acer Nitro V 15 ANV15-51-56QE", "HP Victus 16-r0356TX"],
                            sort={"by": "price", "order": "asc"}, limit=1), "")["items"]
    check(R, "ตัวที่ถูกกว่า = รุ่นแรกหลังเรียงราคา", ["Acer Nitro V 15 ANV15-51-56QE"], [x["name"] for x in lookup], "")
    check(R, "นอกเรื่อง = ข้อความสำเร็จรูป ไม่เรียก LLM", True,
          v2.respond(plan(intent="offtopic"), "วันนี้ฝนจะตกไหม", [], {}).startswith("เรื่องนี้ผมตอบไม่ได้"), "")
    check(R, "ยี่ห้อที่ไม่มี = บอกชื่อยี่ห้อนั้น", True,
          "MacBook" in v2.respond(plan(intent="not_in_store"), "MacBook Air M3 ราคาเท่าไหร่", [], {}), "")
    p = plan(intent="recommend", usage="เล่นเกม", filters={"screen": 16, "price_max": 30000})
    check(R, "ตัดการใช้งานที่ลูกค้าไม่ได้พูด", ("เล่นเกม", None),
          (v2._ground_usage(p, "จอ 16 นิ้ว งบไม่เกิน 30,000 มีรุ่นไหนบ้าง", [], None), p["usage"]), "")
    p = plan(intent="recommend", usage="ตัดต่อวิดีโอ")
    check(R, "เก็บการใช้งานที่ลูกค้าพูดไว้ก่อนหน้า", (None, "ตัดต่อวิดีโอ"),
          (v2._ground_usage(p, "มีรุ่นไหนอีกบ้าง", [{"role": "user", "content": "ตัดต่อวิดีโอ งบ 40000"}], None),
           p["usage"]), "")
    for q, filters, usage, want in (
            ("ขอรุ่นที่ใช้ Core i7 งบไม่เกิน 35k", {"cpu": "Core i7", "price_max": 35000}, None, "list"),
            ("งบ 25000 มีรุ่นไหนบ้าง", {"price_max": 25000}, None, "list"),
            ("RTX 4060 งบ 40000 แนะนำหน่อย", {"gpu": "RTX 4060", "price_max": 40000}, None, "recommend"),
            ("เล่นเกม งบ 25000", {"price_max": 25000}, "เล่นเกม", "recommend"),
            ("งบ 20000", {"price_max": 20000}, None, "recommend")):
        p = plan(intent="recommend", filters=filters, usage=usage)
        v2.check_plan(p, q, [], None)
        check(R, f"แนะนำ หรือ ขอรายชื่อ «{q}»", want, p["intent"], "")
    if hasattr(v2, "tidy"):   # คำตอบละเอียด + คำแนะนำท้ายคำตอบ + สตรีม
        check(R, "เก็บกวาด ครับครับ", "ได้เลยครับ แล้วครับ", v2.tidy("ได้เลยครับครับ แล้วครับ ครับ"), "")
        check(R, "รายชื่อสั้นใส่สเปกหลักครบ / ยาวใส่เฉพาะที่ถาม", (True, False),
              ("CPU" in v2._list_line(plan(filters={"ram_gb": 24}), 13),
               "CPU" in v2._list_line(plan(filters={"ram_gb": 16}), 30)), "")
        check(R, "ภาพรวมร้านมีช่วงราคาแยกยี่ห้อ", True, "Lenovo 25 รุ่น (" in v2.store_stats(), "")
        result = {}
        events = list(v2.respond_stream(plan(intent="greeting"), "สวัสดี", [], {}, None, result))
        check(R, "ทักทาย = ข้อความสำเร็จรูป ไม่สตรีม ไม่เรียก AI", ([], True),
              (events, result["answer"].startswith("สวัสดีครับ")), "")
    if hasattr(v2, "unsupported_terms"):   # เงื่อนไขที่ระบบยังคัดไม่ได้ -> บอกลูกค้าตรง ๆ
        for q, want in (
                ("รุ่นไหนมีจอ 240Hz", ["ความถี่จอ (Hz)"]),
                ("รุ่นไหนใช้ SSD Gen 5 หรือ Gen 3", ["Gen ของ SSD"]),
                ("รุ่นไหนใช้ Windows 11 Home Single Language", ["รุ่นของ Windows"]),
                ("รุ่นไหนจอความละเอียดสูงกว่า 2560x1600", ["ความละเอียดจอ"]),
                ("จอ 2K งบ 40000", ["ความละเอียดจอ"]),
                ("รุ่นไหนรองรับชาร์จผ่าน USB-C", ["พอร์ตและช่องเชื่อมต่อ"]),
                ("รุ่นไหนผ่านมาตรฐาน MIL-STD 810H", ["มาตรฐาน MIL-STD"]),
                ("รุ่นไหนมี Number pad", ["แป้นตัวเลข (Number pad)"]),
                ("SSD 1TB งบ 30000", ["ความจุ SSD"]),
                ("น้ำหนักไม่เกิน 2 kg", ["น้ำหนักเครื่อง"]),
                ("แบตอยู่ได้ 10 ชั่วโมง", ["ชั่วโมงใช้งานแบต"]),
                ("จอ 165Hz ความละเอียด QHD", ["ความถี่จอ (Hz)", "ความละเอียดจอ"])):
            check(R, f"จับเงื่อนไขที่คัดไม่ได้ «{q}»", want, v2.unsupported_terms(q), "")
        for q in ("จอ 16 นิ้ว งบไม่เกิน 30,000", "งบ 30k เล่นเกม", "งบ 25k", "แรม 16GB งบ 25000", "RTX 4060 มีรุ่นไหนบ้าง",
                  "ต้องการเครื่องเบา แบตอึด พกไปร้านกาแฟ", "Intel UHD Graphics รุ่นไหนถูกสุด", "Legion 9i หนักกี่กิโล",
                  "Core i7 Gen 13 งบ 35000", "แรม 32GB ขึ้นไปตัวไหนถูกที่สุด"):
            check(R, f"ไม่จับเงื่อนไขที่คัดได้ «{q}»", [], v2.unsupported_terms(q), "")

        def with_terms(p, terms):
            p["unsupported"] = terms
            return p
        result = {}
        p = with_terms(plan(intent="list"), ["ความถี่จอ (Hz)"])
        events = list(v2.respond_stream(p, "รุ่นไหนมีจอ 240Hz", [], {}, None, result))
        check(R, "ถามแต่เรื่องที่คัดไม่ได้ = บอกตรง ๆ ไม่เรียก AI ไม่ส่งรุ่นที่ไม่ได้กรอง", ([], True, False),
              (events, result["answer"].startswith("ตอนนี้ผมยังคัดรุ่นตามความถี่จอ (Hz) ให้ไม่ได้ครับ"),
               "40 รุ่น" in result["answer"]), result["answer"][:80])
        check(R, "คำถามภาพรวมที่มีเรื่องคัดไม่ได้ = บอกตรง ๆ", True,
              bool(v2.unsupported_reply(with_terms(plan(intent="overview"), ["รุ่นของ Windows"]))), "")
        p = with_terms(plan(intent="list", filters={"price_max": 30000}), ["ความถี่จอ (Hz)"])
        lead = v2.unsupported_lead(p)
        check(R, "มีเงื่อนไขอื่นด้วย = กรองตามที่คัดได้ + ประโยคนำบอกว่าเรื่องไหนคัดไม่ได้", (None, True, True),
              (v2.unsupported_reply(p), lead.startswith("ผมยังคัดรุ่นตามความถี่จอ (Hz) ให้ไม่ได้ครับ"),
               "ราคาไม่เกิน 30,000 บาท" in lead), lead)
        check(R, "เว้นวรรคระหว่างไทยกับอังกฤษ", True,
              "ตาม Gen ของ SSD ให้ไม่ได้" in v2.unsupported_lead(
                  with_terms(plan(intent="recommend", usage="เล่นเกม"), ["Gen ของ SSD"])), "")
        p = with_terms(plan(intent="lookup", models=["Asus ZenBook 14 OLED"]), ["ความละเอียดจอ"])
        check(R, "ถามสเปกของรุ่นที่ระบุ = ตอบตามปกติ (AI อ่านสเปกรุ่นนั้นได้)", (None, ""),
              (v2.unsupported_reply(p), v2.unsupported_lead(p)), "")
        saved = v2.complete_stream
        v2.complete_stream = lambda *a, **k: iter(["มีทั้งหมด", " 7 รุ่นครับ"])
        try:
            p = with_terms(plan(intent="list", filters={"ram_gb": 24}), ["Gen ของ SSD"])
            result = {}
            events = list(v2.respond_stream(p, "แรม 24 ตัวไหนใช้ SSD Gen 5", [], {}, None, result))
            check(R, "สตรีม: ประโยคนำขึ้นก่อน แล้วต่อด้วยคำตอบของ AI", (True, True),
                  (events[0]["text"].startswith("ผมยังคัดรุ่นตาม Gen ของ SSD"),
                   result["answer"].startswith(events[0]["text"]) and result["answer"].endswith("มีทั้งหมด 7 รุ่นครับ")),
                  str(events[:2]))
        finally:
            v2.complete_stream = saved
        picked = {"items": [], "total": 99, "notes": [], "missing": [], "source": "filter"}
        check(R, "ไม่ได้กรองอะไร = บอก AI ว่ารายการมาจากทั้งร้าน ไม่ใช่ตรงเงื่อนไข", (True, True),
              ("จากทั้งร้าน 99 รุ่น" in v2.build_answer_prompt(plan(intent="list"), picked, "ขอสเปคทุกตัว", []),
               "จากที่ตรงเงื่อนไข 99 รุ่น" in v2.build_answer_prompt(plan(intent="list", filters={"ram_gb": 24}),
                                                                     picked, "แรม 24", [])), "")

    if hasattr(v2, "spec_extras"):   # ความละเอียดจอ / SSD / Windows อ่านจากคำถามด้วยโค้ด
        for q, want in (
                ("รุ่นไหนจอความละเอียดสูงกว่า 2560x1600", {"res_gt": 2560}),
                ("จอ 2560x1600 มีรุ่นไหนบ้าง", {"res_w": 2560, "res_h": 1600}),
                ("จอ 2K งบไม่เกิน 40000", {"res_w": 2560}),
                ("จอ 2K ขึ้นไป", {"res_min": 2560}),
                ("จอ 4K มีไหม", {"res_w": 3840}),
                ("รุ่นไหนใช้ SSD Gen 5 หรือ Gen 3", {"ssd_gen": [3, 5]}),
                ("SSD Gen 4 ขึ้นไป งบ 30000", {"ssd_gen": [4, 5]}),
                ("SSD 1TB งบ 30000", {"storage_gb": 1000}),
                ("ความจุ 1TB ขึ้นไป", {"storage_min": 1000}),
                ("รุ่นไหนใช้ Windows 11 Home Single Language", {"os": "single language"}),
                ("มีรุ่นที่เป็น Windows 11 Pro ไหม", {"os": "pro"})):
            check(R, f"อ่านเงื่อนไขสเปกจากคำถาม «{q}»", want, v2.spec_extras(q)[0], "")
        for q in ("Core i7 Gen 13 งบ 35000", "Intel UHD Graphics รุ่นไหนถูกสุด", "งบ 25k เล่นเกม",
                  "แรม 16GB งบ 25000", "Legion Pro 5 ราคาเท่าไหร่", "จอ 16 นิ้ว งบ 30000"):
            check(R, f"ไม่อ่านเป็นเงื่อนไขจอ/SSD/Windows «{q}»", {}, v2.spec_extras(q)[0], "")
        for q in ("รุ่นไหนใช้ SSD Gen 5 หรือ Gen 3", "รุ่นไหนจอความละเอียดสูงกว่า 2560x1600",
                  "รุ่นไหนใช้ Windows 11 Home Single Language", "SSD 1TB งบ 30000"):
            covered = v2.spec_extras(q)[1]
            check(R, f"คัดได้แล้ว ไม่บอกว่าคัดไม่ได้ «{q}»", [],
                  [t for t in v2.unsupported_terms(q) if t not in covered], "")
        disp, stor = (lambda lp: lp["spec"]["DISPLAY"]), (lambda lp: lp["spec"]["STORAGE"])
        width = lambda lp: int(re.search(r"\((\d{4})\s*x", disp(lp)).group(1)) if re.search(r"\((\d{4})\s*x", disp(lp)) else 0
        for fl, want, label in (
                ({"res_gt": 2560}, sum(width(lp) > 2560 for lp in laptops), "จอละเอียดกว่า 2560x1600"),
                ({"res_w": 2560}, sum(width(lp) == 2560 for lp in laptops), "จอกว้าง 2560"),
                ({"ssd_gen": [3, 5]}, sum(bool(re.search(r"Gen [35]$", stor(lp))) for lp in laptops), "SSD Gen 3 หรือ 5"),
                ({"storage_gb": 1000}, sum(stor(lp).startswith("1TB") for lp in laptops), "ความจุ 1TB"),
                ({"os": "single language"}, sum("Single Language" in lp["spec"]["OS"] for lp in laptops),
                 "Windows Single Language")):
            p = plan()
            p["filters"].update(fl)   # โค้ดเป็นคนใส่ (normalize ไม่รับค่าพวกนี้จาก LLM)
            check(R, f"กรอง{label} = ครบทุกรุ่น", want, v2.select(p, "")["total"], "")
        p = plan()
        p["filters"].update({"ssd_gen": [3, 5], "storage_gb": 1000})
        check(R, "อธิบายเงื่อนไขใหม่เป็นภาษาคน", "SSD Gen 3 หรือ 5 · ความจุ 1TB", v2.describe(p["filters"]), "")

    mem = v2.memory(plan(filters={"ram_gb": 24}))
    check(R, "จำเงื่อนไขเฉพาะที่มีค่า (cookie เล็ก)", {"ram_gb": 24}, mem["filters"], "")


if __name__ == "__main__":
    test_budget()
    test_strip_think()
    test_documents()
    test_data()
    test_providers()
    test_tools()
    test_stream()
    test_debug()
    test_warmup()
    test_rag_v2()

    print(f"\n{'='*78}\nUNIT TESTS: ผ่าน {len(PASS)} / {len(PASS)+len(FAIL)}\n{'='*78}")
    if FAIL:
        print(f"\n❌ ไม่ผ่าน {len(FAIL)} เคส:\n")
        for group, case, exp, act, note in FAIL:
            print(f"  [{group}] {case}")
            print(f"      ควรได้: {exp!r}   แต่ได้: {act!r}   ({note})")
    else:
        print("\n✅ ผ่านทั้งหมด")
    sys.exit(1 if FAIL else 0)
