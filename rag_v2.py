"""Tech Buddy RAG รุ่น 2 — self-query retrieval: ตีความคำถามเป็นเงื่อนไข -> กรอง/เรียงจากข้อมูลจริง -> ตอบ

ทำไมทำใหม่: รุ่นแรกค้นตามความหมายอย่างเดียว (เห็นแค่ ~10 จาก 397 เอกสาร) แล้วปะกฎ regex ทีละเรื่อง
คำถามที่ต้องกรองตัวเลข (แรม 24) หรือจัดอันดับทั้งร้าน (CPU แรงสุด) จึงพลาด และกฎใน prompt ชนกันเอง
(เช่น "ยังไม่รู้งบ/การใช้งานให้ถามกลับก่อน" ชนกับ "ขอสเปคทุกตัว") ถามต่อก็จำได้แค่งบ

ขั้นตอน (เรียก LLM 2 ครั้งต่อคำถาม — ยังเป็น RAG ไม่ใช่ agent: ไม่มีการวนเรียกเครื่องมือ)
  1. understand(): LLM แปลงคำถาม + ประวัติแชท + เงื่อนไขเดิม เป็น JSON (intent, filters, sort, limit, usage)
     แล้วโค้ดตรวจ/เติมด้วยตัวอ่านตัวเลขเดิม (งบ แรม การ์ดจอ) ที่ทดสอบแล้ว
  2. select(): กรอง/เรียงจากข้อมูลจริงทั้ง 99 รุ่นด้วยโค้ด แล้วใช้ embedding จัดลำดับตามการใช้งาน
     ข้อมูล 1 รุ่น = 1 เอกสาร (สเปก + คะแนน + ระดับ + รีวิว รวมกัน) AI จึงไม่เจอรุ่นที่มีแต่รีวิวไม่มีสเปก
  3. respond(): ส่งรายการที่คัดแล้วให้ LLM เขียนตามรูปแบบของ intent
     คำถามนอกเรื่อง/ทักทาย/ความรู้ทั่วไป/ยี่ห้อที่ไม่มี ใช้ข้อความสำเร็จรูป ไม่เรียก LLM รอบสอง
"""
import json
import re
import time

import numpy as np
import requests

INTENTS = ("recommend", "list", "rank", "lookup", "compare", "overview",
           "not_in_store", "general", "greeting", "offtopic")
TEMPLATED = ("not_in_store", "general", "greeting", "offtopic")
SORT_FIELDS = {"price": "price", "cpu_score": "cpu_score", "gpu_score": "gpu_score", "ram": "ram_gb"}
RAM_SIZES = {4, 8, 12, 16, 24, 32, 48, 64, 96, 128}
LIST_CAP = 40          # ไล่รายชื่อได้สูงสุด (ทั้งร้านมี 99 รุ่น)
RECOMMEND_POOL = 3     # รุ่นที่ส่งให้ AI เลือกแนะนำ (ถ้าลูกค้าไม่ได้ขอจำนวน)
RANK_TOP = 5           # อันดับที่แนบไปให้ AI เล่าต่อในคำถามจัดอันดับ
NAMED_CAP = 8
PRICE_BAND = 0.15      # "ราคาประมาณ X" = X ±15%
HISTORY_TURNS = 4      # ข้อความย้อนหลังที่ส่งให้ตัวตีความ/ตัวตอบ
OUTSIDE_BRANDS = ("macbook", "apple", "dell", "thinkpad", "surface", "razer", "alienware", "samsung", "huawei", "microsoft")
BRAND_CASE = {"macbook": "MacBook", "thinkpad": "ThinkPad", "alienware": "Alienware"}

_catalog: list[dict] | None = None
_doc_vecs: np.ndarray | None = None


def _core():
    """helper ของ app.py (โหลดโมเดล embedding, อ่านงบ, ระดับคะแนน ...) — import ตอนใช้ กัน import วน"""
    import app as core
    return core


# ───────────────────────── 1. ข้อมูล: 1 รุ่น = 1 เอกสาร + ค่าตัวเลขให้กรองได้ ─────────────────────────
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


def catalog() -> list[dict]:
    global _catalog
    if _catalog is None:
        core = _core()
        out = []
        for lp in core.get_laptops():
            spec, bench = lp["spec"], lp["benchmark"]
            cs, gs = bench["CPU Score"], bench["GPU Score"]
            screen = re.match(r"([\d.]+)\s*inch", spec["DISPLAY"])
            rec = {
                "name": lp["ชื่อสินค้า"], "brand": lp["ชื่อสินค้า"].split()[0], "series": _series(lp["ชื่อสินค้า"]),
                "price": lp["ราคา"], "cpu": spec["CPU"], "gpu": spec["GPU"], "gpu_family": _gpu_family(spec["GPU"]),
                "ram": spec["RAM"], "ram_gb": core._ram_gb(spec["RAM"]), "display": spec["DISPLAY"],
                "screen": float(screen.group(1)) if screen else None, "storage": spec["STORAGE"], "os": spec["OS"],
                "cpu_score": cs, "gpu_score": gs,
                "cpu_level": core.score_level(cs, core.CPU_LEVELS), "gpu_level": core.score_level(gs, core.GPU_LEVELS),
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


def doc_vectors() -> np.ndarray:
    """เวกเตอร์ของเอกสารรุ่นละ 1 ชิ้น (99 ชิ้น) — embed ครั้งแรกที่ใช้ แล้วเก็บไว้ในหน่วยความจำ"""
    global _doc_vecs
    if _doc_vecs is None:
        docs = [lp["doc"] for lp in catalog()]
        _doc_vecs = np.asarray(_core().get_embedder().encode(docs, normalize_embeddings=True, batch_size=16))
    return _doc_vecs


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


# ───────────────────────── 2. ตีความคำถามเป็นเงื่อนไข ─────────────────────────
def _series_list() -> str:
    seen = []
    for lp in catalog():
        s = " ".join(lp["series"].split()[:3])
        if s not in seen:
            seen.append(s)
    return ", ".join(seen)


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


def _screen(v) -> float | None:
    """ขนาดจอ: รับทั้ง 16, "16", "16 นิ้ว" — นอกช่วง 11-19 นิ้วทิ้ง"""
    if v is None or isinstance(v, bool):
        return None
    m = re.search(r"\d+(?:\.\d+)?", str(v))
    size = float(m.group(0)) if m else None
    return size if size and 11 <= size <= 19 else None


_RE_SCREEN = re.compile(r"(?:จอ\s*)?(1[1-9](?:\.\d)?)\s*(?:นิ้ว|\"|inch|in\b)|จอ\s*(1[1-9](?:\.\d)?)(?!\d)", re.I)


def _int(v) -> int | None:
    if v is None or isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return int(v)
    m = re.search(r"\d[\d,]*", str(v))
    return int(m.group(0).replace(",", "")) if m else None


def _parse_json(text: str) -> dict:
    text = re.sub(r"<think>.*?</think>", "", text or "", flags=re.DOTALL)
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end <= start:
        raise ValueError("ไม่มี JSON")
    return json.loads(text[start:end + 1])


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
        if fl[k] is not None and fl[k] < 1000:   # "ไม่เกิน 15 นิ้ว" ไม่ใช่ราคา
            fl[k] = None
    if fl["gpu_family"] not in (None, "rtx", "gtx", "radeon rx", "arc", "integrated"):
        fl["gpu_family"] = None
    s = raw.get("sort")
    if isinstance(s, dict) and s.get("by") in SORT_FIELDS:
        plan["sort"] = {"by": s["by"], "order": "desc" if s.get("order") == "desc" else "asc"}
    if plan["limit"] is not None and not 1 <= plan["limit"] <= LIST_CAP:
        plan["limit"] = None
    return plan


def _safety_net(plan: dict, question: str) -> list[str]:
    """เติมตัวเลขที่ตัวอ่านเดิม (ทดสอบแล้ว) จับได้แต่ LLM ไม่ได้ใส่ — คืนรายการที่เติม"""
    core, fl, added = _core(), plan["filters"], []
    budget, low = core.extract_budget(question), core.extract_min_price(question)
    if budget and not any(fl[k] for k in ("price_max", "price_target")):
        if re.search(r"ประมาณ|ราวๆ|ราว ๆ|แถว", question):
            fl["price_target"] = budget
        else:
            fl["price_max"] = budget
        added.append(f"งบ {budget:,}")
    if low and not fl["price_min"] and fl["price_max"]:
        fl["price_min"] = low
    ram = core.ram_condition(question) if hasattr(core, "ram_condition") else None
    if ram and not (fl["ram_gb"] or fl["ram_min"]):
        gb, at_least = ram
        fl["ram_min" if at_least else "ram_gb"] = gb
        added.append(f"แรม {gb}")
    terms = core.gpu_terms(question)
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


_RE_ADVICE = re.compile(r"แนะนำ|ตัวไหนดี|รุ่นไหนดี|อันไหนดี|คุ้ม|เหมาะ|ควรซื้อ|ช่วยเลือก")
SPEC_FILTERS = ("ram_gb", "ram_min", "gpu", "gpu_family", "cpu", "brand", "series", "screen")
_RE_LIST_Q = re.compile(r"(?:รุ่น|ตัว|เครื่อง|อะไร|ไหน)\s*(?:ไหน|อะไร)?\s*บ้าง|ทุกรุ่น|ทุกตัว|ทั้งหมด")


def _history_text(history: list) -> str:
    recent = history[-HISTORY_TURNS:]
    return "\n".join(f"{'ลูกค้า' if h['role'] == 'user' else 'ผู้ช่วย'}: {h['content'][:600]}" for h in recent) or "(ไม่มี)"


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


# เงื่อนไขที่ยังไม่มีตัวกรอง — บางเรื่องอยู่ในข้อความสเปก (ความละเอียดจอ, Gen ของ SSD, รุ่น Windows)
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
SEARCHABLE = "ยี่ห้อ ซีรีส์ ราคา CPU การ์ดจอ แรม และขนาดจอ"


def unsupported_terms(question: str) -> list[str]:
    """เรื่องที่ลูกค้าขอแต่ระบบยังคัดรุ่นให้ไม่ได้ (คืนชื่อเรื่องภาษาไทย ไม่ซ้ำ ตามลำดับในตาราง)"""
    q = question.lower()
    return [label for label, rx in UNSUPPORTED if rx.search(q)]


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
    added = _safety_net(plan, question) + check_plan(plan, question, history, last_plan)
    plan["unsupported"] = unsupported_terms(question)
    if plan["unsupported"]:
        added.append("ยังคัดตามเรื่องนี้ไม่ได้: " + ", ".join(plan["unsupported"]))
    if trace is not None:
        trace["understand"] = {"ms": round((time.perf_counter() - started) * 1000), "raw": raw_text[:1500],
                               "plan": plan, "added": added, "error": err}
    return plan


# ───────────────────────── 3. คัดเลือกจากข้อมูลจริง ─────────────────────────
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
    core = _core()
    if fl["price_max"] and lp["price"] > fl["price_max"]:
        return False
    if fl["price_min"] and lp["price"] < fl["price_min"]:
        return False
    if fl["ram_gb"] and lp["ram_gb"] != fl["ram_gb"]:
        return False
    if fl["ram_min"] and lp["ram_gb"] < fl["ram_min"]:
        return False
    if fl["gpu"] and not core.gpu_matches(lp["gpu"], re.sub(r"([a-z])(\d)", r"\1 \2", fl["gpu"].lower())):
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
    return True


def describe(fl: dict) -> str:
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
    if fl["price_min"] and fl["price_max"]:
        parts.append(f"ราคา {fl['price_min']:,}–{fl['price_max']:,} บาท")
    elif fl["price_max"]:
        parts.append(f"ราคาไม่เกิน {fl['price_max']:,} บาท")
    elif fl["price_min"]:
        parts.append(f"ราคาตั้งแต่ {fl['price_min']:,} บาท")
    if fl["price_target"]:
        parts.append(f"ราคาใกล้ {fl['price_target']:,} บาท")
    return " · ".join(parts)


def _price_note(lp: dict, fl: dict) -> str:
    if fl["price_max"]:
        diff = fl["price_max"] - lp["price"]
        return f"อยู่ในงบ เหลือ {diff:,} บาท" if diff >= 0 else f"เกินงบ {-diff:,} บาท"
    if fl["price_target"]:
        diff = lp["price"] - fl["price_target"]
        return "ตรงราคาที่ต้องการ" if diff == 0 else (f"แพงกว่าที่ต้องการ {diff:,} บาท" if diff > 0
                                                    else f"ถูกกว่าที่ต้องการ {-diff:,} บาท")
    return ""


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
        core = _core()
        t0 = time.perf_counter()
        q_vec = np.asarray(core.get_embedder().encode([plan["usage"] or question], normalize_embeddings=True))[0]
        vecs = doc_vectors()
        index = {lp["name"]: i for i, lp in enumerate(cat)}
        sims = {lp["name"]: float(vecs[index[lp["name"]]] @ q_vec) for lp in items}
        items.sort(key=lambda lp: -sims[lp["name"]])
        if trace is not None:
            trace["embed"] = {"model": core.EMBED_MODEL, "dims": int(q_vec.shape[0]),
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


# ───────────────────────── 4. ตอบ ─────────────────────────
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


def _list_line(plan: dict, n: int) -> str:
    """รูปแบบบรรทัดของคำตอบแบบรายชื่อ — รายชื่อสั้นใส่สเปกหลักครบ รายชื่อยาวใส่เฉพาะที่ลูกค้าถาม"""
    if plan["detail"] or n <= LIST_FULL_MAX:
        line = ("'- **ชื่อรุ่นเต็ม** — X บาท — CPU ... (ระดับ...) · GPU ... (ระดับ...) · แรม ... · ความจุ ... · จอ ...'")
        if plan["detail"]:
            line += " ต่อด้วยคะแนน CPU/GPU และจุดเด่นจากรีวิว 1 ข้อ"
        return line
    return f"'- **ชื่อรุ่นเต็ม** — X บาท — {_spec_hint(plan)}'"


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
    if fl["screen"]:
        bits.append("จอ")
    return ", ".join(bits or ["CPU", "การ์ดจอ", "แรม"])


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


def complete(provider: dict, messages: list, max_tokens: int, temperature: float = 0.1,
             json_mode: bool = False) -> tuple[str, dict, float]:
    """เรียก endpoint แบบ OpenAI ของ provider — คืน (ข้อความ, response JSON, วินาที)"""
    core = _core()
    body = {"model": provider["model"], "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    if json_mode:
        body["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json"}
    if provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"
    started = time.perf_counter()
    try:
        resp = requests.post(f"{provider['base_url']}/chat/completions", headers=headers, json=body, timeout=120)
        if json_mode and resp.status_code == 400:   # endpoint บางตัวไม่รองรับ response_format
            body.pop("response_format")
            resp = requests.post(f"{provider['base_url']}/chat/completions", headers=headers, json=body, timeout=120)
        resp.raise_for_status()
    except requests.RequestException as e:
        raise core.LLMError(f"{provider['label']} error: {e} — เช็ค API key หรือ endpoint") from e
    data = resp.json()
    return data["choices"][0]["message"].get("content") or "", data, time.perf_counter() - started


_RE_DOUBLE_KRAB = re.compile(r"ครับ(?:[ \t]*ครับ)+")


def tidy(text: str) -> str:
    """เก็บกวาดคำตอบ: "ครับครับ" -> "ครับ" """
    return _RE_DOUBLE_KRAB.sub("ครับ", text)


def complete_stream(provider: dict, messages: list, max_tokens: int, temperature: float = 0.1,
                    meta: dict | None = None):
    """เหมือน complete() แต่ทยอยคืนข้อความทีละท่อนระหว่างที่ AI พิมพ์ (stream)
    meta เก็บ usage / finish_reason / ความคิด / วินาที ไว้ให้โหมด debug หลังจบ"""
    core = _core()
    body = {"model": provider["model"], "messages": messages, "temperature": temperature, "max_tokens": max_tokens,
            "stream": True, "stream_options": {"include_usage": True}}
    headers = {"Content-Type": "application/json"}
    if provider.get("api_key"):
        headers["Authorization"] = f"Bearer {provider['api_key']}"
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
        raise core.LLMError(f"{provider['label']} error: {e} — เช็ค API key หรือ endpoint") from e
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
            raise core.LLMError(f"{provider['label']} error: {e}") from e
    if meta is not None:
        meta.update({"seconds": time.perf_counter() - started, "usage": usage,
                     "finish_reason": finish, "thinking": "".join(thinking)})


def respond(plan: dict, question: str, history: list, provider: dict, trace: dict | None = None) -> str:
    """ตอบแบบรอทั้งก้อน (/chat) — ใช้ขั้นตอนเดียวกับแบบสตรีม"""
    result = {}
    for _ in respond_stream(plan, question, history, provider, trace, result):
        pass
    return result["answer"]


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
    core = _core()
    meta, raw, shown = {}, "", ""
    if lead:
        yield {"type": "delta", "text": lead}
        shown = lead
    for piece in complete_stream(provider, [{"role": "system", "content": ANSWER_RULES},
                                            {"role": "user", "content": prompt}], max_tokens=max_tokens, meta=meta):
        raw += piece
        text = lead + tidy(core.visible_answer(raw))
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
                        "max_tokens": max_tokens, **core.llm_trace(data, raw, meta.get("seconds", 0))}
    result["answer"] = lead + (tidy(core.strip_think(raw)) or core.EMPTY_ANSWER)


def memory(plan: dict) -> dict:
    """เงื่อนไขที่จำไว้ใช้กับข้อความถัดไป (เก็บใน session cookie — ต้องเล็ก)"""
    if plan["intent"] in TEMPLATED or plan["intent"] == "overview":
        return {}
    return {"intent": plan["intent"], "filters": {k: v for k, v in plan["filters"].items() if v},
            "models": plan["models"][:4], "usage": plan["usage"]}
