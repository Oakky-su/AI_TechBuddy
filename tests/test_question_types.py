"""ชุดทดสอบหลักของ Tech Buddy — ถามแบบที่ลูกค้าถามจริง แล้วตรวจคำตอบกับ laptops.json อัตโนมัติ

หมวด: ภาพรวมร้าน / ถูก-แพงสุด / กรองสเปก (การ์ดจอ แรม CPU ยี่ห้อ จอ) / จัดอันดับคะแนน / ถามสเปกรุ่น /
แนะนำตามงบ / เทียบรุ่น / ถามต่อหลายข้อความ / ปฏิเสธ (นอกเรื่อง ยี่ห้อที่ไม่มี ความรู้ทั่วไป)

ใช้กับแอปที่รันอยู่:
    python tests/test_question_types.py [URL]          # ค่าเริ่มต้น http://127.0.0.1:5000

ตัวเลือก (env):
    QT_GAP=3.5      วินาทีที่เว้นก่อนแต่ละข้อความ (เว็บสาธารณะจำกัด 20 ข้อความ/นาที — เครื่องทดสอบตั้ง 0 ได้)
    QT_ENGINE=pi|classic   เลือกโหมดตอบ (ไม่ตั้ง = ค่าเริ่มต้นของเซิร์ฟเวอร์)
    QT_ONLY=แท็ก,แท็ก  รันเฉพาะเคสที่ระบุ (ไว้ไล่แก้ทีละข้อ)
    QT_OUT=<ไฟล์>    บันทึกคำตอบเต็ม + ผลตรวจเป็น JSON

ทุกเคสเริ่มแชทใหม่ เคสที่เป็น list ของข้อความ = ถามต่อกันในแชทเดียว ตรวจคำตอบสุดท้าย
นอกจากตัวตรวจของแต่ละเคส ทุกคำตอบต้องผ่านการตรวจรวม:
  - คะแนน CPU/GPU และราคาที่อ้างต้องมีจริงในข้อมูล (หรือคำนวณจากตัวเลขในข้อมูล/คำถามได้)
  - ห้ามแนะนำยี่ห้อที่ไม่มีในร้าน
  - คำถามที่บอกเงื่อนไขชัดแล้ว ห้ามตอบด้วยการถามกลับอย่างเดียว ("ยังไม่ได้ระบุงบ/การใช้งาน")
"""
import json
import os
import re
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:5000").rstrip("/")
GAP = float(os.getenv("QT_GAP", "3.5"))
ENGINE = os.getenv("QT_ENGINE", "").strip()
ONLY = {t.strip() for t in os.getenv("QT_ONLY", "").split(",") if t.strip()}
LAPTOPS = json.load(open(os.getenv("LAPTOPS_PATH") or os.path.join(BASE, "laptops.json"), encoding="utf-8"))

# ---- อ่านข้อมูลจริง ----
name = lambda lp: lp["ชื่อสินค้า"]
price = lambda lp: lp["ราคา"]
cpu = lambda lp: lp["benchmark"]["CPU Score"]
gpu = lambda lp: lp["benchmark"]["GPU Score"]
gpu_name = lambda lp: lp["spec"]["GPU"].lower()
brand = lambda lp: name(lp).split()[0]


def ram_gb(lp) -> int:
    m = re.search(r"(\d+)\s*GB", lp["spec"]["RAM"])
    return int(m.group(1)) if m else 0


def code(n: str) -> str:
    """รหัสรุ่น เช่น "ANV15-51-56QE" — คำตอบมักเขียนชื่อย่อ แต่รหัสรุ่นมักครบ"""
    return next((t for t in n.split() if "-" in t), n)


def mentioned(ans: str, lp) -> bool:
    a, n = ans.lower(), name(lp).lower()
    c = code(name(lp)).lower()
    return n in a or (len(c) >= 6 and c in a)


def named(ans: str) -> list:
    return [lp for lp in LAPTOPS if mentioned(ans, lp)]


def has_num(ans: str, n: int) -> bool:
    return f"{n:,}" in ans or re.search(rf"(?<![\d,]){n}(?![\d,])", ans) is not None


def find(prefix: str):
    return next(lp for lp in LAPTOPS if name(lp).startswith(prefix))


def pool(pred) -> list:
    return sorted((lp for lp in LAPTOPS if pred(lp)), key=price)


# ---- ตรวจรวมทุกคำตอบ ----
REAL_PRICES = {price(lp) for lp in LAPTOPS}
REAL_SCORES = {cpu(lp) for lp in LAPTOPS} | {gpu(lp) for lp in LAPTOPS}
BRANDS = sorted({brand(lp) for lp in LAPTOPS})
OUTSIDE_BRANDS = ["Dell", "MacBook", "Apple", "ThinkPad", "Razer", "Alienware", "Surface", "Samsung Galaxy Book"]
_SCORE_RE = re.compile(r"(?:CPU|GPU)(?:\s*Score)?\s*[:：]?\s*(?:คะแนน\s*)?([\d,]{3,7})|คะแนน\s*(?:CPU|GPU)?\s*[:：]?\s*([\d,]{3,7})")
_BAHT_RE = re.compile(r"([\d,]{4,})\s*บาท")
DENY = ["ไม่มี", "ไม่พบ", "ตอบไม่ได้", "ช่วยได้เฉพาะ", "ไม่สามารถ", "ไม่ครอบคลุม"]
_ASKBACK_RE = re.compile(r"ยังไม่ได้ระบุ|ยังไม่ได้บอก|ไว้ใช้ทำอะไร|ใช้งานด้านไหน|งบประมาณประมาณเท่าไหร่|บอกงบ")


def fake_scores(ans: str) -> list[int]:
    out = []
    for m in _SCORE_RE.finditer(ans):
        n = int((m.group(1) or m.group(2)).replace(",", ""))
        if n >= 1000 and n not in REAL_SCORES:
            out.append(n)
    return out


def fake_prices(ans: str, question: str) -> list[int]:
    """ราคา (บาท) ที่ไม่ใช่ราคาจริง/ตัวเลขในคำถาม/ส่วนต่างที่คำนวณได้ = แต่งขึ้น"""
    asked = {int(x.replace(",", "")) for x in re.findall(r"\d[\d,]{3,}", question)}
    asked |= {int(x) * 1000 for x in re.findall(r"(\d+)\s*k", question, re.I)}
    ok = REAL_PRICES | asked
    out = []
    for raw in _BAHT_RE.findall(ans):
        n = int(raw.replace(",", ""))
        if n < 1000 or n in ok:
            continue
        if any(abs(p - q) == n for p in REAL_PRICES for q in asked | REAL_PRICES):
            continue
        out.append(n)
    return out


_REFUSE_RE = re.compile(r"ยังไม่ได้ระบุ|ยังไม่ได้บอก|ยังไม่สามารถแนะนำ|ไม่สามารถแนะนำ")


def askback_only(ans: str) -> bool:
    """ปฏิเสธจะตอบแล้วถามกลับ ("ยังไม่ได้ระบุงบ...") หรือคำตอบสั้นที่มีแต่คำถามกลับ ไม่มีข้อมูลรุ่นใด ๆ
    (ตอบครบแล้วปิดท้ายด้วยคำถามชวนคุยต่อ = ไม่นับ)"""
    if named(ans):
        return False
    return bool(_REFUSE_RE.search(ans)) or (bool(_ASKBACK_RE.search(ans)) and len(ans) < 120)


def denies(ans: str) -> bool:
    return any(w in ans for w in DENY)


# ---- ตัวตรวจแต่ละแบบ ----
def c_has(*needles, label=""):
    def check(ans):
        missing = [n for n in needles
                   if (not has_num(ans, n) if isinstance(n, int) else n.lower() not in ans.lower())]
        return not missing, label or f"ต้องมี {list(needles)}" + (f" · ขาด {missing}" if missing else "")
    return check


def c_model(lp, *extra):
    """ต้องเอ่ยรุ่นนี้ + ตัวเลข/ข้อความเพิ่มเติม (ราคา คะแนน ชื่อ CPU ...)"""
    def check(ans):
        missing = [x for x in extra if (not has_num(ans, x) if isinstance(x, int) else x.lower() not in ans.lower())]
        ok = mentioned(ans, lp) and not missing
        return ok, f"ต้องเจอ {name(lp)}" + (f" + {list(extra)}" if extra else "") + (
            f" · ขาด {missing}" if missing else "") + ("" if mentioned(ans, lp) else " · ไม่ได้เอ่ยรุ่นนี้")
    return check


def c_any_of(lps, *extra):
    """ข้อไหนในกลุ่มนี้ก็ได้ (เช่น คะแนนสูงสุดเท่ากันหลายรุ่น)"""
    def check(ans):
        ok = any(mentioned(ans, lp) for lp in lps) and all(has_num(ans, x) for x in extra)
        return ok, f"ต้องเจอ {' หรือ '.join(name(lp) for lp in lps)}" + (f" + {list(extra)}" if extra else "")
    return check


def c_all(expected, cond, label, order_desc=False):
    """ต้องไล่ครบทุกรุ่นที่ตรง และทุกรุ่นที่เอ่ยต้องตรงเงื่อนไข (ลำดับราคาตามที่ขอ ถ้าระบุ)"""
    def check(ans):
        got = named(ans)
        hit = [lp for lp in expected if lp in got]
        bad = [name(lp) for lp in got if not cond(lp)]
        ok = len(hit) == len(expected) and not bad and bool(expected)
        note = f"เจอ {len(hit)}/{len(expected)} รุ่น ({label})"
        if bad:
            note += f" · มีรุ่นที่ไม่ตรง: {', '.join(bad[:3])}"
        if order_desc and len(expected) > 1:
            a = ans.lower()
            hi, lo = name(expected[-1]).lower(), name(expected[0]).lower()
            if not (hi in a and lo in a and a.index(hi) < a.index(lo)):
                ok, note = False, note + " · ไม่ได้เรียงจากแพงไปถูก"
        return ok, note
    return check


def c_within(cond, label, need=1):
    """ทุกรุ่นที่เอ่ยต้องตรงเงื่อนไข และต้องเอ่ยอย่างน้อย need รุ่น"""
    def check(ans):
        got = named(ans)
        bad = [name(lp) for lp in got if not cond(lp)]
        ok = len(got) >= need and not bad
        note = f"{len(got)} รุ่น ({label})" + (f" · ไม่ตรง: {', '.join(bad[:3])}" if bad else "")
        return ok, note
    return check


def c_budget(budget, low=0, need=1):
    def check(ans):
        ok, note = c_within(lambda lp: low <= price(lp) <= budget,
                            f"ราคา {low:,}–{budget:,}" if low else f"ไม่เกิน {budget:,}", need)(ans)
        left = {int(x.replace(",", "")) for x in re.findall(r"เหลือ(?:อีก)?\s*([\d,]+)", ans)}
        wrong = sorted(left - {budget - price(lp) for lp in named(ans)})
        if wrong:
            ok, note = False, note + f" · ส่วนต่างงบไม่ตรง {wrong}"
        return ok, note
    return check


def c_brands(ans):
    got = [b for b in BRANDS if b.lower() in ans.lower()]
    return len(got) >= len(BRANDS) - 1, f"ยี่ห้อในร้าน {len(BRANDS)} ยี่ห้อ เอ่ยถึง {len(got)}: {got}"


def c_deny(ans):
    made_up = re.search(r"MacBook[^\n]{0,30}ราคา\s*[\d,]{4,}", ans) is not None
    return denies(ans) and not made_up, "ต้องบอกว่าไม่มีในข้อมูล และไม่แต่งราคาให้"


def c_offtopic(ans):
    return denies(ans) and "?" in ans and not named(ans), "ไม่ตอบเรื่องอื่น และถามกลับเรื่องโน้ตบุ๊ค"


def c_general(ans):
    return denies(ans) and len(ans) < 400, "ความรู้ทั่วไปนอกข้อมูล: ต้องบอกว่าตอบไม่ได้ ไม่อธิบายยาว"


def c_greeting(ans):
    return not named(ans) and "?" in ans, "ทักกลับ แล้วถามงบ/การใช้งาน ไม่ยัดสินค้า"


def c_legion_over(ans):
    says_missing = re.search(r"Legion 5[^\n]{0,40}(ไม่มีในข้อมูล|ไม่มีข้อมูล|ไม่พบ)", ans) is not None
    return ("Legion 5" in ans and not says_missing), "Legion 5 มีแต่เกินงบ — ต้องบอกว่ามีแต่เกินงบ"


# ---- คำตอบที่ถูกต้อง (คำนวณจากข้อมูลจริง) ----
CHEAPEST, PRICIEST = min(LAPTOPS, key=price), max(LAPTOPS, key=price)
CHEAPEST_RTX = min((lp for lp in LAPTOPS if "rtx" in gpu_name(lp)), key=price)
CHEAPEST_32 = min((lp for lp in LAPTOPS if ram_gb(lp) >= 32), key=price)
TOP_CPU = max(LAPTOPS, key=cpu)
TOP_GPU_ALL = [lp for lp in LAPTOPS if gpu(lp) == max(gpu(x) for x in LAPTOPS)]
TOP_CPU_30K = max((lp for lp in LAPTOPS if price(lp) <= 30_000), key=cpu)
TOP_GPU_40K = max((lp for lp in LAPTOPS if price(lp) <= 40_000), key=gpu)
RTX4060_40K = pool(lambda lp: re.search(r"rtx 4060(?!\s*ti)", gpu_name(lp)) and price(lp) <= 40_000)
RTX5070 = pool(lambda lp: re.search(r"rtx 5070(?!\s*ti)", gpu_name(lp)))
RAM32UP = pool(lambda lp: ram_gb(lp) >= 32)
RAM24 = pool(lambda lp: ram_gb(lp) == 24)
RAM16_20K = pool(lambda lp: ram_gb(lp) == 16 and price(lp) <= 20_000)
RYZEN7_35K = pool(lambda lp: "ryzen 7" in lp["spec"]["CPU"].lower() and price(lp) <= 35_000)
LOQ = pool(lambda lp: name(lp).startswith("Lenovo LOQ"))
SCREEN16_30K = pool(lambda lp: re.match(r"16(\.0)?\s*inch", lp["spec"]["DISPLAY"]) and price(lp) <= 30_000)
RAM24_NEAR_24K = [lp for lp in RAM24 if price(lp) == min(price(x) for x in RAM24)]
NITRO, VICTUS16, HPAI15 = find("Acer Nitro V 15 ANV15-51"), find("HP Victus 16-r0356TX"), find("HP AI 15-fd2040TU")
RANGE_24K = lambda lp: 21_000 <= price(lp) <= 27_000

# (หมวด, แท็ก, ข้อความ, ตัวตรวจ, ห้ามถามกลับอย่างเดียว)
CASES = [
    ("ภาพรวม", "จำนวนรุ่น", "ร้านมีโน้ตบุ๊คกี่รุ่น ราคาเริ่มต้นเท่าไหร่", c_has(len(LAPTOPS), price(CHEAPEST)), True),
    ("ภาพรวม", "ยี่ห้อ", "มียี่ห้ออะไรบ้าง", c_brands, True),
    ("ภาพรวม", "ราคาสูงสุด", "ราคาแพงสุดในร้านเท่าไหร่", c_has(price(PRICIEST)), True),

    ("ถูก/แพงสุด", "ถูกสุด", "รุ่นไหนถูกที่สุด", c_model(CHEAPEST, price(CHEAPEST)), True),
    ("ถูก/แพงสุด", "แพงสุด", "รุ่นไหนแพงที่สุด", c_model(PRICIEST, price(PRICIEST)), True),
    ("ถูก/แพงสุด", "RTX ถูกสุด", "รุ่นที่มีการ์ดจอ RTX ที่ถูกที่สุดคือรุ่นไหน", c_model(CHEAPEST_RTX, price(CHEAPEST_RTX)), True),
    ("ถูก/แพงสุด", "แรม 32 ถูกสุด", "แรม 32GB ขึ้นไปตัวไหนถูกที่สุด", c_model(CHEAPEST_32, price(CHEAPEST_32)), True),

    ("กรองสเปก", "RTX 4060 ในงบ", "RTX 4060 มีรุ่นไหนบ้าง งบไม่เกิน 40,000",
     c_all(RTX4060_40K, lambda lp: lp in RTX4060_40K, "RTX 4060 ≤40,000"), True),
    ("กรองสเปก", "RTX 5070", "มีรุ่นไหนใช้ RTX 5070 บ้าง", c_all(RTX5070, lambda lp: lp in RTX5070, "RTX 5070 ไม่ใช่ Ti"), True),
    ("กรองสเปก", "แรม 32 ขึ้นไป", "แรม 32GB ขึ้นไปมีรุ่นไหนบ้าง", c_all(RAM32UP, lambda lp: ram_gb(lp) >= 32, "แรม ≥32GB"), True),
    ("กรองสเปก", "แรม 24 list", "แรม 24 มีตัวไหนบ้าง list ออกมาหน่อย", c_all(RAM24, lambda lp: ram_gb(lp) == 24, "แรม 24GB"), True),
    ("กรองสเปก", "แรม 24 สเปกละเอียด", "แรม 24 ขอสเปคทุกตัวแบบละเอียด", c_all(RAM24, lambda lp: ram_gb(lp) == 24, "แรม 24GB"), True),
    ("กรองสเปก", "แรม 24 เรียงแพงไปถูก", "เรียงจากราคามากไปน้อยให้หน่อยตัว ram 24",
     c_all(RAM24, lambda lp: ram_gb(lp) == 24, "แรม 24GB", order_desc=True), True),
    ("กรองสเปก", "แรม 16 ในงบ", "แรม 16GB งบไม่เกิน 20,000 มีรุ่นไหนบ้าง",
     c_all(RAM16_20K, lambda lp: ram_gb(lp) == 16 and price(lp) <= 20_000, "แรม 16GB ≤20,000"), True),
    ("กรองสเปก", "Ryzen 7 ในงบ", "มี Ryzen 7 รุ่นไหนบ้าง งบไม่เกิน 35,000",
     c_all(RYZEN7_35K, lambda lp: lp in RYZEN7_35K, "Ryzen 7 ≤35,000"), True),
    ("กรองสเปก", "Lenovo LOQ", "Lenovo LOQ มีกี่รุ่น ราคาเท่าไหร่บ้าง", c_all(LOQ, lambda lp: lp in LOQ, "Lenovo LOQ"), True),
    ("กรองสเปก", "จอ 16 นิ้วในงบ", "จอ 16 นิ้ว งบไม่เกิน 30,000 มีรุ่นไหนบ้าง",
     c_all(SCREEN16_30K, lambda lp: lp in SCREEN16_30K, "จอ 16 นิ้ว ≤30,000"), True),

    ("จัดอันดับ", "CPU สูงสุด", "รุ่นไหนคะแนน CPU สูงที่สุด", c_model(TOP_CPU, cpu(TOP_CPU)), True),
    ("จัดอันดับ", "GPU สูงสุด", "รุ่นไหนคะแนน GPU สูงที่สุด", c_any_of(TOP_GPU_ALL, gpu(TOP_GPU_ALL[0])), True),
    ("จัดอันดับ", "CPU แรงสุดในงบ", "งบ 30,000 รุ่นไหน CPU แรงที่สุด", c_model(TOP_CPU_30K, cpu(TOP_CPU_30K)), True),
    ("จัดอันดับ", "การ์ดจอแรงสุดในงบ", "งบ 40,000 การ์ดจอแรงสุดคือรุ่นไหน", c_model(TOP_GPU_40K, gpu(TOP_GPU_40K)), True),

    ("ถามสเปกรุ่น", "คะแนนรุ่น", f"{name(NITRO)} คะแนน CPU กับ GPU เท่าไหร่", c_model(NITRO, cpu(NITRO), gpu(NITRO)), True),
    ("ถามสเปกรุ่น", "CPU+แรมรุ่น", f"{name(VICTUS16)} ใช้ CPU อะไร แรมเท่าไหร่",
     c_model(VICTUS16, "i5-13500HX", ram_gb(VICTUS16)), True),
    ("ถามสเปกรุ่น", "สเปกเต็มรุ่น", f"{name(HPAI15)} สเปคเป็นยังไง",
     c_model(HPAI15, "Core Ultra 7 255U", price(HPAI15)), True),

    ("แนะนำ", "เล่นเกมตามงบ", "โน้ตบุ๊คเล่นเกม งบ 25,000", c_budget(25_000), True),
    ("แนะนำ", "ช่วงราคา", "ทำงานกราฟิก ราคาระหว่าง 30,000 ถึง 40,000", c_budget(40_000, low=30_000), True),
    ("แนะนำ", "งบต่ำเกิน", "งบ 10,000 มีรุ่นไหนบ้าง",
     c_model(CHEAPEST, price(CHEAPEST) - 10_000), True),
    ("แนะนำ", "ขอ 3 รุ่น", "ขอ 3 รุ่น ตัดต่อวิดีโอ งบ 40,000", c_budget(40_000, need=3), True),
    ("แนะนำ", "ราคาประมาณ", "อยากได้ราคาประมาณ 24000 บาท", c_within(RANGE_24K, "ราคา 21,000–27,000"), True),
    ("แนะนำ", "จากรีวิว", "รุ่นไหนแบตอึด พกพาสะดวก งบ 30,000", c_budget(30_000), True),

    ("เทียบรุ่น", "เทียบ 2 รุ่น", f"เทียบ {name(NITRO)} กับ {name(VICTUS16)}",
     c_has(price(NITRO), price(VICTUS16), label=f"ต้องมีราคาทั้งสองรุ่น {price(NITRO):,} / {price(VICTUS16):,}"), True),
    ("เทียบรุ่น", "ส่วนต่างราคา", f"{name(NITRO)} กับ {name(VICTUS16)} ราคาต่างกันกี่บาท",
     c_has(abs(price(NITRO) - price(VICTUS16))), True),
    ("เทียบรุ่น", "รุ่นที่เกินงบ", "เทียบ Legion 5 กับ HP Victus 16 งบ 45000", c_legion_over, True),

    ("ถามต่อ", "จำงบ", ["เล่นเกม งบ 30,000", "มีรุ่นอื่นอีกไหม"], c_budget(30_000), True),
    ("ถามต่อ", "แรม 24 + ราคา 24000", ["แรม 24 มีตัวไหนบ้าง", "แล้วราคา 24000 บาทหละ"],
     c_within(lambda lp: ram_gb(lp) == 24 and price(lp) <= 30_000, "แรม 24GB ราคาใกล้ 24,000"), True),
    ("ถามต่อ", "แรม 24 + ราคาประมาณ", ["แรม 24 มีตัวไหนบ้าง", "อยากได้ราคาประมาณ 24000 บาทอะ"],
     c_within(lambda lp: ram_gb(lp) == 24 and price(lp) <= 30_000, "แรม 24GB ราคาใกล้ 24,000"), True),
    ("ถามต่อ", "ตัวที่ถูกกว่า", [f"เทียบ {name(NITRO)} กับ {name(VICTUS16)}", "ตัวที่ถูกกว่าใช้ CPU อะไร"],
     c_has("i5-13420H"), True),
    ("ถามต่อ", "อันไหนถูกสุด", ["แรม 32GB ขึ้นไปมีรุ่นไหนบ้าง", "อันไหนถูกสุด"], c_model(CHEAPEST_32, price(CHEAPEST_32)), True),

    ("ปฏิเสธ", "ยี่ห้อที่ไม่มี", "MacBook Air M3 ราคาเท่าไหร่", c_deny, False),
    ("ปฏิเสธ", "นอกเรื่อง", "วันนี้ฝนจะตกไหม", c_offtopic, False),
    ("ปฏิเสธ", "ความรู้ทั่วไป", "TDP คืออะไร", c_general, False),
    ("ปฏิเสธ", "ทักทาย", "สวัสดีครับ", c_greeting, False),
]


def ask(s: requests.Session, q: str) -> dict:
    time.sleep(GAP)
    r = s.post(f"{URL}/chat", json={"message": q, **({"engine": ENGINE} if ENGINE else {})}, timeout=180)
    data = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
    return {"status": r.status_code, "answer": data.get("message") or "", "error": data.get("error"),
            "meta": data.get("meta") or {}}


def run_case(s, group, tag, msgs, check, no_askback):
    s.post(f"{URL}/new_chat", timeout=30)
    msgs = msgs if isinstance(msgs, list) else [msgs]
    for one in msgs:
        res = ask(s, one)
    ans = res["answer"]
    if res["error"] or not ans:
        return False, f"ไม่ได้คำตอบ: HTTP {res['status']} {res['error']}", res
    ok, note = check(ans)
    extra = []
    if fakes := fake_scores(ans):
        extra.append(f"คะแนนที่ไม่มีในข้อมูล {fakes}")
    if fakes := fake_prices(ans, " ".join(msgs)):
        extra.append(f"ราคาที่ไม่มีในข้อมูล {fakes}")
    if outside := [b for b in OUTSIDE_BRANDS if b.lower() in ans.lower()] if no_askback else []:
        extra.append(f"เอ่ยยี่ห้อที่ไม่มีในร้าน {outside}")
    if no_askback and askback_only(ans):
        extra.append("ถามกลับแทนการตอบ")
    if extra:
        ok, note = False, note + " · " + " · ".join(extra)
    return ok, note, res


def main():
    s = requests.Session()
    results = []
    cases = [c for c in CASES if not ONLY or c[1] in ONLY]
    for group, tag, msgs, check, no_askback in cases:
        ok, note, res = run_case(s, group, tag, msgs, check, no_askback)
        meta = res["meta"]
        q = msgs if isinstance(msgs, str) else " → ".join(msgs)
        results.append({"group": group, "tag": tag, "question": q, "ok": ok, "note": note,
                        "answer": res["answer"], "elapsed": meta.get("elapsed"), "engine": meta.get("engine")})
        print(f"{'✅' if ok else '❌'} [{group} · {tag}] {q}\n     {note} · {meta.get('elapsed')} วิ", flush=True)

    passed = sum(r["ok"] for r in results)
    secs = [r["elapsed"] for r in results if r["elapsed"]]
    print(f"\n{'=' * 70}\nผ่าน {passed} / {len(results)} เคส"
          + (f" · เฉลี่ย {sum(secs) / len(secs):.1f} วิ/คำตอบ" if secs else ""))
    for g in dict.fromkeys(r["group"] for r in results):
        rs = [r for r in results if r["group"] == g]
        print(f"  {g}: {sum(r['ok'] for r in rs)}/{len(rs)}")
    if os.getenv("QT_OUT"):
        with open(os.environ["QT_OUT"], "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=1)
    sys.exit(0 if passed == len(results) else 1)


if __name__ == "__main__":
    main()
