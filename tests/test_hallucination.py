# ทดสอบการแต่งข้อมูล (hallucination) — เจาะคำถามที่เคยทำให้ context ไม่มีราคา/สเปก
# รัน: python tests/test_hallucination.py   (ต้องมีแอปรันที่ localhost:5000)
#
# นิยามของ "แต่งข้อมูล" ที่ใช้ในไฟล์นี้:
#   AI อ้างตัวเลข (ราคา / คะแนน CPU / คะแนน GPU) ที่ "ไม่ตรงกับรุ่นใดเลย" ที่มันเอ่ยถึง
# ไม่ใช่ "ไม่ยอมอ้างตัวเลข" — การบอกว่าไม่มีข้อมูลถือว่าถูกต้องตาม system prompt
import json
import re
import sys
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent.parent
URL = "http://localhost:5000"
LAPTOPS = json.loads((BASE / "laptops.json").read_text(encoding="utf-8"))

BY_CODE = {}
for lp in LAPTOPS:
    toks = [t for t in lp["ชื่อสินค้า"].split() if re.search(r"\d", t) and "-" in t]
    if toks:
        BY_CODE[toks[-1]] = lp

QUESTIONS = [
    "แบตอึด พกพาสะดวก งบ 30000",
    "โน้ตบุ๊คทำงานเอกสาร งบ 20k",
    "เครื่องร้อนไหม รุ่นไหนระบายความร้อนดี",
    "ขอรุ่นที่ราคาถูกที่สุดในร้าน",
]


def num(s):
    return int(s.replace(",", ""))


def cited(pattern, ans):
    """ดึงตัวเลขที่ AI 'อ้างอย่างเจาะจง' ตามรูปแบบที่กำหนด
    ใช้การจับแบบเจาะจงเพื่อไม่ให้ไปโดนเลขรุ่น CPU เช่น 'Core i7-13620H'"""
    return [num(m) for m in re.findall(pattern, ans)]


print("=" * 80)
print("HALLUCINATION TESTS — AI อ้างตัวเลขที่ไม่มีในฐานข้อมูลหรือไม่")
print("=" * 80)

fails = 0
for q in QUESTIONS:
    s = requests.Session()
    r = s.post(f"{URL}/chat", json={"message": q}, timeout=180)
    ans = r.json().get("message", "")

    mentioned = [lp for code, lp in BY_CODE.items() if code in ans]
    real_prices = {lp["ราคา"] for lp in mentioned}
    real_cpus = {lp["benchmark"]["CPU Score"] for lp in mentioned}
    real_gpus = {lp["benchmark"]["GPU Score"] for lp in mentioned}

    cited_price = cited(r"ราคา\s*([\d,]{5,})\s*บาท", ans)
    cited_cpu = cited(r"(?:คะแนน\s*CPU|CPU\s*Score)[^\d]{0,15}([\d,]{4,})", ans)
    cited_gpu = cited(r"(?:คะแนน\s*GPU|GPU\s*Score)[^\d]{0,15}([\d,]{4,})", ans)

    problems = []
    for v in cited_price:
        if v not in real_prices:
            problems.append(f"อ้างราคา {v:,} ซึ่งไม่ตรงกับรุ่นใดที่เอ่ยถึง {sorted(real_prices)}")
    for v in cited_cpu:
        if v not in real_cpus:
            problems.append(f"อ้างคะแนน CPU {v:,} ซึ่งไม่ตรงกับรุ่นใดที่เอ่ยถึง {sorted(real_cpus)}")
    for v in cited_gpu:
        if v not in real_gpus:
            problems.append(f"อ้างคะแนน GPU {v:,} ซึ่งไม่ตรงกับรุ่นใดที่เอ่ยถึง {sorted(real_gpus)}")

    print(f"\n--- ถาม: {q} ---")
    print(f"    รุ่นที่เอ่ยถึง: {[lp['ชื่อสินค้า'] for lp in mentioned] or '-'}")
    print(f"    อ้างราคา={cited_price or '-'} | อ้าง CPU={cited_cpu or '-'} | อ้าง GPU={cited_gpu or '-'}")
    if problems:
        fails += 1
        print(f"    [FAIL] {'; '.join(problems)}")
    else:
        print("    [PASS] ทุกตัวเลขที่อ้างตรงกับฐานข้อมูล")

print("\n" + "=" * 80)
print(f"HALLUCINATION: แต่งตัวเลข {fails}/{len(QUESTIONS)} คำถาม")
print("=" * 80)
sys.exit(1 if fails else 0)
