"""รันชุดคำถามจาก rag_notebook_test_questions.xlsx แล้วตรวจว่าบอทตอบจาก laptops.json จริงไหม

เกณฑ์ตรวจหลัก: คำตอบต้องไม่เอาความรู้นอกฐานข้อมูลมาใช้
  - ห้ามเอ่ยยี่ห้อ/รุ่นที่ไม่มีในร้าน (Dell, MacBook, ThinkPad, Yoga Slim, Spectre)
    ในลักษณะ "ให้ข้อมูลสเปก/ราคา" — เอ่ยเพื่อบอกว่า "ไม่มีในร้าน" ถือว่าถูกต้อง
  - ตัวเลขราคา/คะแนนที่อ้าง ต้องตรงกับฐานข้อมูล

ใช้: python tests/run_excel_questions.py
"""
import json
import os
import re
import sys
import time

import requests

sys.stdout.reconfigure(encoding="utf-8")

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
URL = "http://localhost:5000"
QFILE = os.path.join(os.environ.get("TEMP", "."), "questions.json")
OUT = os.path.join(os.environ.get("TEMP", "."), "excel_results.json")

LAPTOPS = json.loads(open(os.path.join(BASE, "laptops.json"), encoding="utf-8").read())
REAL_PRICES = {lp["ราคา"] for lp in LAPTOPS}
REAL_CPU = {lp["benchmark"]["CPU Score"] for lp in LAPTOPS}
REAL_GPU = {lp["benchmark"]["GPU Score"] for lp in LAPTOPS}
BRANDS_IN_STORE = {lp["ชื่อสินค้า"].split()[0].lower() for lp in LAPTOPS}

# ยี่ห้อ/ตระกูลที่ "ไม่มีในร้าน" — ถ้าบอทให้สเปกหรือราคาของพวกนี้ = ใช้ความรู้ LLM
OUTSIDE = ["macbook", "dell", "xps", "thinkpad", "spectre", "yoga slim",
           "surface", "razer", "alienware", "apple", "imac"]

# คำที่แสดงว่าบอทยอมรับว่าไม่มีข้อมูล (ถือว่าตอบถูกต้องตามหลัก RAG)
DENIES = ["ไม่มีข้อมูล", "ไม่มีรุ่น", "ไม่มีในร้าน", "ไม่พบ", "ไม่แน่ใจ",
          "ไม่มีโน้ตบุ๊ค", "ไม่สามารถ", "ไม่ได้ระบุ", "ไม่มีรายละเอียด", "ในระบบไม่มี",
          "ไม่ครอบคลุม", "ไม่มีเรื่องนี้", "ตอบไม่ได้"]

# คำถามที่ต้องถามต่อจากคำถามก่อนหน้า (ทดสอบบริบทการสนทนา)
FOLLOW_UP = {"24": "แนะนำโน้ตบุ๊คทำงานทั่วไป งบ 25,000",
             "25": "แนะนำโน้ตบุ๊คเล่นเกม งบ 30,000 สัก 2 รุ่น"}


def cited_numbers(ans, pattern):
    return [int(m.replace(",", "")) for m in re.findall(pattern, ans)]


# ข้อที่ "ต้องปฏิเสธ" เพราะตอบจาก laptops.json ไม่ได้
MUST_DENY = {
    "2": "MacBook ไม่มีในข้อมูล",
    "3": "Yoga Slim 7 ไม่มีในข้อมูล + ไม่มีฟิลด์น้ำหนัก",
    "7": "Dell XPS / HP Spectre ไม่มีในข้อมูล",
    "18": "ความรู้ทั่วไป (TDP)",
    "19": "ความรู้ทั่วไป (ทำไมแบตหมดเร็ว)",
    "20": "ความรู้ทั่วไป (วิธีดูว่าอัปเกรด RAM ได้)",
    "21": "MacBook ไม่มีในข้อมูล",
    "33": "ThinkPad ไม่มีในข้อมูล",
    "34": "MacBook ไม่มีในข้อมูล",
    "35": "ไม่มีฟิลด์ประเทศผลิต",
    "36": "มือถือ อยู่นอกขอบเขต",
    "38": "ไม่มีฟิลด์ประกัน",
    "39": "คาดการณ์อนาคต ไม่มีในข้อมูล",
    "40": "วิธีปลดล็อก BIOS อยู่นอกขอบเขต",
}

# เอ่ยยี่ห้อนอกรายการพร้อม "ให้ข้อมูล" = ใช้ความรู้ LLM (ต่างจากเอ่ยเพื่อปฏิเสธ)
_DESCRIBES = re.compile(
    r"(macbook|dell|xps|thinkpad|spectre|yoga slim|surface)[^。\n]{0,60}"
    r"(ราคา|ชิป|cpu|gpu|ram|รุ่นที่เปิดตัว|เปิดตัว|น้ำหนัก|แนะนำให้ดู|ลองดู)",
    re.I,
)


def check(ans, question, no=""):
    """คืน (ผ่าน/ไม่ผ่าน, เหตุผล)"""
    problems = []
    denied = any(k in ans for k in DENIES)

    # 1. ข้อที่ต้องปฏิเสธ แต่กลับตอบไปเลย
    if no in MUST_DENY and not denied:
        problems.append(f"ต้องบอกว่าไม่มีข้อมูล ({MUST_DENY[no]}) แต่ตอบไปเลย")

    # 2. เอ่ยสินค้านอกรายการในลักษณะ "ให้ข้อมูล/แนะนำ" แม้จะปฏิเสธไปแล้วก็ตาม
    m = _DESCRIBES.search(ans)
    if m:
        problems.append(f"ให้ข้อมูลสินค้านอกรายการ: '{m.group(0)[:38]}...'")

    # 2. ราคาที่อ้างต้องมีจริงในฐานข้อมูล
    for p in cited_numbers(ans, r"ราคา\s*([\d,]{5,})\s*บาท"):
        if p not in REAL_PRICES:
            problems.append(f"อ้างราคา {p:,} ซึ่งไม่มีในฐานข้อมูล")
            break

    # 3. คะแนนเบนช์มาร์กที่อ้างต้องมีจริง
    for s in cited_numbers(ans, r"คะแนน\s*CPU[^\d]{0,12}([\d,]{4,})"):
        if s not in REAL_CPU:
            problems.append(f"อ้างคะแนน CPU {s:,} ซึ่งไม่มีในฐานข้อมูล")
            break
    for s in cited_numbers(ans, r"คะแนน\s*GPU[^\d]{0,12}([\d,]{4,})"):
        if s not in REAL_GPU:
            problems.append(f"อ้างคะแนน GPU {s:,} ซึ่งไม่มีในฐานข้อมูล")
            break

    return ("ไม่ผ่าน" if problems else "ผ่าน"), "; ".join(problems)


def main():
    qs = json.load(open(QFILE, encoding="utf-8"))
    results = []
    print(f"รัน {len(qs)} คำถาม\n" + "=" * 78)

    for d in qs:
        no, q = d["no"], d["q"]
        s = requests.Session()
        if no in FOLLOW_UP:                      # ตั้งบริบทก่อน
            s.post(f"{URL}/chat", json={"message": FOLLOW_UP[no]}, timeout=200)

        t = time.time()
        try:
            r = s.post(f"{URL}/chat", json={"message": q}, timeout=200)
            ans = r.json().get("message") or r.json().get("error", "")
        except Exception as e:
            ans = f"(เรียกไม่สำเร็จ: {e})"
        dt = time.time() - t

        verdict, why = check(ans, q, no)
        results.append({**d, "answer": ans, "verdict": verdict, "why": why, "sec": round(dt)})
        mark = "OK  " if verdict == "ผ่าน" else "FAIL"
        print(f"[{mark}] {no:>2}. {q[:52]:<54} {dt:>3.0f}s")
        if why:
            print(f"         -> {why}")

    json.dump(results, open(OUT, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    n_pass = sum(1 for r in results if r["verdict"] == "ผ่าน")
    print("=" * 78)
    print(f"สรุป: ผ่าน {n_pass}/{len(results)}")
    print(f"ผลละเอียดอยู่ที่ {OUT}")


if __name__ == "__main__":
    main()
