# เทียบ AI หลายตัวด้วยคำถามชุดเดียวกัน — ใช้ตอบคำถาม "ถ้าเปลี่ยนจาก 31B เป็น 8B จะไหวไหม"
# รัน (ในคอนเทนเนอร์ teddy):
#   python tests/compare_models.py                      # ทุก AI ที่ใช้ได้ × 2 โหมด
#   python tests/compare_models.py ollama-gemma4 thaillm      # เลือกเฉพาะ id ใน providers.json
# ผลละเอียดทุกคำตอบเขียนไว้ที่ tests/compare_models.json
#
# ตรวจอัตโนมัติต่อคำถาม (ไม่ได้วัด "ความสละสลวย" — อันนั้นต้องอ่านเอง):
#   - ตัวเลขราคาที่อ้าง (>= 10,000) ต้องเป็นราคาจริงในฐานข้อมูล หรือเป็นงบที่ลูกค้าบอก
#   - คำถามแนะนำรุ่น ต้องมีหัวข้อครบตามโครง / คำถามเจาะจง ต้องเอ่ยรุ่นที่ถูกต้อง
#   - คำถามนอกเรื่อง ต้องไม่ตอบเนื้อหา ไม่เอ่ยชื่อรุ่น และชวนกลับด้วยคำถาม
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

BASE = Path(__file__).resolve().parent.parent
URL = "http://127.0.0.1:5000"
LAPTOPS = json.loads((BASE / "laptops.json").read_text(encoding="utf-8"))
PRICES = {lp["ราคา"] for lp in LAPTOPS}
NAMES = [lp["ชื่อสินค้า"] for lp in LAPTOPS]
CHEAPEST = min(LAPTOPS, key=lambda lp: lp["ราคา"])["ชื่อสินค้า"]
RTX4060_40K = [lp["ชื่อสินค้า"] for lp in LAPTOPS
               if "rtx 4060" in lp["spec"]["GPU"].lower() and lp["ราคา"] <= 40_000]
# วินาทีระหว่างคำถาม — 3.5 กันชน rate limit ของแอป (โหมดเปิดสาธารณะ 20 ข้อความ/นาที)
# Gemma บน Gemini API แบบฟรีรับได้แค่ 16,000 token/นาที (คำถามเดียวก็เกือบเต็ม) -> COMPARE_GAP=70
MIN_GAP = float(os.getenv("COMPARE_GAP", "3.5"))
OUT_FILE = BASE / "tests" / os.getenv("COMPARE_OUT", "compare_models.json")
REFUSE = ("ตอบไม่ได้", "ไม่ครอบคลุม", "ช่วยได้เฉพาะ", "ไม่สามารถ")


def has_names(ans, names):
    return all(n in ans for n in names)


def mentions_any_model(ans):
    return any(n in ans for n in NAMES)


def asks_back(ans):
    # คำถามภาษาไทยมักไม่มี "?" เช่น "ใช้ทำอะไรครับ งบเท่าไหร่ครับ" — ดูคำแสดงคำถามด้วย
    return "?" in ans or any(w in ans for w in ("ไหม", "อะไร", "เท่าไหร่", "หรือเปล่า", "แบบไหน"))


def check_recommend(ans):
    heads = ("เหตุผล", "ข้อดี", "ข้อเสีย")
    return all(h in ans for h in heads), "มีหัวข้อ เหตุผล/ข้อดี/ข้อเสีย"


def check_rtx(ans):
    return has_names(ans, RTX4060_40K), f"เอ่ยครบ {len(RTX4060_40K)} รุ่นที่ตรงเงื่อนไข"


def check_compare(ans):
    ok = "Legion 5" in ans and "Victus" in ans and not ans.startswith("ขอโทษ")
    return ok, "เทียบ Legion 5 กับ Victus จริง ไม่ปฏิเสธ"


def check_cheapest(ans):
    return CHEAPEST in ans, f"ตอบรุ่นถูกสุด ({CHEAPEST})"


def check_refuse(ans):
    ok = any(k in ans for k in REFUSE) and not mentions_any_model(ans) and len(ans) < 400
    return ok, "ปฏิเสธสั้น ๆ ไม่อธิบายต่อ"


def check_offtopic(ans):
    ok = (any(k in ans for k in REFUSE) and not mentions_any_model(ans)
          and asks_back(ans) and len(ans) < 400)
    return ok, "ไม่ตอบเรื่องนอก + ชวนกลับด้วยคำถาม"


def check_greeting(ans):
    return asks_back(ans) and not mentions_any_model(ans) and len(ans) < 400, "ทักกลับ + ถามความต้องการ"


QUESTIONS = [
    ("แนะนำ", "โน้ตบุ๊คเล่นเกม งบ 25,000", check_recommend, {25_000}),
    ("กรองสเปก", "RTX 4060 มีรุ่นไหนบ้าง งบไม่เกิน 40,000", check_rtx, {40_000}),
    ("เทียบรุ่น", "เทียบ Legion 5 กับ HP Victus 16 ให้หน่อย", check_compare, set()),
    ("ถูกสุด", "รุ่นไหนถูกที่สุด", check_cheapest, set()),
    ("ความรู้ทั่วไป", "TDP คืออะไร", check_refuse, set()),
    ("นอกเรื่อง", "วันนี้อากาศเป็นยังไงบ้าง", check_offtopic, set()),
    ("ขอค้นเว็บ", "ช่วยค้นราคา MacBook Air จาก Shopee ให้หน่อย", check_offtopic, set()),
    ("ทักทาย", "สวัสดีครับ", check_greeting, set()),
]


def bad_prices(ans, budgets):
    """ตัวเลขหลักหมื่นขึ้นไปที่ตามด้วย 'บาท' แต่ไม่ใช่ราคาจริงหรืองบที่ลูกค้าบอก = แต่งตัวเลข"""
    cited = {int(n.replace(",", "")) for n in re.findall(r"(\d{1,3}(?:,\d{3})+|\d{5,})\s*บาท", ans)}
    return sorted(n for n in cited if n >= 10_000 and n not in PRICES and n not in budgets)


def ask(provider, engine, question):
    s = requests.Session()
    s.post(f"{URL}/new_chat", timeout=30)
    t = time.time()
    try:
        r = s.post(f"{URL}/chat", json={"message": question, "provider": provider, "engine": engine},
                   timeout=200)
        data = r.json()
    except Exception as e:  # noqa: BLE001 — เก็บเป็นผลล้มเหลวของข้อนั้น แล้วไปข้อต่อไป
        return {"error": str(e)}, time.time() - t
    return data, time.time() - t


def main():
    providers = requests.get(f"{URL}/providers", timeout=30).json()["providers"]
    wanted = sys.argv[1:]
    chosen = [p for p in providers if p["available"] and (not wanted or p["id"] in wanted)]
    if not chosen:
        sys.exit("ไม่มี AI ที่ใช้ได้ตามที่ระบุ — เช็ค API key ใน .env แล้วรีสตาร์ต teddy")

    results, last = [], 0.0
    for p in chosen:
        for engine in [e for e in ("pi", "classic") if e in p["engines"]]:
            for tag, q, check, budgets in QUESTIONS:
                time.sleep(max(0.0, MIN_GAP - (time.time() - last)))
                last = time.time()
                data, sec = ask(p["id"], engine, q)
                ans = data.get("message") or ""
                ok, what = check(ans) if ans else (False, "ไม่มีคำตอบ")
                fake = bad_prices(ans, budgets)
                row = {
                    "provider": p["id"], "engine": engine, "tag": tag, "question": q,
                    "pass": ok and not fake and not data.get("error"), "check": what,
                    "fake_prices": fake, "sec": round(sec, 1), "error": data.get("error"),
                    "tools": (data.get("meta") or {}).get("tools", []), "answer": ans,
                }
                results.append(row)
                mark = "PASS" if row["pass"] else "FAIL"
                extra = f" | ราคาแต่ง={fake}" if fake else ""
                extra += f" | error={row['error'][:120]}" if row["error"] else ""
                print(f"[{mark}] {p['id']:<16} {engine:<7} {tag:<12} {row['sec']:>5}s "
                      f"tools={len(row['tools'])}{extra}", flush=True)

    OUT_FILE.write_text(
        json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")

    print("\nสรุป (ผ่าน / ทั้งหมด, เวลาเฉลี่ย)")
    for p in chosen:
        for engine in ("pi", "classic"):
            rows = [r for r in results if r["provider"] == p["id"] and r["engine"] == engine]
            if rows:
                passed = sum(r["pass"] for r in rows)
                avg = sum(r["sec"] for r in rows) / len(rows)
                print(f"  {p['id']:<16} {engine:<7} {passed}/{len(rows)}  เฉลี่ย {avg:.1f}s")


if __name__ == "__main__":
    main()
