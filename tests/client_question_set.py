"""ชุดคำถามทดสอบของลูกค้า (laptop-questions.pdf, 9 หมวด 58 ข้อ) + เฉลยที่คำนวณจาก laptops.json

รัน:  PYTHONIOENCODING=utf-8 python tests/client_question_set.py http://localhost:5000 [out.json]
แต่ละข้อถามในแชทใหม่ ตรวจกับเฉลยอัตโนมัติ แล้วบันทึกคำตอบเต็มไว้ทำหน้า HTML
ข้อที่ตรวจด้วยโค้ดได้ไม่ครบ (คำแนะนำตามการใช้งาน) ติดป้าย "ตรวจด้วยคน" ให้คนอ่านคำตอบประกอบ
"""
import json
import os
import re
import sys
import time
from pathlib import Path

import requests

sys.stdout.reconfigure(encoding="utf-8")
BASE = Path(__file__).resolve().parent.parent
L = json.load(open(os.getenv("LAPTOPS_PATH", BASE / "laptops.json"), encoding="utf-8"))
URL = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:5000"
OUT = sys.argv[2] if len(sys.argv) > 2 else None
GAP = float(os.getenv("QS_GAP", "3"))


# ───────── ข้อมูลช่วยคำนวณ ─────────
def name(x): return x["ชื่อสินค้า"]
def price(x): return x["ราคา"]
def cpu(x): return x["spec"]["CPU"]
def gpu(x): return x["spec"]["GPU"]
def ram(x): return int(re.search(r"(\d+)\s*GB", x["spec"]["RAM"]).group(1))
def disp(x): return x["spec"]["DISPLAY"]
def size(x): return float(re.match(r"([\d.]+)", disp(x)).group(1))
def res(x):
    m = re.search(r"(\d{3,4})x(\d{3,4})", disp(x))
    return (int(m.group(1)), int(m.group(2))) if m else (0, 0)
def cs(x): return x["benchmark"]["CPU Score"]
def gs(x): return x["benchmark"]["GPU Score"]
def reviews(x): return " / ".join(x.get("reviews", []))
def brand(x): return name(x).split()[0]
def igpu(x): return not re.search(r"RTX|GTX|\bRX\s*\d", gpu(x))
def baht(n): return f"{n:,}"
def by(pat): return [x for x in L if re.search(pat, name(x))]
def one(pat): return by(pat)[0]
def rv(pat): return [x for x in L if re.search(pat, reviews(x), re.I)]
def names(xs): return [name(x) for x in xs]


# ───────── คำถาม + เฉลย ─────────
# check: all = ต้องเอ่ยครบทุกรุ่น / any = อย่างน้อย 1 รุ่น / only = รุ่นที่เอ่ยต้องอยู่ในชุดนี้ / nums = ตัวเลขที่ต้องมี
# nodata = ต้องบอกว่าไม่มีข้อมูล / manual = ตรวจด้วยคนประกอบ
Q = []
def q(cat, text, key, **check):
    Q.append({"id": len(Q) + 1, "cat": cat, "q": text, "key": key, "check": check})

C1, C2, C3, C4, C5, C6, C7, C8, C9 = ("1. ค้นหาข้อมูลรุ่นเดียว", "2. กรองตามงบประมาณ", "3. กรองตามสเปก",
                                      "4. Benchmark และการจัดอันดับ", "5. เปรียบเทียบ", "6. คำถามจากรีวิว",
                                      "7. แนะนำตามการใช้งาน", "8. สรุปภาพรวม", "9. คำถามนอกขอบเขตข้อมูล")

x = one(r"Legion Pro 5 16IAX10H")
q(C1, "Lenovo Legion Pro 5 16IAX10H ราคาเท่าไหร่ ใช้ CPU อะไร",
  f"{name(x)} ราคา {baht(price(x))} บาท ใช้ {cpu(x)}", nums=[price(x)], words=["275HX"])
x = one(r"OMEN 16-ap0195AX")
q(C1, "HP OMEN 16-ap0195AX มี RAM กี่ GB", f"{x['spec']['RAM']}", words=["24"])
x = one(r"ZenBook 14 OLED")
q(C1, "Asus ZenBook 14 OLED จอความละเอียดเท่าไหร่", f"{name(x)}: {disp(x)}", words=["2880"])
x = one(r"Stealth A16")
q(C1, "MSI Stealth A16 AI+ ได้คะแนน GPU benchmark เท่าไหร่", f"{name(x)}: GPU Score {baht(gs(x))}", nums=[gs(x)])
x = one(r"ANV15-52-73BK")
q(C1, "Acer Nitro V 15 ANV15-52-73BK ใช้ RAM แบบไหน", f"{x['spec']['RAM']} (เป็น DDR4)", words=["DDR4"])

lo, hi = min(L, key=price), max(L, key=price)
q(C2, "โน้ตบุ๊กที่ถูกที่สุดคือรุ่นไหน แพงที่สุดคือรุ่นไหน",
  f"ถูกสุด {name(lo)} {baht(price(lo))} บาท / แพงสุด {name(hi)} {baht(price(hi))} บาท", all=[name(lo), name(hi)])
s = sorted([x for x in L if price(x) <= 20000], key=price)
q(C2, "งบไม่เกิน 20,000 บาท มีรุ่นไหนบ้าง", f"{len(s)} รุ่น: " + ", ".join(f"{name(x)} ({baht(price(x))})" for x in s),
  all=names(s), only=names(s))
s = [x for x in L if 30000 <= price(x) <= 40000]
top = max(gs(x) for x in s)
t = sorted([x for x in s if gs(x) == top], key=price)
q(C2, "งบ 30,000–40,000 บาท รุ่นไหนการ์ดจอแรงที่สุด",
  f"GPU Score สูงสุด {baht(top)} มี {len(t)} รุ่นเท่ากัน: " + ", ".join(f"{name(x)} ({baht(price(x))})" for x in t),
  any=names(t), nums=[top])
s = [x for x in L if price(x) == 47990]
q(C2, "มีโน้ตบุ๊กที่ราคา 47,990 บาทกี่รุ่น อะไรบ้าง", f"{len(s)} รุ่น: " + ", ".join(names(s)), all=names(s), only=names(s))
s = sorted([x for x in L if price(x) <= 30000 and gs(x) >= 8500 and "RTX" in gpu(x)], key=price)
q(C2, "งบไม่เกิน 30,000 อยากได้ RTX 4050 ขึ้นไป มีอะไรบ้าง",
  f"การ์ดจอ RTX ที่ GPU Score ≥ 8,500 (ระดับ RTX 4050) และราคาไม่เกิน 30,000 มี {len(s)} รุ่น: "
  + ", ".join(f"{name(x)} ({gpu(x).replace('NVIDIA GeForce ', '')}, {baht(price(x))})" for x in s), all=names(s), only=names(s))

for label, pat, fn in (("RTX 5090", None, lambda x: "5090" in gpu(x)), ("RAM 64GB", None, lambda x: ram(x) == 64),
                       ("RAM แค่ 8GB", None, lambda x: ram(x) == 8), ("จอ 18 นิ้ว", None, lambda x: int(size(x)) == 18)):
    s = sorted([x for x in L if fn(x)], key=price)
    text = {"RTX 5090": "รุ่นไหนใช้ RTX 5090 บ้าง", "RAM 64GB": "รุ่นไหนมี RAM 64GB",
            "RAM แค่ 8GB": "รุ่นไหนยังมี RAM แค่ 8GB", "จอ 18 นิ้ว": "จอ 18 นิ้วมีรุ่นไหนบ้าง"}[label]
    q(C3, text, f"{len(s)} รุ่น: " + ", ".join(names(s)), all=names(s), only=names(s))
s = sorted([x for x in L if igpu(x)], key=price)
q(C3, "รุ่นไหนไม่มีการ์ดจอแยก (ใช้ iGPU)", f"{len(s)} รุ่น: " + ", ".join(names(s)), all=names(s), only=names(s))
g5, g3 = [x for x in L if "Gen 5" in x["spec"]["STORAGE"]], [x for x in L if "Gen 3" in x["spec"]["STORAGE"]]
q(C3, "รุ่นไหนใช้ SSD Gen 5 หรือ Gen 3",
  f"Gen 5 มี {len(g5)} รุ่น: {', '.join(names(g5)) or '-'} / Gen 3 มี {len(g3)} รุ่น: {', '.join(names(g3)) or '-'}",
  all=names(g5 + g3), only=names(g5 + g3))
s = [x for x in L if "Single Language" in x["spec"]["OS"]]
q(C3, "รุ่นไหนใช้ Windows 11 Home Single Language", f"{len(s)} รุ่น: " + ", ".join(names(s)), all=names(s), only=names(s))
s = [x for x in L if re.search(r"AMD|Ryzen", cpu(x)) and "5060" in gpu(x)]
q(C3, "อยากได้ AMD CPU + RTX 5060 มีรุ่นไหนบ้าง", f"{len(s)} รุ่น: " + ", ".join(f"{name(x)} ({cpu(x)})" for x in s),
  all=names(s), only=names(s))
s = [x for x in L if res(x)[0] * res(x)[1] > 2560 * 1600]
q(C3, "รุ่นไหนจอความละเอียดสูงกว่า 2560x1600", f"{len(s)} รุ่น: " + ", ".join(f"{name(x)} ({res(x)[0]}x{res(x)[1]})" for x in s),
  all=names(s), only=names(s))

s = sorted(L, key=lambda x: (-cs(x), price(x)))[:5]
q(C4, "CPU Score สูงที่สุด 5 อันดับแรก", " / ".join(f"{i}. {name(x)} ({baht(cs(x))})" for i, x in enumerate(s, 1)),
  all=names(s[:2]), nums=[cs(s[0]), cs(s[1]), cs(s[2])])   # อันดับ 3-5 คะแนนเท่ากัน ไม่บังคับว่าต้องเป็นรุ่นไหน
top = max(gs(x) for x in L)
t = [x for x in L if gs(x) == top]
q(C4, "GPU Score สูงที่สุดคือรุ่นไหน มีเสมอกันไหม", f"GPU Score {baht(top)} เสมอกัน {len(t)} รุ่น: " + ", ".join(names(t)),
  all=names(t), nums=[top])
a, b = one(r"Legion 9i"), one(r"Raider 16 Max")
q(C4, "Legion 9i กับ MSI Raider 16 ต่างกันตรงไหน ทำไมตัวถูกกว่าได้ CPU Score สูงกว่า",
  f"{name(b)} {baht(price(b))} บาท ถูกกว่า {baht(price(a) - price(b))} บาท แต่ CPU Score {baht(cs(b))} สูงกว่า "
  f"{name(a)} ({baht(cs(a))}) เพราะใช้ {cpu(b)} ใหม่กว่า {cpu(a)} / Legion 9i ได้จอ 18 นิ้ว 3840x2400 แรม 64GB SSD Gen 5 "
  f"ส่วน Raider 16 จอ 16 นิ้ว แรม 32GB / GPU RTX 5090 เท่ากัน", all=[name(a), name(b)], nums=[cs(a), cs(b)])
best = max(L, key=lambda x: gs(x) / price(x))
q(C4, "รุ่นไหนคุ้มที่สุดเมื่อคิด GPU Score ต่อราคา 1,000 บาท",
  f"{name(best)}: GPU {baht(gs(best))} / ราคา {baht(price(best))} = {gs(best) / price(best) * 1000:.1f} คะแนนต่อ 1,000 บาท",
  all=[name(best)])
s = sorted([x for x in L if cs(x) >= 25000 and gs(x) < 6000], key=lambda x: -cs(x))
q(C4, "รุ่นที่ CPU แรงแต่ GPU อ่อน (เหมาะทำงานมากกว่าเล่นเกม) มีอะไรบ้าง",
  f"เกณฑ์ CPU ≥ 25,000 และ GPU < 6,000 มี {len(s)} รุ่น: " + ", ".join(f"{name(x)} (CPU {baht(cs(x))} / GPU {baht(gs(x))})" for x in s),
  any=names(s), only=names(s), manual=True)
s = [x for x in L if "5060" in gpu(x)]
top = max(s, key=cs)
q(C4, "รุ่นที่ใช้ RTX 5060 เหมือนกัน รุ่นไหน CPU แรงสุด",
  f"{name(top)} CPU Score {baht(cs(top))} ({cpu(top)}) จาก RTX 5060 ทั้งหมด {len(s)} รุ่น", all=[name(top)], nums=[cs(top)])

leg = [x for x in by(r"Legion 5 ") if abs(price(x) - 48000) <= 2000]
tuf = [x for x in by(r"TUF Gaming F16") if abs(price(x) - 48000) <= 2000]
q(C5, "Lenovo Legion 5 กับ Asus TUF F16 ในงบประมาณ 48,000 ต่างกันยังไง",
  "Legion 5 ราคาใกล้ 48,000: " + ", ".join(f"{name(x)} ({baht(price(x))}, {cpu(x)}, จอ {disp(x)})" for x in leg)
  + " / TUF F16: " + ", ".join(f"{name(x)} ({baht(price(x))}, {cpu(x)}, จอ {disp(x)})" for x in tuf),
  any=names(leg), any2=names(tuf), manual=True)
s = by(r"OMEN 16")
q(C5, "HP OMEN 16 ทั้ง 4 รุ่นต่างกันตรงไหน", " / ".join(f"{name(x)} {baht(price(x))} {cpu(x)} {gpu(x).replace('NVIDIA GeForce ', '')} "
  f"{x['spec']['RAM'].split(' DDR')[0]} {res(x)[0]}x{res(x)[1]}" for x in s), all=names(s))
a, b = one(r"ANV15-52-501P"), one(r"ANV15-52-73BK")
q(C5, "Acer Nitro V 15 ANV15-52-501P กับ ANV15-52-73BK คุ้มเพิ่มเงิน 2,000 ไหม",
  f"ต่างกัน {baht(price(b) - price(a))} บาท ได้ CPU {cpu(b)} (CPU {baht(cs(b))}) แทน {cpu(a)} ({baht(cs(a))}) ส่วนอื่นเหมือนกัน "
  f"(RTX 5050, แรม 16GB DDR4, จอ 15.6 Full HD)", all=[name(a), name(b)], nums=[2000])
s = by(r"Vivobook S16")
q(C5, "Vivobook S16 สองรุ่นต่างกันแค่อะไร", " / ".join(f"{name(x)} {baht(price(x))} บาท SSD {x['spec']['STORAGE'].split(' SSD')[0]}" for x in s)
  + " (ต่างแค่ความจุ SSD)", all=names(s), words=["1TB", "512"])
a5, a5t = [x for x in L if re.search(r"5070(?! Ti)", gpu(x))], [x for x in L if "5070 Ti" in gpu(x)]
q(C5, "RTX 5070 กับ 5070 Ti คะแนน GPU ต่างกันเท่าไหร่ ราคาต่างกันเท่าไหร่",
  f"RTX 5070 = {baht(gs(a5[0]))} / 5070 Ti = {baht(gs(a5t[0]))} ต่างกัน {baht(gs(a5t[0]) - gs(a5[0]))} คะแนน / "
  f"ราคา RTX 5070 {baht(min(map(price, a5)))}–{baht(max(map(price, a5)))} บาท, 5070 Ti {baht(min(map(price, a5t)))}–{baht(max(map(price, a5t)))} บาท "
  f"(รุ่นถูกสุดต่างกัน {baht(min(map(price, a5t)) - min(map(price, a5)))} บาท)", nums=[gs(a5[0]), gs(a5t[0])], manual=True)

for text, pat, extra in (
        ("รุ่นไหนรีวิวบอกว่าพัดลมดัง", r"พัดลม[^/]{0,15}ดัง|เสียงพัดลม", {}),
        ("รุ่นไหนมีปัญหาจอกระตุกตอนชาร์จ", r"กระตุก[^/]{0,30}ชาร์จ|ชาร์จ[^/]{0,30}กระตุก", {}),
        ("รุ่นไหนรีวิวเตือนเรื่อง OLED burn-in", r"burn.?in", {}),
        ("รุ่นไหนรีวิวบอกว่าแบตอึด 8 ชั่วโมงขึ้นไป", r"แบต[^/]{0,25}(?:[89]|1[0-9])(?:\s*[-–]\s*\d+)?\s*(?:ชม|ชั่วโมง)", {}),
        ("รุ่นไหนรีวิวบอกว่าหนัก พกพาไม่สะดวก", r"หนัก[^/]{0,20}(?:พก|ไม่สะดวก)|พกพา(?:ลำบาก|ยาก|ไม่สะดวก)", {}),
        ("รุ่นไหนรองรับชาร์จผ่าน USB-C", r"USB.?C[^/]{0,25}(?:ชาร์จ|charg|PD|Power Delivery)|ชาร์จ[^/]{0,25}USB.?C", {}),
        ("รุ่นไหนลำโพงดี เช่น B&O, HyperX, Dolby Atmos, Dynaudio", r"B&O|Bang|Dolby|Dynaudio|HyperX[^/]{0,10}(?:ลำโพง|audio)|ลำโพง[^/]{0,10}HyperX", {}),
        ("รุ่นไหนผ่านมาตรฐาน MIL-STD 810H", r"MIL.?STD", {}),
        ("รุ่นไหนมี Number pad", r"number\s*pad|numpad|numeric\s*keypad|แป้นตัวเลข", {})):
    s = rv(pat)
    q(C6, text, f"จากรีวิว {len(s)} รุ่น: " + ", ".join(names(s)), any=names(s), only=names(s), manual=True)
# แบรนด์ที่ถูกบ่นเรื่องบริการหลังการขาย
cnt = {}
for x in rv(r"หลังการขาย|ศูนย์บริการ|เคลม|service"):
    cnt[brand(x)] = cnt.get(brand(x), 0) + 1
topb = max(cnt, key=cnt.get) if cnt else "-"
Q.insert(next(i for i, z in enumerate(Q) if z["cat"] == C6) + 1, {
    "id": 0, "cat": C6, "q": "แบรนด์ไหนถูกบ่นเรื่องบริการหลังการขายบ่อยที่สุด",
    "key": "นับรุ่นที่รีวิวพูดถึงบริการหลังการขาย/เคลม แยกยี่ห้อ: " + ", ".join(f"{b} {n} รุ่น" for b, n in sorted(cnt.items(), key=lambda kv: -kv[1]))
           + f" → มากที่สุดคือ {topb}", "check": {"words": [topb], "manual": True}})

s = sorted([x for x in L if price(x) <= 35000 and not igpu(x)], key=lambda x: -(cs(x) + gs(x)))
q(C7, "นักศึกษาวิศวะ ใช้ SolidWorks/AutoCAD งบ 35,000 ควรซื้อรุ่นไหน",
  "ควรเป็นรุ่นที่ราคาไม่เกิน 35,000 และมีการ์ดจอแยก เช่น " + ", ".join(f"{name(x)} ({baht(price(x))})" for x in s[:5]),
  only=[name(x) for x in L if price(x) <= 35000], manual=True)
s = sorted([x for x in L if price(x) <= 15000], key=price)
q(C7, "ซื้อให้ลูกเรียนมัธยม ไม่เล่นเกม งบ 15,000", f"รุ่นในงบมี {len(s)} รุ่น: " + ", ".join(f"{name(x)} ({baht(price(x))})" for x in s),
  any=names(s), only=names(s), manual=True)
s = sorted([x for x in L if price(x) <= 60000 and gs(x) >= 14500], key=lambda x: (-gs(x), -cs(x)))
q(C7, "ตัดต่อวิดีโอ Premiere + เล่นเกม AAA งบ 60,000",
  "ควรเป็นรุ่นไม่เกิน 60,000 ที่การ์ดจอระดับสูง (GPU ≥ 14,500) เช่น " + ", ".join(f"{name(x)} ({baht(price(x))}, GPU {baht(gs(x))})" for x in s[:5]),
  only=[name(x) for x in L if price(x) <= 60000], manual=True)
q(C7, "ต้องการเครื่องเบา แบตอึด พกไปร้านกาแฟ", "ควรเป็นรุ่นบางเบาไม่มีการ์ดจอแยก ที่รีวิวพูดถึงน้ำหนักเบาหรือแบตอึด "
  "(ไม่มีข้อมูลน้ำหนักและแบตในสเปก ต้องอ้างจากรีวิว)", manual=True)
q(C7, "อยากเล่นเกมแต่ไม่อยากให้เครื่องดูเป็นเกมมิ่ง", "ควรเป็นรุ่นที่มีการ์ดจอแยกและรีวิวบอกว่าดีไซน์เรียบ ไม่ดูเป็นเกมมิ่ง: "
  + ", ".join(names(rv(r"ไม่ดูเป็นเกมมิ่ง|ไม่เหมือนเกมมิ่ง|ดูเป็นโน้ตบุ๊[กค]ทำงาน|ดีไซน์เรียบ|เรียบหรู")[:6])), manual=True)
s = rv(r"พัดลม[^/]{0,15}ดัง|เสียงพัดลม|coil whine")
q(C7, "ทำงานในออฟฟิศที่ต้องเงียบ ควรเลี่ยงรุ่นไหน", f"รุ่นที่รีวิวบอกว่าพัดลมดังหรือมีเสียง {len(s)} รุ่น เช่น " + ", ".join(names(s)[:8]),
  any=names(s), manual=True)
s = rv(r"Copilot\+?|NPU")
q(C7, "อยากได้ Copilot+ PC รัน AI ได้ มีรุ่นไหนบ้าง", f"รีวิวที่พูดถึง Copilot+ หรือ NPU {len(s)} รุ่น: " + ", ".join(names(s)),
  any=names(s), manual=True)
q(C7, "desktop replacement ที่คุ้มที่สุด", "ไม่มีเฉลยตายตัว ควรเป็นรุ่นจอใหญ่ 16-18 นิ้ว สเปกแรง พร้อมเหตุผลด้านราคาต่อสเปก", manual=True)

cnt = {}
for x in L:
    cnt[brand(x)] = cnt.get(brand(x), 0) + 1
q(C8, "แต่ละแบรนด์มีกี่รุ่นในรายการ", ", ".join(f"{b} {n}" for b, n in sorted(cnt.items(), key=lambda kv: -kv[1])),
  words=[f"{n}" for n in sorted(set(cnt.values()))])
s = [price(x) for x in L if "5060" in gpu(x)]
avg = sum(s) / len(s)
q(C8, "ราคาเฉลี่ยของโน้ตบุ๊กที่ใช้ RTX 5060", f"{len(s)} รุ่น ราคาเฉลี่ย {avg:,.0f} บาท", nums=[round(avg)], approx=True)
ib = {}
for x in L:
    if igpu(x):
        ib[brand(x)] = ib.get(brand(x), 0) + 1
topb = max(ib, key=ib.get)
q(C8, "แบรนด์ไหนมีรุ่นที่ไม่มีการ์ดจอแยกมากที่สุด", ", ".join(f"{b} {n} รุ่น" for b, n in sorted(ib.items(), key=lambda kv: -kv[1]))
  + f" → {topb}", words=[topb])
s = by(r"LOQ")
q(C8, "ช่วงราคาของ Lenovo LOQ ทั้งหมดเท่าไหร่", f"{len(s)} รุ่น ราคา {baht(min(map(price, s)))}–{baht(max(map(price, s)))} บาท",
  nums=[min(map(price, s)), max(map(price, s))])
gc = {}
for x in L:
    g = gpu(x).replace("NVIDIA GeForce ", "")
    gc[g] = gc.get(g, 0) + 1
topg = max(gc, key=gc.get)
q(C8, "การ์ดจอรุ่นไหนพบบ่อยที่สุดในรายการ", ", ".join(f"{g} {n}" for g, n in sorted(gc.items(), key=lambda kv: -kv[1])[:5])
  + f" → {topg}", words=[topg.split()[-1]])

q(C9, "รุ่นไหนมีจอ 240Hz", "ไม่มีข้อมูล สเปกไม่มีค่า refresh rate", nodata=True)
q(C9, "Legion 9i หนักกี่กิโล", "ไม่มีข้อมูลน้ำหนักในสเปก", nodata=True)
q(C9, "มี MacBook ในรายการไหม", "ไม่มี MacBook ในรายการ", nodata=True)
q(C9, "ประกันกี่ปี", "ไม่มีข้อมูลประกันในสเปก (มีแค่รีวิวของ Acer Nitro V 15 ANV15-52-73BK ที่พูดถึง 3 ปี Onsite)", nodata=True)
q(C9, "ร้านไหนขายถูกสุด ตอนนี้ลดราคาไหม", "ไม่มีข้อมูลร้านค้าหรือโปรโมชั่น", nodata=True)

for i, z in enumerate(Q, 1):
    z["id"] = i


# ───────── ตรวจ ─────────
ALL_NAMES = names(L)
_NODATA = re.compile(r"ไม่มีข้อมูล|ไม่ครอบคลุม|ไม่ได้ระบุ|ไม่มีระบุ|ไม่มีในข้อมูล|ไม่มีรายละเอียด|ไม่ทราบ|ไม่มี.{0,20}(?:ในรายการ|ในร้าน|ข้อมูล)|ตอบไม่ได้")


def mentioned(ans):
    return [n for n in ALL_NAMES if n in ans]


def grade(z, ans):
    c, notes, ok = z["check"], [], True
    nums = [int(n.replace(",", "")) for n in re.findall(r"\d[\d,]*\d|\d", ans) if n.replace(",", "").isdigit()]
    if c.get("all"):
        miss = [n for n in c["all"] if n not in ans]
        if miss:
            ok = False
            notes.append(f"ขาด {len(miss)} จาก {len(c['all'])} รุ่น")
    for key in ("any", "any2"):
        if c.get(key) and not any(n in ans for n in c[key]):
            ok = False
            notes.append("ไม่ได้เอ่ยรุ่นที่ถูกต้องเลย")
    if c.get("only"):
        extra = [n for n in mentioned(ans) if n not in c["only"]]
        if extra:
            ok = False
            notes.append("มีรุ่นที่ไม่ตรงปน: " + ", ".join(extra[:4]))
    for n in c.get("nums", []):
        if c.get("approx"):
            if not any(abs(v - n) <= max(500, n * 0.01) for v in nums):
                ok = False
                notes.append(f"ไม่มีตัวเลขใกล้ {n:,}")
        elif n not in nums:
            ok = False
            notes.append(f"ไม่มีตัวเลข {n:,}")
    for w in c.get("words", []):
        if w.lower() not in ans.lower():
            ok = False
            notes.append(f"ไม่มีคำว่า {w}")
    if c.get("nodata") and not _NODATA.search(ans):
        ok = False
        notes.append("ไม่ได้บอกว่าไม่มีข้อมูล")
    return ok, notes


def ask(text):
    s = requests.Session()
    t0 = time.time()
    try:
        r = s.post(f"{URL}/chat", json={"message": text}, timeout=300)
        d = r.json() if r.headers.get("content-type", "").startswith("application/json") else {}
        return (d.get("message") or ""), r.status_code, (d.get("error") or ""), time.time() - t0
    except requests.RequestException as e:
        return "", 0, str(e), time.time() - t0


if __name__ == "__main__":
    if "--keys" in sys.argv:
        for z in Q:
            print(f"{z['id']:>2}. [{z['cat']}] {z['q']}\n    เฉลย: {z['key'][:300]}")
        sys.exit(0)
    only = os.getenv("QS_ONLY")
    results = []
    for z in Q:
        if only and str(z["id"]) not in only.split(","):
            continue
        ans, status, err, secs = ask(z["q"])
        if status != 200 or not ans:      # สะดุดชั่วคราว (502 / timeout) -> ถามซ้ำ 1 ครั้ง
            time.sleep(5)
            ans, status, err, secs = ask(z["q"])
        ok, notes = grade(z, ans) if ans else (False, [f"ไม่ได้คำตอบ: HTTP {status} {err}"])
        auto = "ผ่าน" if ok else "ไม่ผ่าน"
        print(f"{'✅' if ok else '❌'} {z['id']:>2}. {z['q']}  ({secs:.1f} วิ){'  [ตรวจด้วยคน]' if z['check'].get('manual') else ''}")
        if notes:
            print("      " + " · ".join(notes))
        results.append({**z, "answer": ans, "auto_ok": ok, "notes": notes, "seconds": round(secs, 1)})
        time.sleep(GAP)
    passed = sum(r["auto_ok"] for r in results)
    print(f"\nผ่านตามเกณฑ์อัตโนมัติ {passed} / {len(results)}")
    if OUT:
        Path(OUT).write_text(json.dumps({"url": URL, "when": time.strftime("%Y-%m-%d %H:%M"), "results": results},
                                        ensure_ascii=False, indent=1), encoding="utf-8")
