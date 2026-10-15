"""ดู "ข้อมูลดิบ" ที่ระบบหามาได้ ก่อนถูกส่งไปให้ ThaiLLM แต่งเป็นคำตอบ

ใช้ยังไง (รันจากรากโปรเจค):
    python tests/debug_q.py "โน้ตบุ๊คเล่นเกม งบ 25000" --prompt
    python tests/debug_q.py "รุ่นไหนแบตอึดที่สุด"
    python tests/debug_q.py "สวัสดี" --prompt
    python tests/debug_q.py "RAM 8GB พอไหมสำหรับการใช้งานในปีนี้"
    python tests/debug_q.py "MSI VenturePro 15 AI A1UDXG-029TH ใช้ CPU อะไร"
    python tests/debug_q.py "รุ่นที่ราคาอยู่ระหว่าง 30,000–40,000 บาท"
    python tests/debug_q.py "โน้ตบุ๊คงบ 10000"
    python tests/debug_q.py "โน้ตบุ๊คเล่นเกม งบ 10000"
    python tests/debug_q.py "รุ่นไหนถูกที่สุดในร้าน" --prompt    # ดู prompt เต็มด้วย

ไม่เรียก LLM เลย จึงตอบทันทีและไม่เปลืองโควตา
ต้องเปิด ENABLE_DEBUG=true (ตั้งไว้ใน docker-compose.yml แล้ว)
"""
import sys

import requests

# กัน UnicodeEncodeError บน Windows (console เป็น cp874) จะได้ไม่ต้องตั้ง PYTHONIOENCODING เอง
sys.stdout.reconfigure(encoding="utf-8")

URL = "http://localhost:5000/debug"
LINE = "=" * 78


def main():
    args = [a for a in sys.argv[1:] if a != "--prompt"]
    show_prompt = "--prompt" in sys.argv

    if not args:
        print(__doc__)
        sys.exit(1)

    question = " ".join(args)

    try:
        r = requests.post(URL, json={"message": question}, timeout=120)
    except requests.ConnectionError:
        print("ต่อ http://localhost:5000 ไม่ได้ — แอปรันอยู่หรือเปล่า?")
        sys.exit(1)

    if r.status_code != 200:
        print(f"[{r.status_code}] {r.json().get('error', r.text[:200])}")
        sys.exit(1)

    d = r.json()

    print(LINE)
    print(f"  คำถาม: {question}")
    print(LINE)

    # ── 1. ระบบตีความคำถามได้ว่าอะไร ──
    print("\n[1] สิ่งที่ระบบตีความได้ (ก่อนค้นหา)\n")
    interp = d["สิ่งที่ระบบตีความได้"]
    for k, v in interp.items():
        mark = "" if v in (None, False) else "  <--"
        print(f"    {k:<22} = {v}{mark}")

    # ── 2. เอกสารที่ดึงมาเป็น context ──
    docs = d["เอกสารที่ดึงมา"]
    print(f"\n[2] เอกสารที่ดึงมาเป็น context — รวม {docs['รวม']} ชิ้น")
    print(f"    (สเปก {len(docs['สเปก'])} / รีวิว {len(docs['รีวิว'])} / ภาพรวมร้าน {len(docs['ภาพรวมร้าน'])})\n")

    for i, s in enumerate(docs["สเปก"], 1):
        name = s.split(" ราคา ")[0]
        price = s.split(" ราคา ")[1].split(" บาท")[0] if " ราคา " in s else "?"
        print(f"    สเปก {i}. {name}  ({price} บาท)")

    for i, s in enumerate(docs["รีวิว"], 1):
        head = s.replace("รีวิวจากผู้ใช้รุ่น ", "")[:70]
        print(f"    รีวิว {i}. {head}...")

    for s in docs["ภาพรวมร้าน"]:
        print(f"    ภาพรวม. {s[:70]}...")

    # ── 3. ตารางส่วนต่างงบที่คำนวณด้วยโค้ด ──
    table = d.get("ตารางส่วนต่างงบ")
    print("\n[3] ตารางส่วนต่างงบ (คำนวณด้วยโค้ด ไม่ให้ LLM คิดเลขเอง)\n")
    if table:
        for line in table.strip().splitlines():
            print(f"    {line}")
    else:
        print("    (ลูกค้าไม่ได้บอกงบ จึงไม่มีตาราง)")

    # ── 4. prompt เต็ม ──
    prompt = d["prompt_ที่จะส่งให้_LLM"]
    print(f"\n[4] prompt ที่จะส่งให้ LLM — ยาว {len(prompt):,} ตัวอักษร")
    if show_prompt:
        print()
        print(prompt)
    else:
        tail = prompt[prompt.rfind("คำถามลูกค้า:"):]
        print("    (ใส่ --prompt เพื่อดูทั้งหมด) ท้าย prompt ที่โมเดลให้น้ำหนักมากที่สุด:\n")
        for line in tail.splitlines():
            print(f"    {line}")

    print(f"\n{LINE}")


if __name__ == "__main__":
    main()
