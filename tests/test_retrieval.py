# ทดสอบชั้น retrieval ตรง ๆ (vector search + ตัวกรองงบ) ไม่ผ่าน LLM
# ต้องรันในที่ที่มี chromadb + chroma_db + โมเดล bge-m3 => รันใน container:
#   docker cp tests/test_retrieval.py ai_teddy-teddy-1:/app/test_retrieval.py
#   docker exec ai_teddy-teddy-1 python /app/test_retrieval.py
import json
import sys
from pathlib import Path

sys.path.insert(0, "/app")
from app import retrieve, get_collection  # noqa: E402

LAPTOPS = json.loads(Path("/app/laptops.json").read_text(encoding="utf-8"))
PRICES = {lp["ชื่อสินค้า"]: lp["ราคา"] for lp in LAPTOPS}
PASS, FAIL = [], []


def check(name, ok, detail):
    (PASS if ok else FAIL).append((name, detail))
    print(f"{'[PASS]' if ok else '[FAIL]'} {name}\n       {detail}")


def is_summary(doc):
    # ใช้ตัวเดียวกับแอป เพื่อไม่ให้เทสต์ล้าสมัยเมื่อข้อความในเอกสารสรุปเปลี่ยน
    from app import is_summary_doc
    return is_summary_doc(doc)


def doc_price(doc):
    """หาราคาของรุ่นที่เอกสารนี้พูดถึง โดยจับคู่ชื่อรุ่นในข้อความ
    ข้ามเอกสารภาพรวมร้าน เพราะมันเอ่ยชื่อรุ่นถูกสุด/แพงสุดไว้ด้วยโดยตั้งใจ"""
    if is_summary(doc):
        return None, None
    for name, price in PRICES.items():
        if name in doc:
            return name, price
    return None, None


print("=" * 78)
print("RETRIEVAL TESTS (ใน container — ไม่เรียก LLM)")
print("=" * 78)

col = get_collection()
check("collection มีเอกสารครบ", col.count() == 99 + 297 + 1,
      f"นับได้ {col.count()} เอกสาร (คาด {99+297+1} = สเปก 99 + รีวิว 297 + สรุป 1)")

# ── 1. ตัวกรองงบ: ทุกเอกสารที่คืนมาต้องราคา <= งบ ──
for budget in [15000, 20000, 25000, 30000, 50000]:
    docs, over = retrieve("โน้ตบุ๊คเล่นเกม", budget=budget)
    bad = []
    for d in docs:
        name, price = doc_price(d)
        if price is not None and price > budget:
            bad.append((name, price))
    check(f"กรองงบ {budget:,} — ไม่มีเอกสารเกินงบ", not bad and not over,
          f"ได้ {len(docs)} เอกสาร | เกินงบ={bad or '-'} | over_budget={over}")

# ── 2. งบต่ำกว่ารุ่นถูกสุด -> over_budget ต้องเป็น True ──
docs, over = retrieve("โน้ตบุ๊ค", budget=5000)
check("งบ 5,000 (ต่ำกว่ารุ่นถูกสุด 12,990) -> over_budget=True", over is True,
      f"over_budget={over} | ได้ {len(docs)} เอกสาร (ค้นใหม่แบบไม่กรอง)")

# ── 3. ไม่ระบุงบ -> ไม่กรอง ──
docs, over = retrieve("โน้ตบุ๊คเล่นเกมแรงที่สุด", budget=None)
check("ไม่ระบุงบ -> over_budget=False", over is False,
      f"over_budget={over} | ได้ {len(docs)} เอกสาร")

# ── 4. เอกสารภาพรวมร้านต้องถูกแนบมาเสมอ และต้องไม่กินโควตาเอกสารสเปก ──
docs, over = retrieve("โน้ตบุ๊คทำงาน", budget=20000)
n_summary = sum(1 for d in docs if is_summary(d))
n_spec = sum(1 for d in docs if " บาท | " in d)
check("แนบเอกสารภาพรวมมาด้วยเสมอ", n_summary == 1,
      f"มีเอกสารสรุป {n_summary} อัน")
check("เอกสารสรุปไม่กินโควตาเอกสารสเปก", n_spec >= 1,
      f"สเปก {n_spec} อัน จาก context ทั้งหมด {len(docs)} อัน")

# ── 5. context ต้องมีเอกสารสเปก (ที่มีราคา) ไม่ใช่รีวิวล้วน ──
# นี่คือเคสที่เคยพังก่อนแก้: คำถามเชิงความรู้สึกดึงแต่รีวิว -> LLM ไม่เห็นราคา -> แต่งตัวเลขเอง
for q, budget in [("โน้ตบุ๊คเล่นเกม", 25000), ("แบตอึด พกพาสะดวก", 30000),
                  ("จอสวย ทำงานกราฟิก", 40000), ("เครื่องร้อนไหม", 25000),
                  ("ขอรุ่นที่ราคาถูกที่สุด", None), ("มีรุ่นอื่นให้เทียบอีกไหม", None)]:
    docs, _ = retrieve(q, budget=budget)
    n_spec = sum(1 for d in docs if " บาท | " in d)
    check(f"context มีเอกสารสเปก: '{q}'", n_spec > 0,
          f"สเปก {n_spec}/{len(docs)} เอกสาร | ถ้า 0 = LLM ไม่เห็นราคาเลย")

# ── 6. จำนวนรุ่นที่อยู่ในงบ vs ที่ค้นเจอ ──
docs, _ = retrieve("โน้ตบุ๊คราคาประหยัด", budget=15000)
in_budget_real = sorted(p for p in PRICES.values() if p <= 15000)
names = {doc_price(d)[0] for d in docs if doc_price(d)[0]}
check("งบ 15,000 — ค้นเจอรุ่นในงบจริง", bool(names),
      f"ในฐานข้อมูลมี {len(in_budget_real)} รุ่น <= 15,000 | context ครอบคลุม {len(names)} รุ่น: {sorted(names)}")

print("\n" + "=" * 78)
print(f"RETRIEVAL: ผ่าน {len(PASS)}/{len(PASS)+len(FAIL)}")
print("=" * 78)
for name, detail in FAIL:
    print(f"  [FAIL] {name}\n         {detail}")
sys.exit(1 if FAIL else 0)
