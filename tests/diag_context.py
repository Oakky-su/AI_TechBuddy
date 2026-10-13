# วินิจฉัย: คำถามแต่ละแบบดึง "เอกสารสเปก" (มีราคา) มาได้กี่อันจาก top-5
# รันใน container: docker exec ai_teddy-teddy-1 python /app/diag_context.py
import sys
sys.path.insert(0, "/app")
from app import retrieve, extract_budget  # noqa: E402

QUESTIONS = [
    "อยากได้โน้ตบุ๊คเล่นเกม งบไม่เกิน 25000",
    "โน้ตบุ๊คเล่นเกม",
    "โน้ตบุ๊คทำงานเอกสาร งบ 20k",
    "แบตอึด พกพาสะดวก งบ 30000",
    "จอสวย ทำงานกราฟิก งบ 40000",
    "เครื่องร้อนไหม",
    "ขอรุ่นที่ราคาถูกที่สุด",
    "Acer Nitro V15 ANV15-41-R488 ราคาเท่าไหร่",
    "มีรุ่นอื่นให้เทียบอีกไหม",
    "RTX 4060 รุ่นไหนดี งบ 40000",
]

print(f"{'คำถาม':<48} {'งบ':>8} {'สเปก':>5} {'รีวิว':>6} {'สรุป':>5}")
print("-" * 80)
for q in QUESTIONS:
    b = extract_budget(q)
    docs, over = retrieve(q, budget=b)
    n_spec = sum(1 for d in docs if " บาท | " in d)
    n_rev = sum(1 for d in docs if d.startswith("รีวิวจากผู้ใช้รุ่น"))
    n_sum = sum(1 for d in docs if d.startswith("ร้านมีโน้ตบุ๊ค"))
    flag = "  <-- ไม่มีราคาใน context เลย" if n_spec == 0 else ""
    print(f"{q[:46]:<48} {str(b):>8} {n_spec:>5} {n_rev:>6} {n_sum:>5}{flag}")

print("\n" + "=" * 80)
print("ตัวอย่าง context ที่ได้รับจริง — 'แบตอึด พกพาสะดวก งบ 30000'")
print("=" * 80)
docs, _ = retrieve("แบตอึด พกพาสะดวก งบ 30000", budget=30000)
for i, d in enumerate(docs, 1):
    print(f"{i}. {d[:150]}")
