# ชุดทดสอบ logic ล้วน (ไม่ต้องใช้โมเดล/API) — รัน: python tests/test_unit.py
# ครอบคลุม: extract_budget(), strip_think(), laptops_to_documents(), ความสมบูรณ์ของ laptops.json
import json
import sys
import types
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

# stub chromadb — ไม่ได้ติดตั้งบนเครื่อง host (อยู่ใน Docker image) และเทสต์ชุดนี้ไม่ได้ใช้
if "chromadb" not in sys.modules:
    stub = types.ModuleType("chromadb")
    stub.PersistentClient = lambda *a, **k: None
    sys.modules["chromadb"] = stub

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


if __name__ == "__main__":
    test_budget()
    test_strip_think()
    test_documents()
    test_data()

    print(f"\n{'='*78}\nUNIT TESTS: ผ่าน {len(PASS)} / {len(PASS)+len(FAIL)}\n{'='*78}")
    if FAIL:
        print(f"\n❌ ไม่ผ่าน {len(FAIL)} เคส:\n")
        for group, case, exp, act, note in FAIL:
            print(f"  [{group}] {case}")
            print(f"      ควรได้: {exp!r}   แต่ได้: {act!r}   ({note})")
    else:
        print("\n✅ ผ่านทั้งหมด")
    sys.exit(1 if FAIL else 0)
