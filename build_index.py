# สคริปต์สร้าง Vector Index — รันครั้งเดียวก่อนเริ่มใช้งานแอป (หรือรันตอน build Docker image)
# ขั้นตอน: อ่าน laptops.json -> แปลงเป็นประโยคภาษาไทย -> embedding ด้วย bge-m3 -> เก็บลง Chroma (persistent)
# การใช้งาน: python build_index.py

import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import chromadb

BASE_DIR = Path(__file__).parent
JSON_PATH = BASE_DIR / "laptops.json"
CHROMA_DIR = BASE_DIR / "chroma_db"      # โฟลเดอร์เก็บ vector DB แบบถาวร
COLLECTION_NAME = "laptops"
EMBED_MODEL = "BAAI/bge-m3"              # โมเดล embedding หลายภาษา (~2GB)


def laptops_to_documents(laptops):
    """แปลงข้อมูลโน้ตบุ๊คแต่ละรุ่นเป็นประโยคภาษาไทยสำหรับทำ embedding
    คืนค่า 3 ลิสต์ขนานกัน: documents (ข้อความ), metadatas, ids
    - 1 เอกสาร "สเปก" ต่อ 1 รุ่น (ตัวหลักสำหรับค้นหา)
    - 1 เอกสาร "รีวิว" ต่อ 1 รีวิว (ติดชื่อรุ่นไว้ให้ค้นเจอง่าย)
    """
    documents, metadatas, ids = [], [], []

    for lp in laptops:
        name = lp["ชื่อสินค้า"]
        price = lp["ราคา"]
        spec = lp.get("spec", {})
        bench = lp.get("benchmark", {})
        no = lp.get("ลำดับ", len(ids))

        # ── เอกสารสเปก ──
        spec_sentence = (
            f"{name} ราคา {price:,} บาท | "
            f"ซีพียู (CPU): {spec.get('CPU','-')} | "
            f"การ์ดจอ (GPU): {spec.get('GPU','-')} | "
            f"แรม (RAM): {spec.get('RAM','-')} | "
            f"หน้าจอ (DISPLAY): {spec.get('DISPLAY','-')} | "
            f"ความจุ (STORAGE): {spec.get('STORAGE','-')} | "
            f"ระบบปฏิบัติการ (OS): {spec.get('OS','-')} | "
            f"คะแนนเบนช์มาร์ก CPU {bench.get('CPU Score','-')} / GPU {bench.get('GPU Score','-')}"
        )
        documents.append(spec_sentence)
        metadatas.append({"type": "spec", "name": name, "price": price})
        ids.append(f"spec_{no}")

        # ── เอกสารรีวิว ──
        for j, review in enumerate(lp.get("reviews", [])):
            documents.append(f"รีวิวจากผู้ใช้รุ่น {name}: {review}")
            metadatas.append({"type": "review", "name": name, "price": price})
            ids.append(f"review_{no}_{j}")

    # ── เอกสารสรุปภาพรวมร้าน (ตอบ "มีกี่รุ่น / ราคาเริ่มต้น-สูงสุด") ──
    prices = [lp["ราคา"] for lp in laptops]
    documents.append(
        f"ร้านมีโน้ตบุ๊คให้เลือกทั้งหมด {len(laptops)} รุ่น "
        f"ราคาเริ่มต้น {min(prices):,} บาท ไปจนถึง {max(prices):,} บาท"
    )
    metadatas.append({"type": "summary", "name": "ภาพรวม", "price": 0})
    ids.append("summary_0")

    return documents, metadatas, ids


def build():
    print("Loading laptops.json...")
    laptops = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    print(f"  โหลดข้อมูลโน้ตบุ๊คได้ {len(laptops)} รุ่น")

    documents, metadatas, ids = laptops_to_documents(laptops)
    n_spec = sum(m["type"] == "spec" for m in metadatas)
    n_review = sum(m["type"] == "review" for m in metadatas)
    print(f"  สร้างได้ {len(documents)} เอกสาร (สเปก {n_spec} + รีวิว {n_review} + สรุป 1)")

    print(f"Loading embedding model ({EMBED_MODEL})... ครั้งแรกจะดาวน์โหลด ~2GB")
    embedder = SentenceTransformer(EMBED_MODEL)

    print("Generating embeddings...")
    embeddings = embedder.encode(
        documents, normalize_embeddings=True, show_progress_bar=True, batch_size=32
    ).tolist()

    print("Writing Chroma vector DB...")
    client = chromadb.PersistentClient(path=str(CHROMA_DIR))
    try:
        client.delete_collection(COLLECTION_NAME)  # ลบของเก่าถ้ารันซ้ำ
    except Exception:
        pass
    collection = client.create_collection(COLLECTION_NAME)
    collection.add(ids=ids, documents=documents, embeddings=embeddings, metadatas=metadatas)

    print(f"เสร็จแล้ว! เก็บลง vector DB {collection.count()} เอกสาร ที่: {CHROMA_DIR}")


if __name__ == "__main__":
    build()
