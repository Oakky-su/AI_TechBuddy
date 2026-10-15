"""กรอกคำตอบและผลทดสอบกลับลงไฟล์ rag_notebook_test_questions.xlsx

เขียนคอลัมน์ F (คำตอบจากระบบ) / G (ผ่าน-ไม่ผ่าน) / H (หมายเหตุ)
โดยไม่ใช้ openpyxl — แก้ XML ใน .xlsx ตรง ๆ แล้วบันทึกเป็นไฟล์ใหม่

ใช้: python tests/fill_excel.py
"""
import json
import os
import re
import shutil
import sys
import zipfile
from xml.sax.saxutils import escape

sys.stdout.reconfigure(encoding="utf-8")

NS = "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}"
SRC = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "..", "rag_notebook_test_questions.xlsx")
SRC = os.path.abspath(SRC)
DST = SRC.replace(".xlsx", "_ผลทดสอบ.xlsx")
RESULTS = os.path.join(os.environ.get("TEMP", "."), "excel_results.json")

MAX_CELL = 900          # ตัดคำตอบไม่ให้เซลล์ยาวเกินจนเปิดยาก


def main():
    results = {r["no"]: r for r in json.load(open(RESULTS, encoding="utf-8"))}
    shutil.copy(SRC, DST)

    z = zipfile.ZipFile(SRC)
    sheet = z.read("xl/worksheets/sheet1.xml").decode("utf-8")

    # เขียนค่าเป็น inline string (t="inlineStr") จะได้ไม่ต้องยุ่งกับ sharedStrings
    def cell_xml(ref, text):
        return (f'<c r="{ref}" t="inlineStr"><is><t xml:space="preserve">'
                f'{escape(str(text)[:MAX_CELL])}</t></is></c>')

    filled = 0
    for r in results.values():
        row = r["row"]
        new_cells = "".join([
            cell_xml(f"F{row}", r["answer"].replace("\n", " ")),
            cell_xml(f"G{row}", r["verdict"]),
            cell_xml(f"H{row}", r["why"] or "-"),
        ])
        # หา <row r="N" ...> ... </row> แล้วแทรก/แทนที่เซลล์ F,G,H
        pat = re.compile(rf'(<row[^>]*\br="{row}"[^>]*>)(.*?)(</row>)', re.S)
        m = pat.search(sheet)
        if not m:
            continue
        body = re.sub(r'<c r="[FGH]\d+".*?(?:/>|</c>)', "", m.group(2), flags=re.S)
        sheet = sheet[:m.start()] + m.group(1) + body + new_cells + m.group(3) + sheet[m.end():]
        filled += 1

    # เขียน xlsx ใหม่ (คัดลอกทุกไฟล์เดิม ยกเว้น sheet1 ที่แทนด้วยตัวแก้แล้ว)
    with zipfile.ZipFile(DST, "w", zipfile.ZIP_DEFLATED) as out:
        for item in z.infolist():
            data = sheet.encode("utf-8") if item.filename == "xl/worksheets/sheet1.xml" \
                else z.read(item.filename)
            out.writestr(item, data)
    z.close()

    n_pass = sum(1 for r in results.values() if r["verdict"] == "ผ่าน")
    print(f"กรอกผลแล้ว {filled} แถว — ผ่าน {n_pass}/{len(results)}")
    print(f"บันทึกไฟล์ใหม่: {DST}")


if __name__ == "__main__":
    main()
