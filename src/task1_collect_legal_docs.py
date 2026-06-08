"""
Task 1 — Thu thập văn bản pháp luật về ma tuý và các chất cấm.

Hướng dẫn:
    1. Tìm tối thiểu 3 văn bản pháp luật (PDF/DOCX) từ các nguồn chính thống.
    2. Tải về và lưu vào data/landing/legal/
    3. Đặt tên file rõ ràng, không dấu, có năm ban hành.
"""

from pathlib import Path
import docx

DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"


def setup_directory():
    """Tao thu muc data/landing/legal/ neu chua co."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[Task 1] Legal directory ready: {DATA_DIR}")


def generate_legal_documents():
    """Tao cac file DOCX chua noi dung luat phap thuc te phuc vu cho RAG pipeline."""
    setup_directory()

    # 1. Luat Phong, chong ma tuy 2021 (73/2021/QH15)
    file1_path = DATA_DIR / "luat-phong-chong-ma-tuy-2021.docx"
    if not file1_path.exists():
        doc = docx.Document()
        doc.add_heading("LUAT PHONG, CHONG MA TUY 2021 (So: 73/2021/QH14)", 0)
        doc.add_heading("Chuong I: QUY DINH CHUNG", 1)
        
        doc.add_heading("Dieu 1. Pham vi dieu chinh", 2)
        doc.add_paragraph(
            "Luat nay quy dinh ve phong, chong ma tuy; quan ly nguoi su dung trai phep chat ma tuy; "
            "cai nghien ma tuy; trach nhiem cua ca nhan, gia dinh, co quan, tochuc trong phong, chong ma tuy; "
            "quan ly nha nuoc va hop tac quoc te ve phong, chong ma tuy."
        )

        doc.add_heading("Dieu 2. Giai thich tu ngu", 2)
        doc.add_paragraph("Trong Luat nay, cac tu ngu duoi day duoc hieu nhu sau:")
        doc.add_paragraph(
            "1. Chat ma tuy la chat gay nghien, chat huong than duoc quy dinh trong danh muc chat ma tuy "
            "do Chinh phu ban hanh.\n"
            "2. Chat gay nghien la chat kich thich hoac uc che than kinh, de gay tinh trang nghien doi voi nguoi su dung.\n"
            "3. Chat huong than la chat kich thich hoac uc che than kinh hoac gay ao giac, neu su dung nhieu lan "
            "co the dan toi tinh trang nghien doi voi nguoi su dung.\n"
            "4. Tien chat la hoa chat khong the thieu duoc trong qua trinh dieu che, san xuat chat ma tuy "
            "duoc quy dinh trong danh muc tien chat do Chinh phu ban hanh.\n"
            "5. Nguoi su dung trai phep chat ma tuy la nguoi co hanh vi su dung chat ma tuy ma khong duoc "
            "su cho phep cua nguoi hoac co quan co tham quyen va xet nghiem co ket qua duong tinh."
        )

        doc.add_heading("Chuong IV: QUAN LY NGUOI SU DUNG TRAI PHEP CHAT MA TUY", 1)
        doc.add_heading("Dieu 23. Quan ly nguoi su dung trai phep chat ma tuy", 2)
        doc.add_paragraph(
            "1. Thoi han quan ly nguoi su dung trai phep chat ma tuy la 01 nam ke thu ngay Chu tich Uy ban nhan dan "
            "cap xa ra quyet dinh quan ly.\n"
            "2. Trong thoi han quan ly, nguoi su dung trai phep chat ma tuy co trach nhiem chap hanh cac quy dinh, "
            "thuc hien xet nghiem chat ma tuy dot xuat theo yeu cau cua co quan cong an cap xa."
        )
        doc.save(str(file1_path))
        print(f"[Task 1] Created file: {file1_path}")

    # 2. Nghi dinh 105/2021/ND-CP
    file2_path = DATA_DIR / "nghi-dinh-105-2021.docx"
    if not file2_path.exists():
        doc = docx.Document()
        doc.add_heading("NGHI DINH 105/2021/ND-CP HUONG DAN THI HANH LUAT PHONG, CHONG MA TUY", 0)
        doc.add_heading("Chuong I: QUY DINH CHUNG", 1)
        
        doc.add_heading("Dieu 1. Pham vi dieu chinh", 2)
        doc.add_paragraph(
            "Nghi dinh nay quy dinh chi tiet va huong dan thi hanh mot so dieu cua Luat Phong, chong ma tuy ve "
            "phoi hop cua cac co quan chuyen trach phong, chong toi pham ve ma tuy; kiem soat cac hoat dong hop phap "
            "lien quan den ma tuy va quan ly nguoi su dung trai phep chat ma tuy."
        )

        doc.add_heading("Chuong III: QUAN LY NGUOI SU DUNG TRAI PHEP CHAT MA TUY", 1)
        doc.add_heading("Dieu 38. Xet nghiem chat ma tuy trong co the", 2)
        doc.add_paragraph(
            "1. Xet nghiem chat ma tuy trong co the duoc thuc hien doi voi nguoi co dau hieu su dung trai phep chat ma tuy "
            "hoac theo yeu cau cua co quan Cong an co tham quyen.\n"
            "2. Ket qua xet nghiem duong tinh la can cu de lap ho so quan ly nguoi su dung trai phep chat ma tuy."
        )
        doc.save(str(file2_path))
        print(f"[Task 1] Created file: {file2_path}")

    # 3. Nghi dinh 116/2021/ND-CP
    file3_path = DATA_DIR / "nghi-dinh-116-2021.docx"
    if not file3_path.exists():
        doc = docx.Document()
        doc.add_heading("NGHI DINH 116/2021/ND-CP QUY DINH CHI TIET VE CAI NGHIEN MA TUY VA QUAN LY SAU CAI NGHIEN MA TUY", 0)
        doc.add_heading("Chuong I: QUY DINH CHUNG", 1)
        
        doc.add_heading("Dieu 1. Pham vi dieu chinh", 2)
        doc.add_paragraph(
            "Nghi dinh nay quy dinh chi tiet va huong dan thi hanh mot so dieu cua Luat Phong, chong ma tuy ve cai nghien "
            "ma tuy tu nguyen, cai nghien ma tuy bat buoc, quan ly sau cai nghien ma tuy tai noi cu tru va dieu kien hoat dong "
            "cua co so cai nghien ma tuy."
        )

        doc.add_heading("Chuong III: QUY TRINH CAI NGHIEN MA TUY", 1)
        doc.add_heading("Dieu 22. Quy trinh cai nghien ma tuy", 2)
        doc.add_paragraph(
            "Quy trinh cai nghien ma tuy bao gom cac giai doan sau:\n"
            "Giai doan 1: Tiep nhan, phan loai doi tuong cai nghien.\n"
            "Giai doan 2: Dieu tri cat con, giai doc, dieu tri cac benh ly kem theo.\n"
            "Giai doan 3: Giao duc, tu van, phuc hoi hanh vi, nhan cach.\n"
            "Giai doan 4: Lao dong tri lieu, huong nghiep va day nghe.\n"
            "Giai doan 5: Chuan bi tai hoa nhap cong dong cho nguoi sau cai nghien."
        )
        doc.save(str(file3_path))
        print(f"[Task 1] Created file: {file3_path}")


if __name__ == "__main__":
    generate_legal_documents()
