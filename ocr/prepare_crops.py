"""
AI-Hub "024.OCR 데이터(교육)" 원천데이터(이미지) + 라벨(bbox+정답텍스트)에서
단어/어절 단위 crop 이미지를 잘라내 패킹된 바이너리로 저장한다.

수백만 개의 작은 crop 파일을 개별 저장하면 NTFS에서 느리므로,
crop을 JPEG로 인코딩해 하나의 .bin 파일에 이어붙이고
(offset, length, text) 인덱스를 별도 json으로 저장한다.
"""
import io
import json
import zipfile
from pathlib import Path

from PIL import Image

SRC_ROOT = Path(r"D:\024.OCR 데이터(교육)\01-1.정식개방데이터")
OUT_DIR = Path(__file__).parent / "crops"
OUT_DIR.mkdir(exist_ok=True)

TARGET_HEIGHT = 32
MAX_WIDTH = 320
MIN_BOX_SIZE = 4
MAX_TEXT_LEN = 20

GRADES = {
    "E": ["G1", "G2", "G3", "G4", "G5", "G6"],
    "M": ["G1", "G2", "G3"],
    "H": ["G1", "G2", "G3"],
}


def bbox_to_rect(box):
    xs, ys = box["x"], box["y"]
    x0, x1 = min(xs), max(xs)
    y0, y1 = min(ys), max(ys)
    return x0, y0, x1, y1


def process_split(split: str, img_prefix: str, lbl_prefix: str, limit_images: int | None = None):
    """split: 'Training' | 'Validation'"""
    img_dir = SRC_ROOT / split / "01.원천데이터"
    lbl_dir = SRC_ROOT / split / "02.라벨링데이터"

    bin_path = OUT_DIR / f"{split.lower()}.bin"
    index_path = OUT_DIR / f"{split.lower()}_index.json"

    index = []
    n_images = 0
    n_crops = 0
    charset = set()

    with open(bin_path, "wb") as binf:
        offset = 0
        for level, grades in GRADES.items():
            for grade in grades:
                img_zip_path = img_dir / f"{img_prefix}_OCR(edu)_{level}_{grade}.zip"
                lbl_zip_path = lbl_dir / f"{lbl_prefix}_OCR(edu)_{level}_{grade}.zip"
                if not img_zip_path.exists() or not lbl_zip_path.exists():
                    print(f"  스킵 (파일 없음): {img_zip_path.name}")
                    continue

                img_zf = zipfile.ZipFile(img_zip_path)
                lbl_zf = zipfile.ZipFile(lbl_zip_path)
                lbl_names = lbl_zf.namelist()

                for lbl_name in lbl_names:
                    stem = Path(lbl_name).stem
                    img_name = f"/{stem}.png"
                    if img_name not in img_zf.namelist():
                        img_name = img_name.lstrip("/")
                        if img_name not in img_zf.namelist():
                            continue

                    try:
                        data = json.loads(lbl_zf.read(lbl_name).decode("utf-8"))
                        img_bytes = img_zf.read(img_name)
                        img = Image.open(io.BytesIO(img_bytes)).convert("L")
                    except Exception:
                        continue

                    n_images += 1
                    for box in data.get("Bbox", []):
                        text = (box.get("data") or "").strip()
                        if not text or len(text) > MAX_TEXT_LEN:
                            continue
                        x0, y0, x1, y1 = bbox_to_rect(box)
                        w, h = x1 - x0, y1 - y0
                        if w < MIN_BOX_SIZE or h < MIN_BOX_SIZE:
                            continue

                        crop = img.crop((x0, y0, x1, y1))
                        new_w = max(1, min(MAX_WIDTH, round(crop.width * TARGET_HEIGHT / crop.height)))
                        crop = crop.resize((new_w, TARGET_HEIGHT), Image.BILINEAR)

                        buf = io.BytesIO()
                        crop.save(buf, format="JPEG", quality=90)
                        jpg_bytes = buf.getvalue()

                        binf.write(jpg_bytes)
                        index.append({"offset": offset, "length": len(jpg_bytes), "text": text})
                        offset += len(jpg_bytes)
                        charset.update(text)
                        n_crops += 1

                    if limit_images and n_images >= limit_images:
                        break
                img_zf.close()
                lbl_zf.close()
                print(f"  [{level}_{grade}] 누적 이미지 {n_images}, crop {n_crops}")
                if limit_images and n_images >= limit_images:
                    break
            if limit_images and n_images >= limit_images:
                break

    with open(index_path, "w", encoding="utf-8") as f:
        json.dump(index, f, ensure_ascii=False)

    print(f"[{split}] 이미지 {n_images}개 → crop {n_crops}개, 고유문자 {len(charset)}개")
    return charset


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-images", type=int, default=None,
                     help="grade당 이미지 수 제한 (스모크테스트용)")
    args = ap.parse_args()

    print("[1/2] Training crop 추출...")
    charset_train = process_split("Training", "TS", "TL", args.limit_images)

    print("[2/2] Validation crop 추출...")
    charset_val = process_split("Validation", "VS", "VL", args.limit_images)

    charset = sorted(charset_train | charset_val)
    with open(OUT_DIR / "charset.json", "w", encoding="utf-8") as f:
        json.dump(charset, f, ensure_ascii=False)
    print(f"charset 저장 완료: {len(charset)}자")


if __name__ == "__main__":
    main()
