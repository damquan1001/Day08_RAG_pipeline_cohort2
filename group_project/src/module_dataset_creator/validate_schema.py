"""Kiểm tra golden_dataset.json trước khi push."""

import json
import sys
from pathlib import Path

DATASET_PATH = Path(__file__).parent / "golden_dataset.json"
MIN_ITEMS = 15
REQUIRED_KEYS = ("question", "expected_answer", "expected_context")


def validate_dataset(path: Path = DATASET_PATH) -> list[str]:
    errors: list[str] = []

    if not path.exists():
        return [f"Khong tim thay file: {path}"]

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [f"JSON khong hop le: {exc}"]

    if not isinstance(data, list):
        errors.append("Root phai la JSON array")
        return errors

    if len(data) < MIN_ITEMS:
        errors.append(f"Can toi thieu {MIN_ITEMS} cau, hien co {len(data)}")

    questions: list[str] = []
    for i, item in enumerate(data, 1):
        if not isinstance(item, dict):
            errors.append(f"Item #{i}: phai la object")
            continue

        for key in REQUIRED_KEYS:
            if key not in item:
                errors.append(f"Item #{i}: thieu truong '{key}'")

        ctx = item.get("expected_context")
        if not isinstance(ctx, list) or not ctx:
            errors.append(f"Item #{i}: expected_context phai la array khong rong")
        elif not all(isinstance(c, str) and c.strip() for c in ctx):
            errors.append(f"Item #{i}: expected_context phai chua chuoi khong rong")

        q = item.get("question", "")
        if not isinstance(q, str) or not q.strip():
            errors.append(f"Item #{i}: question rong")
        elif q.strip() in questions:
            errors.append(f"Item #{i}: question trung lap")
        else:
            questions.append(q.strip())

        ans = item.get("expected_answer", "")
        if not isinstance(ans, str) or len(ans.strip()) < 20:
            errors.append(f"Item #{i}: expected_answer qua ngan")

    return errors


def main() -> int:
    errors = validate_dataset()
    if errors:
        print("[FAIL] golden_dataset.json chua hop le:")
        for err in errors:
            print(f"  - {err}")
        return 1

    count = len(json.loads(DATASET_PATH.read_text(encoding="utf-8")))
    print(f"[OK] golden_dataset.json hop le ({count} cau hoi)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
