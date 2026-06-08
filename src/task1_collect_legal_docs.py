"""Task 1 - Validate collected Vietnamese drug-law documents."""

from pathlib import Path


DATA_DIR = Path(__file__).parent.parent / "data" / "landing" / "legal"
VALID_EXTENSIONS = {".pdf", ".docx", ".doc"}


def setup_directory():
    """Create data/landing/legal/ if needed."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Directory ready: {DATA_DIR}")


def list_legal_documents() -> list[Path]:
    """Return visible legal PDF/DOC/DOCX files."""
    if not DATA_DIR.exists():
        return []
    return [
        path
        for path in sorted(DATA_DIR.iterdir())
        if path.is_file() and path.suffix.lower() in VALID_EXTENSIONS
    ]


def validate_legal_documents(min_files: int = 3, min_size_bytes: int = 1024) -> list[Path]:
    """Validate that required real legal documents are present."""
    setup_directory()
    files = list_legal_documents()
    if len(files) < min_files:
        raise FileNotFoundError(
            f"Expected at least {min_files} legal PDF/DOC/DOCX files in {DATA_DIR}; "
            f"found {len(files)}."
        )

    too_small = [path.name for path in files if path.stat().st_size <= min_size_bytes]
    if too_small:
        raise ValueError(f"Legal document files are too small: {too_small}")

    return files


if __name__ == "__main__":
    files = validate_legal_documents()
    print(f"Found {len(files)} legal documents")
