from __future__ import annotations

import os
import urllib.parse
from pathlib import Path

try:
    import google.generativeai as genai
except ImportError:
    genai = None

try:
    import weaviate
    from weaviate.classes.config import Configure, DataType, Property
    from weaviate.classes.init import Auth
except ImportError:
    weaviate = None
    Configure = None
    DataType = None
    Property = None
    Auth = None

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None


PROJECT_DIR = Path(__file__).resolve().parent
REPO_DIR = PROJECT_DIR.parent
DATA_DIR = REPO_DIR / "data" / "standardized"
COLLECTION_NAME = "DrugLawDocs"


def load_environment() -> None:
    if load_dotenv is not None:
        load_dotenv(PROJECT_DIR / ".env")
        load_dotenv(REPO_DIR / ".env")


def configure_gemini() -> str:
    api_key = os.getenv("GEMINI_API_KEY", "")
    if api_key and genai is not None:
        genai.configure(api_key=api_key)
    return api_key


def get_embedding(text: str, api_key: str) -> list[float]:
    if not api_key or genai is None:
        return [0.0] * 1024

    for model_name in ["models/text-embedding-004", "models/embedding-001"]:
        try:
            response = genai.embed_content(
                model=model_name,
                content=text,
                task_type="retrieval_document",
            )
            if isinstance(response, dict) and "embedding" in response:
                return response["embedding"]
            if hasattr(response, "embedding"):
                return response.embedding
        except Exception as exc:
            print(f"[Warning] embedding failed with {model_name}: {exc}")
    return [0.0] * 1024


def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        return splitter.split_text(text)
    except ImportError:
        return fallback_chunk_text(text, chunk_size=chunk_size)


def fallback_chunk_text(text: str, chunk_size: int) -> list[str]:
    paragraphs = [paragraph.strip() for paragraph in text.split("\n\n")]
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for paragraph in paragraphs:
        if not paragraph:
            continue
        if current and current_len + len(paragraph) > chunk_size:
            chunks.append("\n\n".join(current))
            current = [paragraph]
            current_len = len(paragraph)
        else:
            current.append(paragraph)
            current_len += len(paragraph) + 2

    if current:
        chunks.append("\n\n".join(current))
    return chunks


def connect_weaviate(url: str, api_key: str):
    if weaviate is None:
        raise RuntimeError(
            "Missing dependency: install weaviate-client before importing data."
        )

    if "weaviate.cloud" in url or "weaviate.network" in url:
        return weaviate.connect_to_weaviate_cloud(
            cluster_url=url,
            auth_credentials=Auth.api_key(api_key) if api_key and Auth else None,
        )

    parsed = urllib.parse.urlparse(url)
    host = parsed.hostname or "localhost"
    port = parsed.port or 8080
    return weaviate.connect_to_local(host=host, port=port)


def collect_markdown_files() -> list[tuple[Path, str]]:
    files: list[tuple[Path, str]] = []
    for doc_type in ["legal", "news"]:
        folder = DATA_DIR / doc_type
        if not folder.exists():
            continue
        files.extend((path, doc_type) for path in folder.glob("*.md"))
    return files


def recreate_collection(client):
    if Configure is None or DataType is None or Property is None:
        raise RuntimeError("Installed weaviate-client does not expose v4 classes.")

    if client.collections.exists(COLLECTION_NAME):
        print(f"Collection {COLLECTION_NAME} exists. Deleting before re-import...")
        client.collections.delete(COLLECTION_NAME)

    print(f"Creating collection {COLLECTION_NAME}")
    return client.collections.create(
        name=COLLECTION_NAME,
        vectorizer_config=Configure.Vectorizer.none(),
        properties=[
            Property(name="content", data_type=DataType.TEXT),
            Property(name="source", data_type=DataType.TEXT),
            Property(name="doc_type", data_type=DataType.TEXT),
        ],
    )


def main() -> int:
    load_environment()
    gemini_key = configure_gemini()
    weaviate_url = os.getenv("WEAVIATE_URL", "http://localhost:8080")
    weaviate_key = os.getenv("WEAVIATE_API_KEY", "")

    files = collect_markdown_files()
    if not files:
        print(f"[Error] No markdown files found under {DATA_DIR}")
        return 1

    print(f"Connecting to Weaviate: {weaviate_url}")
    client = connect_weaviate(weaviate_url, weaviate_key)
    try:
        collection = recreate_collection(client)
        objects = []
        for file_path, doc_type in files:
            print(f"Processing {file_path.name}")
            text = file_path.read_text(encoding="utf-8")
            for chunk in chunk_text(text):
                objects.append(
                    {
                        "properties": {
                            "content": chunk,
                            "source": file_path.name,
                            "doc_type": doc_type,
                        },
                        "vector": get_embedding(chunk, gemini_key),
                    }
                )

        print(f"Uploading {len(objects)} chunks")
        with collection.batch.dynamic() as batch:
            for item in objects:
                batch.add_object(
                    properties=item["properties"],
                    vector=item["vector"],
                )
    finally:
        client.close()

    print("Import completed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
