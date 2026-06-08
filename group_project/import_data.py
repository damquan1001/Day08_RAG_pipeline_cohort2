import os
from pathlib import Path
import urllib.parse
import weaviate
from weaviate.classes.init import Auth
from weaviate.classes.config import Configure, Property, DataType
import google.generativeai as genai
from dotenv import load_dotenv

project_dir = Path(__file__).parent
load_dotenv(project_dir / ".env")

# Configure Gemini
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

def get_embedding(text: str) -> list[float]:
    """Generate embedding vector using text-embedding-004 with fallback."""
    if not api_key:
        return [0.0] * 1024
    
    # Try text-embedding-004
    try:
        response = genai.embed_content(
            model="models/text-embedding-004",
            content=text,
            task_type="retrieval_document"
        )
        if isinstance(response, dict) and "embedding" in response:
            return response["embedding"]
        elif hasattr(response, "embedding"):
            return response.embedding
        return [0.0] * 1024
    except Exception as e:
        # Try falling back to embedding-001
        try:
            response = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            if isinstance(response, dict) and "embedding" in response:
                return response["embedding"]
            elif hasattr(response, "embedding"):
                return response.embedding
            return [0.0] * 1024
        except Exception as e2:
            print(f"[Warning] Failed to generate embedding with both models: {e2}")
            return [0.0] * 1024

def chunk_text(text: str, chunk_size: int = 1000, chunk_overlap: int = 200) -> list[str]:
    """Chunk document text into smaller segments."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        return splitter.split_text(text)
    except ImportError:
        paragraphs = text.split("\n\n")
        chunks = []
        current_chunk = []
        current_len = 0
        for p in paragraphs:
            p = p.strip()
            if not p:
                continue
            if current_len + len(p) > chunk_size:
                if current_chunk:
                    chunks.append("\n\n".join(current_chunk))
                current_chunk = [p]
                current_len = len(p)
            else:
                current_chunk.append(p)
                current_len += len(p) + 2
        if current_chunk:
            chunks.append("\n\n".join(current_chunk))
        return chunks

def main():
    weaviate_url = os.getenv("WEAVIATE_URL")
    weaviate_key = os.getenv("WEAVIATE_API_KEY")
    
    if not weaviate_url or not weaviate_key:
        print("[Loi] Chua cau hinh WEAVIATE_URL hoac WEAVIATE_API_KEY trong file .env")
        return
        
    print(f"Connecting to Weaviate Cloud at: {weaviate_url}")
    client = weaviate.connect_to_weaviate_cloud(
        cluster_url=weaviate_url,
        auth_credentials=Auth.api_key(weaviate_key)
    )
    
    try:
        class_name = "DrugLawDocs"
        if client.collections.exists(class_name):
            print(f"Collection {class_name} already exists. Deleting to re-import...")
            client.collections.delete(class_name)
            
        print(f"Creating Collection: {class_name}")
        collection = client.collections.create(
            name=class_name,
            vectorizer_config=Configure.Vectorizer.none(),
            properties=[
                Property(name="content", data_type=DataType.TEXT),
                Property(name="source", data_type=DataType.TEXT),
                Property(name="doc_type", data_type=DataType.TEXT),
            ]
        )
        
        data_dir = project_dir.parent / "data" / "standardized"
        files_to_process = []
        for file_path in (data_dir / "legal").glob("*.md"):
            files_to_process.append((file_path, "legal"))
        for file_path in (data_dir / "news").glob("*.md"):
            files_to_process.append((file_path, "news"))
            
        print(f"Found {len(files_to_process)} document files to process.")
        
        all_objects = []
        for file_path, doc_type in files_to_process:
            print(f"Processing file: {file_path.name}")
            text = file_path.read_text(encoding="utf-8")
            chunks = chunk_text(text)
            
            for chunk in chunks:
                vector = get_embedding(chunk)
                all_objects.append({
                    "properties": {
                        "content": chunk,
                        "source": file_path.name,
                        "doc_type": doc_type
                    },
                    "vector": vector
                })
                
        print(f"Created {len(all_objects)} chunks with vector embeddings.")
        print("Uploading data to Weaviate Cloud...")
        
        with collection.batch.dynamic() as batch:
            for obj in all_objects:
                batch.add_object(
                    properties=obj["properties"],
                    vector=obj["vector"]
                )
                
        print("✓ Uploaded data to Weaviate Cloud successfully!")
        
    finally:
        client.close()
        print("Closed Weaviate connection.")

if __name__ == "__main__":
    main()
