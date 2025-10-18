# pinecone_upload.py - Using Hugging Face Embeddings
import json
import time
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from pinecone import Pinecone, ServerlessSpec
import config

# -----------------------------
# Config
# -----------------------------
DATA_FILE = "vietnam_travel_dataset.json"
BATCH_SIZE = 32

INDEX_NAME = config.PINECONE_INDEX_NAME
VECTOR_DIM = config.PINECONE_VECTOR_DIM  # 384 for all-MiniLM-L6-v2

# -----------------------------
# Initialize clients
# -----------------------------
print("Loading Hugging Face embedding model...")
embedding_model = SentenceTransformer(config.HF_EMBEDDING_MODEL)
print(f"Model loaded: {config.HF_EMBEDDING_MODEL}")

pc = Pinecone(api_key=config.PINECONE_API_KEY)

# -----------------------------
# Create managed index if it doesn't exist
# -----------------------------
existing_indexes = pc.list_indexes().names()
if INDEX_NAME not in existing_indexes:
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )
    print("Waiting for index to be ready...")
    time.sleep(10)
else:
    print(f"Index {INDEX_NAME} already exists.")

# Connect to the index
index = pc.Index(INDEX_NAME)

# -----------------------------
# Helper functions
# -----------------------------
def get_embeddings(texts):
    """Generate embeddings using Sentence Transformers."""
    embeddings = embedding_model.encode(
        texts,
        show_progress_bar=False,
        convert_to_numpy=True
    )
    return embeddings.tolist()

def chunked(iterable, n):
    for i in range(0, len(iterable), n):
        yield iterable[i:i+n]

# -----------------------------
# Main upload
# -----------------------------
def main():
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        nodes = json.load(f)

    items = []
    for node in nodes:
        semantic_text = node.get("semantic_text") or (node.get("description") or "")[:1000]
        if not semantic_text.strip():
            continue
        meta = {
            "id": node.get("id"),
            "type": node.get("type"),
            "name": node.get("name"),
            "city": node.get("city", node.get("region", "")),
            "tags": node.get("tags", [])
        }
        items.append((node["id"], semantic_text, meta))

    print(f"Preparing to upsert {len(items)} items to Pinecone...")

    for batch in tqdm(list(chunked(items, BATCH_SIZE)), desc="Uploading batches"):
        ids = [item[0] for item in batch]
        texts = [item[1] for item in batch]
        metas = [item[2] for item in batch]

        # Generate embeddings using Hugging Face model
        embeddings = get_embeddings(texts)

        vectors = [
            {"id": _id, "values": emb, "metadata": meta}
            for _id, emb, meta in zip(ids, embeddings, metas)
        ]

        index.upsert(vectors)
        time.sleep(0.2)

    print("All items uploaded successfully.")
    
    # Print index stats
    time.sleep(2)
    stats = index.describe_index_stats()
    print(f"\nIndex Statistics:")
    print(f"Total vectors: {stats.total_vector_count}")

# -----------------------------
if __name__ == "__main__":
    main()