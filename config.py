# config_example.py — copy to config.py and fill with real values.
NEO4J_URI="neo4j+s://d820ed33.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD="JTHuD4y0Ub8_1LKBGYBSDWROTcLAPXF1Fgiq-f3lJtI"

HUGGINGFACE_API_KEY = "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"

# Hugging Face Models
HF_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
HF_CHAT_MODEL = "mistralai/Mistral-7B-Instruct-v0.2"
PINECONE_API_KEY = "pcsk_6vaWqp_4Vz1cXErimsa73Y4ns9B6NHtAPmyKr9F3p1kDARxbB1q9afsSPpaLwjXAijpJgt" # your Pinecone API key
PINECONE_ENV = "us-east-1"   # example
PINECONE_INDEX_NAME = "vietnam-travel"
PINECONE_VECTOR_DIM = 384      # adjust to embedding model used (text-embedding-3-large ~ 3072? check your model); we assume 1536 for common OpenAI models — change if needed.
