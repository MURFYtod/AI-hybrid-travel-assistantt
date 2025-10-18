# ==============================================================
# hybrid_chat.py - Optimized Hybrid AI Chat (with async + caching)
# ==============================================================

import json
import asyncio
import aiohttp
import time
from typing import List, Dict
from functools import lru_cache
from sentence_transformers import SentenceTransformer
from huggingface_hub import InferenceClient
from pinecone import Pinecone, ServerlessSpec
from neo4j import GraphDatabase
import config

# ==============================================================
# CONFIG
# ==============================================================
EMBED_MODEL = config.HF_EMBEDDING_MODEL
CHAT_MODEL = config.HF_CHAT_MODEL
TOP_K = 5
INDEX_NAME = config.PINECONE_INDEX_NAME

# ==============================================================
# INITIALIZATION
# ==============================================================
print("Loading embedding model...")
embedding_model = SentenceTransformer(EMBED_MODEL)
print(f"✅ Embedding model loaded: {EMBED_MODEL}")

print("Initializing Hugging Face Inference Client...")
hf_client = InferenceClient(token=config.HUGGINGFACE_API_KEY)
print(f"✅ Chat model: {CHAT_MODEL}")

# Pinecone setup
pc = Pinecone(api_key=config.PINECONE_API_KEY)
if INDEX_NAME not in pc.list_indexes().names():
    print(f"Creating managed index: {INDEX_NAME}")
    pc.create_index(
        name=INDEX_NAME,
        dimension=config.PINECONE_VECTOR_DIM,
        metric="cosine",
        spec=ServerlessSpec(cloud="aws", region="us-east-1")
    )
index = pc.Index(INDEX_NAME)

# Neo4j setup
driver = GraphDatabase.driver(
    config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASSWORD)
)

# ==============================================================
# === Improvement: Embedding Cache (Bonus Insight) ============
# This reduces redundant encoding calls for repeated queries.
# ==============================================================
from functools import lru_cache
from typing import List


embedding_cache = {}  # Cache for embeddings
response_cache = {}   # Cache for chat responses

def embed_text(text: str) -> List[float]:
    """Return cached embeddings to avoid redundant computation."""
    if text in embedding_cache:
        print("✅ [CACHE HIT] Using stored embedding.")
        return embedding_cache[text]

    print("⚙️ [CACHE MISS] Generating new embedding...")
    embedding = embedding_model.encode(text, show_progress_bar=False, convert_to_numpy=True).tolist()
    embedding_cache[text] = embedding
    return embedding

def call_chat_cached(prompt: str, max_tokens=800, stream=False):
    """Return cached chat response if available, else generate and cache."""
    if prompt in response_cache:
        print("✅ [CACHE HIT] Returning stored chat response.")
        return response_cache[prompt]

    print("⚙️ [CACHE MISS] Generating new chat response...")
    response = call_chat(prompt, max_tokens=max_tokens, stream=stream)
    response_cache[prompt] = response
    return response
# ==============================================================
# === Pinecone Query ===========================================
# ==============================================================
async def pinecone_query_async(query_text: str, top_k=TOP_K) -> List[Dict]:
    """Perform async vector search on Pinecone."""
    vec = embed_text(query_text)
    res = index.query(
        vector=vec,
        top_k=top_k,
        include_metadata=True,
        include_values=False
    )
    print(f"DEBUG: Pinecone returned {len(res['matches'])} results")
    return res["matches"]

# ==============================================================
# === Neo4j Query (Async with Thread Executor) ================
# ==============================================================
async def fetch_graph_context_async(node_ids: List[str]) -> List[Dict]:
    """Fetch graph relations asynchronously from Neo4j."""
    facts = []
    loop = asyncio.get_event_loop()

    def run_query(nid):
        with driver.session() as session:
            q = (
                "MATCH (n:Entity {id:$nid})-[r]-(m:Entity) "
                "RETURN type(r) AS rel, labels(m) AS labels, m.id AS id, "
                "m.name AS name, m.type AS type, m.description AS description "
                "LIMIT 10"
            )
            recs = session.run(q, nid=nid)
            return [
                {
                    "source": nid,
                    "rel": r["rel"],
                    "target_id": r["id"],
                    "target_name": r["name"],
                    "target_desc": (r["description"] or "")[:400],
                    "labels": r["labels"]
                }
                for r in recs
            ]

    tasks = [loop.run_in_executor(None, run_query, nid) for nid in node_ids]
    results = await asyncio.gather(*tasks)
    for group in results:
        facts.extend(group)

    print(f"DEBUG: Graph returned {len(facts)} facts")
    return facts

# ==============================================================
# === Improvement: Context Summarization Helper ================
# ==============================================================
def search_summary(pinecone_matches, graph_facts) -> str:
    """Summarize top nodes and common relationships."""
    top_nodes = [m["metadata"].get("name", "Unknown") for m in pinecone_matches[:5]]
    rels = list({f["rel"] for f in graph_facts})
    summary = (
        f"✅ Top Nodes: {', '.join(top_nodes)}\n"
        f"🔗 Common Relations: {', '.join(rels)}"
    )
    return summary

# ==============================================================
# === Build Prompt ============================================
# ==============================================================
def build_prompt(user_query, pinecone_matches, graph_facts):
    vec_context = []
    for m in pinecone_matches[:10]:
        meta = m["metadata"]
        score = m.get("score", None)
        snippet = f"- {meta.get('name','')} ({meta.get('type','')}) - ID: {m['id']}"
        if meta.get("city"):
            snippet += f" | City: {meta.get('city')}"
        if score:
            snippet += f" | Relevance: {score:.3f}"
        vec_context.append(snippet)

    graph_context = []
    for f in graph_facts[:20]:
        graph_context.append(
            f"- {f['source']} --[{f['rel']}]--> {f['target_name']} ({f['target_id']}): {f['target_desc'][:150]}"
        )

    prompt = f"""
System Prompt:
You are a professional AI travel planner.
Use the vector search results and graph context to craft a **realistic, engaging, and personalized travel itinerary** for Vietnam.
- Format it **day-wise** (Day 1, Day 2, etc.)
- Include **attractions, activities, local cuisine, and IDs**
- Write in a **friendly, visually appealing tone** using emojis and bold headers
- Avoid repetition and end with a short **farewell line**

User Question:
{user_query}

Relevant Context (Semantic Search):
{chr(10).join(vec_context)}

Knowledge Graph Insights:
{chr(10).join(graph_context)}

Answer:
"""
    return prompt

# ==============================================================
# === Hugging Face Chat Call ==================================
# ==============================================================
def call_chat(prompt: str, max_tokens=800, stream=False):
    """Send prompt to Hugging Face chat model."""
    try:
        messages = [{"role": "user", "content": prompt}]
        out = hf_client.chat_completion(
            model=CHAT_MODEL,
            messages=messages,
            max_tokens=max_tokens,
            temperature=0.7,
            stream=stream,
        )
        if hasattr(out, "choices") and len(out.choices) > 0:
            return out.choices[0].message.content.strip()
        return str(out)
    except Exception as e_chat:
        # Fallback if chat API fails
        print("⚠️ Chat mode failed, switching to text_generation.")
        try:
            resp = hf_client.text_generation(
                model=CHAT_MODEL,
                prompt=prompt,
                max_new_tokens=max_tokens,
                temperature=0.7,
            )
            if isinstance(resp, dict):
                return resp.get("generated_text", "").strip()
            return str(resp).strip()
        except Exception as e_text:
            return f"[Error] chat_error: {e_chat} | textgen_error: {e_text}"

# ==============================================================
# === Async Hybrid Chat Runner ================================
# ==============================================================
async def handle_query(query: str):
    """Fetch Pinecone + Neo4j context in parallel and generate chat."""
    start_time = time.time()

    pinecone_task = asyncio.create_task(pinecone_query_async(query))
    pinecone_matches = await pinecone_task
    node_ids = [m["id"] for m in pinecone_matches]

    graph_task = asyncio.create_task(fetch_graph_context_async(node_ids))
    graph_facts = await graph_task

    print("\n📊 Context Summary:")
    print(search_summary(pinecone_matches, graph_facts))

    prompt = build_prompt(query, pinecone_matches, graph_facts)

    print("\n🤖 Generating travel plan...")
    response = call_chat(prompt)
    print(f"\n✅ Completed in {time.time() - start_time:.2f}s")

    return response

# ==============================================================
# === Interactive CLI =========================================
# ==============================================================
def interactive_chat():
    print("\n" + "="*60)
    print("🌏 Vietnam Travel Assistant (Async + Cached + Hugging Face)")
    print("="*60)
    print("Type 'exit' to quit\n")

    loop = asyncio.get_event_loop()
    while True:
        query = input("\nEnter your travel question: ").strip()
        if not query or query.lower() in ("exit", "quit"):
            print("Thank you for using the Vietnam Travel Assistant! 👋")
            break
        print("\n🔍 Fetching and processing asynchronously...")
        response = loop.run_until_complete(handle_query(query))
        print("\n" + "="*60)
        print("📝 ANSWER:")
        print("="*60)
        print(response)
        print("="*60)

# ==============================================================
# MAIN
# ==============================================================
if __name__ == "__main__":
    interactive_chat()
