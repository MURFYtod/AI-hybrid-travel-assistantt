# Hybrid AI Assistant System

## Overview

This project implements a **hybrid AI retrieval system** combining **Pinecone (vector database)**, **Neo4j (graph database)**, and **LLM-based reasoning** to provide **accurate, contextually enriched answers** to travel-related queries. The system leverages semantic similarity for candidate retrieval and graph relationships for structured reasoning.

## Architecture

1. **Vector Embeddings & Pinecone**

   * Text data is chunked and converted to embeddings using an LLM.
   * Pinecone stores embeddings for semantic retrieval.
   * Metadata per vector enables filtering by source, document ID, or chunk.

2. **Graph Relations & Neo4j**

   * Neo4j stores entities and their relationships (e.g., destinations → attractions → hotels).
   * Graph traversal enriches context and supports relational reasoning.

3. **Hybrid Retrieval Pipeline**

   * **Step 1:** Pinecone retrieves top-K semantically similar candidates.
   * **Step 2:** Neo4j enriches candidates with relational context.
   * **Step 3:** LLM generates the final answer using the combined context.

## Key Features

* **Batch ingestion to Pinecone** for efficient vector upserts.
* **Deterministic vector IDs** for safe repeated runs.
* **Metadata filtering** allows narrowing search results.
* **Modular adapters** for Pinecone and Neo4j to separate data access logic.
* **Interactive CLI** in `hybrid_chat.py` for real-time user queries.
* Optional caching and chain-of-thought reasoning for improved output quality.

## Setup & Usage

1. **Environment Variables**

```bash
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Run ingestion**

```bash
python pinecone_upload.py --data vietnam_travel_dataset.json --batch-size 32 --index-name vietnam-travel
```

4. **Load data to Neo4j**

```bash
python load_to_neo4j.py
```

5. **Visualize graph (optional)**

```bash
python visualize_graph.py
```

6. **Run hybrid query**

```bash
python hybrid_chat.py
```

7. Ask: `create a romantic 4 day itinerary for Vietnam`

## Performance & Scalability

* Handles **current dataset efficiently** with batch ingestion and metadata filtering.
* Structured to allow **future scalability** if datasets grow.
* Caching and modular design support incremental optimizations.

## Highlights & Improvements

* **Hybrid reasoning:** Combines Pinecone embeddings with Neo4j graph relationships for better context and relevance.
* **Batch ingestion & deterministic IDs:** Ensures safe and efficient vector uploads.
* **Metadata filtering:** Reduces unnecessary queries and improves response quality.
* **Modular adapters:** Pinecone and Neo4j logic separated for maintainability and future extensions.
* **CLI & prompt optimization:** Enhanced interactive interface and prompt clarity for better LLM answers.
* **Optional async & caching:** Can be integrated to parallelize requests and speed up response.

## Project Structure

```
├─ config.py          # Environment-based config (tokens, index names)
├─ pinecone_upload.py # Vector ingestion to Pinecone
├─ load_to_neo4j.py  # Load dataset into Neo4j
├─ visualize_graph.py # Optional graph visualization
├─ hybrid_chat.py     # Main interactive query pipeline
├─ pinecone_adapter.py# Adapter for vector DB
├─ neo4j_adapter.py   # Adapter for graph DB
├─ requirements.txt
├─ README.md
└─ datasets/
```

## References

* [Pinecone Documentation](https://www.pinecone.io/docs/)
* [Neo4j Documentation](https://neo4j.com/docs/)
* [Hugging Face Transformers](https://huggingface.co/docs/transformers/index)

---

