import chromadb

client = chromadb.PersistentClient(path="./chromadb")

collection = client.get_or_create_collection(
    name="production_knowledge_base",
    metadata={"hnsw:space": "cosine"}
)

print("Collection successfully initialized")
print(f"Collection Name: {collection.name}")
print(f"Collection Document Count: {collection.count()}")


collection.upsert(
    ids=["doc_1","doc_2","doc_3"],
    documents = [
        "ChromaDB is an open-source vector database built for AI applications and vector search.",
        "Python is a high-level programming language widely used in data science and machine learning.",
        "Vector databases store high-dimensional embeddings to enable fast nearest-neighbor retrieval."
    ],
    metadatas=[
        {"category": "database", "priority": "high"},
        {"category": "language", "priority": "medium"},
        {"category": "database", "priority": "high"}
    ]
)

print(f"Updated Document Count: {collection.count()}")

query_text = "How do vector stores search embeddings?"

results = collection.query(
    query_texts=[query_text],
    n_results=2,
    where={"category":"database"}
)

print("\n--- Similarity Search Results ---")
print(f"Query: '{query_text}'\n")

for i in range(len(results['ids'][0])):
    print(f"Rank {i+1}:")
    print(f"  ID:       {results['ids'][0][i]}")
    print(f"  Document: {results['documents'][0][i]}")
    print(f"  Document: {results['metadatas'][0][i]}")
    print(f"  Document: {results['distances'][0][i]:.4f}")