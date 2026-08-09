import math
from collections import Counter

docs = [
    {
        "id": 1,
        "text": "PostgreSQL is a powerful relational database supporting SQL queries and JSON storage.",
        "category": "database",
        "year": 2021,
        "views": 1500
    },
    {
        "id": 2,
        "text": "Qdrant and Pinecone are specialized vector databases designed for fast similarity search.",
        "category": "database",
        "year": 2023,
        "views": 3200
    },
    {
        "id": 3,
        "text": "Python and TypeScript are popular programming languages for modern web development.",
        "category": "programming",
        "year": 2022,
        "views": 4500
    },
    {
        "id": 4,
        "text": "Hybrid search combines dense vector embeddings with sparse BM25 keyword ranking.",
        "category": "search",
        "year": 2023,
        "views": 2100
    },
    {
        "id": 5,
        "text": "Elasticsearch provides robust full-text BM25 inverted index capabilities.",
        "category": "search",
        "year": 2020,
        "views": 1800
    }
]

def filter_docs(docs,filters):
    filtered=[]
    for doc in docs:
        match=True
        for key,expected_val in filters.items():
            if doc.get(key) != expected_val:
                match = False
                break
        if match:
            filtered.append(doc)
    return filtered

def compute_bm25_scores(query, docs):
    query_tokens = query.lower().split()
    total_docs = len(docs)
    if total_docs == 0:
        return{}
    
    avgdl = sum(len(d["text"].split()) for d in docs) / total_docs
    k1, b = 1.5, 0.75
    scores = {}
    
    for doc in docs:
        doc_tokens = doc["text"].lower().replace(".","").replace(",","").split()
        doc_len = len(doc_tokens)
        tf_counts =Counter(doc_tokens)
        
        score = 0.0
        for token in query_tokens:
            if token in tf_counts:
                tf = tf_counts[token]
                idf = math.log((total_docs + 1) / (1 + sum(1 for d in docs if token in d["text"].lower()))) + 1
                numerator = tf * (k1 + 1)
                denominator = tf * k1 * (1 - b + b * (doc_len / avgdl))
                score += idf * (numerator/denominator)
        scores[doc["id"]] = score
    return scores

def compute_dense_scores(docs):
    simulated_dense_scores = {
        1: 0.82,
        2: 0.95,
        3: 0.15,
        4: 0.89,
        5: 0.72
    }
    return {d["id"]: simulated_dense_scores.get(d["id"], 0.0) for d in docs}

def reciprocal_rank_fusion(sparse_scores, dense_scores, k =60):
    sparse_ranked = [doc_id for doc_id, _ in sorted(sparse_scores.items(), key=lambda x: x[1],reverse=True)]
    dense_ranked = [doc_id for doc_id, _ in sorted(dense_scores.items(), key=lambda x: x[1],reverse=True)]

    rrf_scores = {}
    
    for rank, doc_id in enumerate(sparse_ranked, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id,0.0) + (1.0/ (k +rank))
        
    for rank, doc_id in enumerate(dense_ranked, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id,0.0) + (1.0/ (k +rank))
            
    return rrf_scores

if __name__ == "__main__":
    query = "database search"
    metadata_filter = {"category": "database"}
    
    print("==================================================")
    print(f"QUERY: '{query}' | FILTER: {metadata_filter}")
    print("==================================================\n")
    
    all_dense = compute_dense_scores(docs)
    print("[1] Pure Unfiltered Dense Search Results:")
    for doc_id, score in sorted(all_dense.items(), key=lambda x: x[1],reverse=True):
        doc = next(d for d in docs if d["id"] == doc_id)
        print(f"    - Doc {doc_id} [{doc['category']}]: Score = {score:.4f} | \"{doc['text']}\"")
        
    filtered_docs = filter_docs(docs, metadata_filter)
    sparse_scores = compute_bm25_scores(query, filtered_docs)
    dense_scores = compute_dense_scores(filtered_docs)
    hybrid_rrf_scores = reciprocal_rank_fusion(sparse_scores, dense_scores)
    
    print("\n[2] Filtered Hybrid Search Results (Pre-filtered -> RRF Fusion):")
    for doc_id, rrf_score in sorted(hybrid_rrf_scores.items(), key=lambda x: x[1],reverse=True):
        doc = next(d for d in docs if d["id"] == doc_id)
        print(f"    - Doc {doc_id} [{doc['category']}]: RRF Score = {rrf_score:.5f} | \"{doc['text']}\"")