import os
from dotenv import load_dotenv
from google import genai
from query_expander import QueryExpander
from multi_query_retriever import MultiQueryRetriever

load_dotenv()

# --- Mock Vector Store for Demonstration ---
class MockVectorStore:
    def __init__(self):
        # A tiny knowledge base with distinct wording
        self.documents = [
            {"id": "doc_1", "text": "Python database connection pool exhaustion can cause timeout errors when connections aren't released.", "keywords": ["timeout", "python", "database", "connection"]},
            {"id": "doc_2", "text": "To resolve DB connectivity issues in Python applications, check socket timeout settings in psycopg2 or SQLAlchemy.", "keywords": ["db connectivity", "socket timeout", "psycopg2", "sqlalchemy"]},
            {"id": "doc_3", "text": "HTTP status 500 mitigation strategies for cloud infrastructure.", "keywords": ["500", "http", "cloud"]},
            {"id": "doc_4", "text": "Mitigate network latency and socket timeouts by tuning firewall idle connection limits.", "keywords": ["socket timeout", "network", "firewall", "mitigate"]}
        ]

    def search(self, query: str, top_k: int = 2):
        """Simulates vector search using simple keyword match scoring."""
        query_words = set(query.lower().split())
        scored_docs = []
        for doc in self.documents:
            # Simple overlap score mock
            overlap = sum(1 for word in query_words if word in doc["text"].lower())
            score = overlap / (len(query_words) + 1)
            if score > 0:
                scored_docs.append({"id": doc["id"], "text": doc["text"], "score": round(score, 3)})
        
        scored_docs.sort(key=lambda x: x["score"], reverse=True)
        return scored_docs[:top_k]

# --- Setup Test ---
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

expander = QueryExpander(client=client, num_variations=3)
vector_store = MockVectorStore()
retriever = MultiQueryRetriever(vector_store=vector_store, query_expander=expander)

test_query = "How do I fix a database connection timeout in Python?"

print("================ SINGLE QUERY RETRIEVAL ================")
single_results = vector_store.search(test_query, top_k=2)
for res in single_results:
    print(f"[{res['id']}] Score: {res['score']} | {res['text']}")

print("\n================ MULTI-QUERY RETRIEVAL ================")
multi_results = retriever.retrieve(test_query, top_k_per_query=2)
for res in multi_results:
    print(f"[{res['id']}] Hits: {res['hit_count']} | Max Score: {res['max_score']} | Retrieved by {len(res['retrieved_by'])} query variation(s)")
    print(f"   Text: {res['text']}")