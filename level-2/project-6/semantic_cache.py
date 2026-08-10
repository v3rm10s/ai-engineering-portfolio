import time
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple

@dataclass
class CacheEntry:
    query: str
    embedding: np.ndarray
    response: str
    
class SemanticCache:
    def __init__(self, threshold: float =0.90):
        self.threshold = threshold
        self.cache: list[CacheEntry] = []
        
    def _cosine_similarity(self, vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return float(dot_product / (norm_a * norm_b))
    
    def add(self, query:str, embedding:np.ndarray, response:str) -> None:
        entry = CacheEntry(query=query, embedding=embedding, response=response)
        self.cache.append(entry)
        
    def lookup(self, query_embedding: np.ndarray) -> Optional[Tuple[str,float]]:
        if not self.cache:
            return None
    
        best_score = -1.0
        best_entry = None
        
        for entry in self.cache:
            score = self._cosine_similarity(query_embedding, entry.embedding)
            if score > best_score:
                best_score = score
                best_entry = entry
                
        if best_entry and best_score >= self.threshold:
            return best_entry.response, best_score
        
        return None
    
def mock_get_embedding(text:str) -> np.ndarray:
    text_lower = text.lower()
    if "capital" in text_lower and "france" in text_lower:
        vec = np.array([0.9,0.1,0.01])
    elif "python" in text_lower:
        vec = np.array([0.01, 0.95,0.05])
    else:
        vec = np.random.rand(3)
    return vec / np.linalg.norm(vec)
    
def mock_llm_api_call(query: str) -> str:
    time.sleep(1.5)
    if "france" in query.lower():
        return "The capital of France is Paris."
    return f"This is a generated response for: '{query}'"
    
def query_llm_with_cache(query: str, cache: SemanticCache) -> str:
    start_time = time.time()
    query_vec = mock_get_embedding(query)
    cached_result = cache.lookup(query_vec)
        
    if cached_result:
        response, score = cached_result
        elapsed_ms = (time.time() - start_time) * 1000
        print(f"\n[CACHE HIT | Score: {score:.4f}] Latency: {elapsed_ms:.2f}ms")
        print(f"Response: {response}")
        return response

    # 3. Cache Miss - Call LLM
    print(f"\n[CACHE MISS] Fetching from LLM...")
    response = mock_llm_api_call(query)
    elapsed_ms = (time.time() - start_time) * 1000
    
    # 4. Asynchronously save to cache
    cache.add(query, query_vec, response)
    
    print(f"Latency: {elapsed_ms:.2f}ms")
    print(f"Response: {response}")
    return response
        
    
if __name__ == "__main__":
    
    cache = SemanticCache(threshold=0.90)

    print("--- Test 1: First query (Cache Miss) ---")
    query_llm_with_cache("What is the capital of France?", cache)

    print("\n--- Test 2: Semantically similar query (Cache Hit expected) ---")
    query_llm_with_cache("Tell me the capital city of France", cache)

    print("\n--- Test 3: Completely new query (Cache Miss expected) ---")
    query_llm_with_cache("What is Python programming?", cache)
    