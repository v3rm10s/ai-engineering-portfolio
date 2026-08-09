import numpy as np
from typing import List, Dict, Any, Optional

class InMemoryVectorStore:
    def __init__(self):
        self.documents: List[Dict[str, Any]] = []
        
    def add_documents(
        self,
        texts: List[str],
        embeddings: List[List[float]] | np.ndarray,
        metadatas: Optional[List[Dict[str, Any]]] = None
    ) -> List[str]:
        
        if len(texts) != len(embeddings):
            raise ValueError("The number of texts and embeddings must match!")

        if metadatas is None:
            metadatas = [{}] * len(texts)
            
        inserted_ids = []
        
        for text, embedding, metadata in zip(texts, embeddings, metadatas):
            doc_id = f"doc_{len(self.documents)}"
            
            doc = {
                "id": doc_id,
                "text": text,
                "embedding": embedding,
                "metadata": metadata
            }
            
            self.documents.append(doc)
            inserted_ids.append(doc_id)
        
        return inserted_ids
    
    @staticmethod
    def _cosine_similarity(v1: np.ndarray, v2: np.ndarray) -> float:
        norm1 = np.linalg.norm(v1)
        norm2 = np.linalg.norm(v2)
        if norm1 == 0 or norm2 == 0:
            return 0.0
        return float(np.dot(v1, v2) / (norm1 * norm2))
    
    def search(
        self,
        query_embedding: List[float] | np.ndarray,
        k: int = 3
    ) -> List[Dict[str, Any]]:
        results = []
        query_vec = np.array(query_embedding)
        
        for doc in self.documents:
            doc_vec = np.array(doc["embedding"])
            
            score = self._cosine_similarity(query_vec, doc_vec)
            
            res = doc.copy()
            res["score"] = score
            results.append(res)
    
        results.sort(key=lambda x: x["score"],reverse=True)
            
        return results[:k]
                
    
if __name__=="__main__":
    store = InMemoryVectorStore()
    
    sample_texts = [
        "Python data science and machine learning",
        "Baking delicious chocolate cakes",
        "Deep learning and neural network architectures"
    ]
    
    sample_vectors = np.array([
        [0.9, 0.8, 0.1, 0.0],  # Programming/Tech
        [0.0, 0.1, 0.9, 0.8],  # Cooking
        [0.8, 0.9, 0.2, 0.1]   # Deep Learning (close to Programming)
    ])
    
    store.add_documents(sample_texts, sample_vectors)
    
    query_vector = np.array([0.85, 0.85, 0.1,0.0])
    
    matches = store.search(query_vector, k=2)
    
    print("Top Search Results:")
    for match in matches:
        print(f"Score: {match['score']:.4f} | Text: {match['text']}")