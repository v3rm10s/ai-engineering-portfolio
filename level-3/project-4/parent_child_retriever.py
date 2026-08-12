import uuid
from dataclasses import dataclass, field
from typing import List, Dict, Set,Optional
import numpy as np

# Note: Using sentence-transformers for vector embeddings
from sentence_transformers import SentenceTransformer

@dataclass
class ParentNode:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)

@dataclass
class ChildNode:
    id: str
    parent_id: str
    text: str
    metadata: dict = field(default_factory=dict)

class ParentChildStore:
    def __init__(self, parent_chunk_size: int = 200, child_chunk_size: int = 60):
        self.parent_chunk_size = parent_chunk_size
        self.child_chunk_size = child_chunk_size
        self.parent_store: Dict[str, ParentNode] = {}
        self.child_nodes: List[ChildNode] = []
        self.child_embeddings: Optional[np.ndarray] = None
        self.model = SentenceTransformer("all-MiniLM-L6-v2")

    def _chunk_text(self, text: str, max_chars: int) -> List[str]:
        words = text.split()
        chunks, current_chunk, current_length = [], [], 0
        for word in words:
            current_chunk.append(word)
            current_length += len(word) + 1
            if current_length >= max_chars:
                chunks.append(" ".join(current_chunk))
                current_chunk, current_length = [], 0
        if current_chunk:
            chunks.append(" ".join(current_chunk))
        return chunks

    def add_document(self, doc_text: str):
        parent_texts = self._chunk_text(doc_text, self.parent_chunk_size)
        
        for p_text in parent_texts:
            parent_id = str(uuid.uuid4())
            parent_node = ParentNode(id=parent_id, text=p_text)
            self.parent_store[parent_id] = parent_node
            
            child_texts = self._chunk_text(p_text, self.child_chunk_size)
            for c_text in child_texts:
                child_id = str(uuid.uuid4())
                child_node = ChildNode(id=child_id, parent_id=parent_id, text=c_text)
                self.child_nodes.append(child_node)

    def build_index(self):
        """Embeds all child nodes into a matrix for vector search."""
        texts = [c.text for c in self.child_nodes]
        print(f"Embedding {len(texts)} child chunks...")
        embeddings = self.model.encode(texts, convert_to_numpy=True)
        # Normalize vectors for cosine similarity
        self.child_embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        print("Vector index successfully built!")

    def retrieve(self, query: str, top_k_children: int = 3) -> List[ParentNode]:
        """
        Executes Small-to-Big Retrieval:
        1. Embeds query & calculates similarity with Child embeddings
        2. Identifies top-k matching Child nodes
        3. Maps back to parent_ids and DEDUPLICATES parent nodes
        """
        if self.child_embeddings is None:
            raise ValueError("Index not built! Call build_index() first.")

        # Embed query
        q_emb = self.model.encode([query], convert_to_numpy=True)
        q_emb = q_emb / np.linalg.norm(q_emb, axis=1, keepdims=True)

        # Cosine similarity scores across all child chunks
        scores = np.dot(self.child_embeddings, q_emb.T).flatten()
        top_indices = np.argsort(scores)[::-1][:top_k_children]

        retrieved_parents: List[ParentNode] = []
        seen_parent_ids: Set[str] = set()

        print("\n--- MATCHED CHILD CHUNKS (Vector Search) ---")
        for rank, idx in enumerate(top_indices, 1):
            matched_child = self.child_nodes[idx]
            score = scores[idx]
            print(f"Rank {rank} (Score: {score:.3f}): \"{matched_child.text}\" (Parent ID: {matched_child.parent_id[:8]}...)")

            # Deduplicate parent lookups
            if matched_child.parent_id not in seen_parent_ids:
                seen_parent_ids.add(matched_child.parent_id)
                retrieved_parents.append(self.parent_store[matched_child.parent_id])

        return retrieved_parents


# Execution & Testing
if __name__ == "__main__":
    sample_doc = (
        "The Apollo 11 mission launched on July 16, 1969, carrying Neil Armstrong, Buzz Aldrin, and Michael Collins. "
        "Neil Armstrong became the first human to walk on the Moon on July 20. "
        "The Saturn V rocket was used to launch the spacecraft into orbit. "
        "The Lunar Module Eagle landed in the Sea of Tranquility, while Collins orbited above in the Command Module Columbia."
    )

    store = ParentChildStore(parent_chunk_size=200, child_chunk_size=60)
    store.add_document(sample_doc)
    store.build_index()

    # Query targeting a specific small detail
    query = "Who walked on the moon?"
    retrieved_parents = store.retrieve(query, top_k_children=3)

    print("\n--- FINAL RETRIEVED PARENT CONTEXT FOR LLM SYNTHESIS ---")
    for i, parent in enumerate(retrieved_parents, 1):
        print(f"\nParent Context {i} [{parent.id[:8]}...]:\n\"{parent.text}\"")