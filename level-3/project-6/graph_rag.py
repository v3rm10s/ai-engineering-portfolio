import networkx as nx
import numpy as np

class KnowledgeGraphStore:
    def __init__(self):
        self.graph = nx.DiGraph()

    def add_relationship(self, source: str, relation: str, target: str):
        self.graph.add_edge(source.lower(), target.lower(), relation=relation)

    def get_neighborhood(self, entity: str, hops: int = 2) -> list[str]:
        entity = entity.lower()
        if entity not in self.graph:
            return []
        
        nodes = {entity}
        current_layer = {entity}
        for _ in range(hops):
            next_layer = set()
            for node in current_layer:
                neighbors = set(self.graph.successors(node)).union(set(self.graph.predecessors(node)))
                next_layer.update(neighbors)
            nodes.update(next_layer)
            current_layer = next_layer

        subgraph = self.graph.subgraph(nodes)
        facts = []
        for u, v, data in subgraph.edges(data=True):
            facts.append(f"({u.title()}) -[{data['relation']}]-> ({v.title()})")
        return facts

    def extract_entities(self, text: str) -> list[str]:
        text_lower = text.lower()
        found_entities = []
        for node in self.graph.nodes():
            if node in text_lower:
                found_entities.append(node)
        return found_entities


class VectorStore:
    def __init__(self):
        self.documents = []
        self.embeddings = []

    def add_document(self, doc_id: str, text: str, embedding: np.ndarray):
        self.documents.append({"id": doc_id, "text": text})
        self.embeddings.append(embedding)

    def search(self, query_embedding: np.ndarray, top_k: int = 2) -> list[dict]:
        if not self.embeddings:
            return []
        
        matrix = np.array(self.embeddings)
        norm_matrix = matrix / np.linalg.norm(matrix, axis=1, keepdims=True)
        norm_query = query_embedding / np.linalg.norm(query_embedding)
        
        scores = np.dot(norm_matrix, norm_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        
        return [
            {"id": self.documents[idx]["id"], "text": self.documents[idx]["text"], "score": float(scores[idx])}
            for idx in top_indices
        ]


class GraphRAGEngine:
    def __init__(self, kg_store: KnowledgeGraphStore, vector_store: VectorStore):
        self.kg = kg_store
        self.vector_store = vector_store

    def build_payloads(self, query_text: str, query_embedding: np.ndarray, top_k: int = 2, hops: int = 2):
        # 1. Retrieve Unstructured Text Chunks
        chunks = self.vector_store.search(query_embedding, top_k=top_k)
        
        # 2. Extract Entities & Fetch Graph Facts
        entities = self.kg.extract_entities(query_text)
        graph_facts = []
        for entity in entities:
            facts = self.kg.get_neighborhood(entity, hops=hops)
            for f in facts:
                if f not in graph_facts:
                    graph_facts.append(f)

        # 3. Assemble Pure Vector Prompt Payload
        vector_text_block = "\n".join([f"- {c['text']}" for c in chunks])
        pure_vector_prompt = (
            f"Query: {query_text}\n\n"
            f"=== RETRIEVED CONTEXT (TEXT ONLY) ===\n"
            f"{vector_text_block}\n\n"
            f"Answer the query based ONLY on the context above."
        )

        # 4. Assemble GraphRAG Payload (Structured Graph + Unstructured Text)
        graph_facts_block = "\n".join([f"- {f}" for f in graph_facts]) if graph_facts else "- No direct graph facts found."
        graph_rag_prompt = (
            f"Query: {query_text}\n\n"
            f"=== STRUCTURED KNOWLEDGE GRAPH FACTS ===\n"
            f"{graph_facts_block}\n\n"
            f"=== UNSTRUCTURED TEXT PASSAGES ===\n"
            f"{vector_text_block}\n\n"
            f"Answer the query using both structured facts and text passages."
        )

        return pure_vector_prompt, graph_rag_prompt


# --- Step 3 Run Execution ---
if __name__ == "__main__":
    kg = KnowledgeGraphStore()
    kg.add_relationship("Alice", "MANAGES", "Project Apollo")
    kg.add_relationship("Project Apollo", "USES", "GraphDB")
    kg.add_relationship("Bob", "CONTRIBUTES_TO", "Project Apollo")

    vector_store = VectorStore()
    vector_store.add_document("doc_1", "Alice is a lead engineering manager at TechCorp.", np.array([1.0, 0.1, 0.0, 0.2]))
    vector_store.add_document("doc_2", "Project Apollo uses GraphDB as its primary data layer.", np.array([0.1, 0.9, 0.8, 0.1]))
    vector_store.add_document("doc_3", "Bob is a backend developer working on infrastructure.", np.array([0.0, 0.2, 0.1, 0.9]))

    engine = GraphRAGEngine(kg, vector_store)

    # Question requiring multi-hop connection across entities:
    query = "Who is involved with the database system managed under Alice's project scope?"
    query_emb = np.array([0.8, 0.6, 0.5, 0.1])

    pure_prompt, graph_rag_prompt = engine.build_payloads(query, query_emb)

    print("==========================================")
    print("1. PURE VECTOR RAG PAYLOAD:")
    print("==========================================")
    print(pure_prompt)
    print("\n==========================================")
    print("2. HYBRID GRAPHRAG PAYLOAD:")
    print("==========================================")
    print(graph_rag_prompt)