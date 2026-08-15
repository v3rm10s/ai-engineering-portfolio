import chromadb
from typing import List
from src.ingestion import DocumentElement, ElementType

class MultimodalIndexer:
    def __init__(self, collection_name: str = "multimodal_rag", persist_directory: str | None = None):
        """Initializes ChromaDB vector store for multimodal indexing."""
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client() # In-memory client for testing/runtime
            
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )

    def index_elements(self, elements: List[DocumentElement]):
        """Embeds document element summaries and stores raw paths in metadata."""
        ids = []
        documents = []
        metadatas = []

        for elem in elements:
            if not elem.summary:
                raise ValueError(f"Element {elem.id} has no summary. Run summarizer first.")
            
            ids.append(elem.id)
            documents.append(elem.summary)
            metadatas.append({
                "element_type": elem.element_type.value,
                "content": elem.content,
                "summary": elem.summary
            })

        self.collection.add(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        print(f"[✓] Indexed {len(elements)} multimodal elements into vector store.")

    def search(self, query: str, top_k: int = 2) -> List[DocumentElement]:
        """Performs vector similarity search against summaries and reconstructs elements."""
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )

        retrieved_elements: List[DocumentElement] = []
        
        if not results["ids"] or not results["ids"][0]:
            return retrieved_elements

        for i in range(len(results["ids"][0])):
            elem_id = results["ids"][0][i]
            meta = results["metadatas"][0][i]
            
            retrieved_elements.append(
                DocumentElement(
                    id=elem_id,
                    element_type=ElementType(meta["element_type"]),
                    content=meta["content"],
                    summary=meta["summary"],
                    metadata={"distance": results["distances"][0][i] if "distances" in results else None}
                )
            )

        return retrieved_elements