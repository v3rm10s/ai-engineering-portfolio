from typing import List, Dict, Any
from query_expander import QueryExpander

class MultiQueryRetriever:
    def __init__(self, vector_store, query_expander: QueryExpander):
        """
        :param vector_store: Any vector store object/client with a search method.
        :param query_expander: An instance of QueryExpander.
        """
        self.vector_store = vector_store
        self.expander = query_expander

    def retrieve(self, raw_query: str, top_k_per_query: int = 3) -> List[Dict[str, Any]]:
        """
        Expands the raw query, retrieves candidate chunks for each query,
        and deduplicates the results by document/chunk ID.
        """
        # 1. Expand query into multiple variations
        queries = self.expander.generate_variations(raw_query)
        print(f"\n[MultiQueryRetriever] Executing search across {len(queries)} queries...")

        unique_chunks: Dict[str, Dict[str, Any]] = {}

        # 2. Iterate through each query variation
        for q in queries:
            # Assume vector_store.search(query, top_k) returns a list of dicts:
            # [{'id': 'chunk_1', 'text': '...', 'score': 0.85}, ...]
            results = self.vector_store.search(q, top_k=top_k_per_query)

            for item in results:
                doc_id = item['id']
                score = item.get('score', 0.0)

                if doc_id not in unique_chunks:
                    # New chunk discovered
                    unique_chunks[doc_id] = {
                        "id": doc_id,
                        "text": item['text'],
                        "max_score": score,
                        "hit_count": 1,
                        "retrieved_by": [q]
                    }
                else:
                    # Chunk already exists -> update metrics
                    unique_chunks[doc_id]["hit_count"] += 1
                    unique_chunks[doc_id]["retrieved_by"].append(q)
                    if score > unique_chunks[doc_id]["max_score"]:
                        unique_chunks[doc_id]["max_score"] = score

        # 3. Sort final deduplicated chunks by highest max_score
        deduplicated_results = list(unique_chunks.values())
        deduplicated_results.sort(key=lambda x: x["max_score"], reverse=True)

        return deduplicated_results