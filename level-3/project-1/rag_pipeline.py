from dataclasses import dataclass, field
from typing import List,Dict,Any
import uuid
import time
import re
import numpy as np
from sentence_transformers import SentenceTransformer

@dataclass
class DocumentChunk:
    chunk_id: str
    doc_id: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class RAGResponse:
    query: str
    answer: str
    is_answerable: bool
    citations: List[str]
    retrieved_chunk_ids: List[str]
    latency_ms: float

@dataclass
class SearchResult:
    chunk: DocumentChunk
    score:float

@dataclass
class AugmentedPrompt:
    system_prompt: str
    user_prompt: str
    is_answerable: bool
    retrieved_ids: List[str]

class DocumentIngestor:
    def __init__(self, chunk_size:int = 250, chunk_overlap:int = 30):
        self.chunk_size =chunk_size
        self.chunk_overlap = chunk_overlap
        
    def process_document(self, doc_id: str,raw_text:str) -> List[DocumentChunk]:
        chunks = []
        stride = max(1, self.chunk_size - self.chunk_overlap)
        
        chunk_index = 0
        for start in range(0, len(raw_text), stride):
            chunk_text = raw_text[start: start + self.chunk_size]
            
            chunk_id = f"{doc_id}#c{chunk_index}"
            doc_chunk = DocumentChunk(
                chunk_id = chunk_id,
                doc_id=doc_id,
                content=chunk_text,
                metadata={"start_char": start, "end_char": start + len(chunk_text)}
            )
            chunks.append(doc_chunk)
            chunk_index += 1
            
        return chunks

class VectorRetriever:
    def __init__(self, score_threshold: float = 0.3, model_name: str = "all-MiniLM-L6-v2"):
        self.score_threshold = score_threshold
        # Load lightweight local embedding model
        self.model = SentenceTransformer(model_name)
        self.chunks: List[DocumentChunk] = []
        self.embeddings: np.ndarray = None

    def add_chunks(self, chunks: List[DocumentChunk]):
        """Embeds each chunk's content and stores them in memory."""
        if not chunks:
            return
            
        texts = [c.content for c in chunks]
        # Generate embeddings (shape: [num_chunks, embedding_dim])
        new_embeddings = self.model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
        
        # Store chunks and stack/append embeddings
        self.chunks.extend(chunks)
        if self.embeddings is None:
            self.embeddings = new_embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, new_embeddings])

    def search(self, query: str, top_k: int = 3) -> List[SearchResult]:
        """Embeds query, calculates cosine similarity, and filters by score_threshold."""
        if self.embeddings is None or len(self.chunks) == 0:
            return []

        # 1. Embed query (normalized so dot product = cosine similarity)
        query_vec = self.model.encode(query, convert_to_numpy=True, normalize_embeddings=True)

        # 2. Compute similarity scores via dot product (since vectors are unit length)
        # query_vec shape: (dim,), embeddings shape: (N, dim) -> scores shape: (N,)
        scores = np.dot(self.embeddings, query_vec)

        # 3. Pair chunks with scores
        results = [
            SearchResult(chunk=self.chunks[i], score=float(scores[i]))
            for i in range(len(scores))
        ]

        # 4. Sort by score descending
        results.sort(key=lambda x: x.score, reverse=True)

        # 5. Apply relevance threshold filtering
        filtered_results = [r for r in results if r.score >= self.score_threshold]

        # 6. Return top_k
        return filtered_results[:top_k]

class PromptAugmenter:
    def __init__(self, fallback_message: str = "I cannot answer this query based on the provided context."):
        self.fallback_message = fallback_message

    def build_prompt(self, query: str, search_results: List[SearchResult]) -> AugmentedPrompt:
        """
        Formats search_results into an augmented prompt with strict citation rules.
        Handles empty context gracefully.
        """
        if not search_results:
            return AugmentedPrompt(
                system_prompt="",
                user_prompt=self.fallback_message,
                is_answerable=False,
                retrieved_ids=[]
            )

        # 1. Format context string with chunk IDs
        context_blocks = [
            f"[{res.chunk.chunk_id}]: {res.chunk.content}" 
            for res in search_results
        ]
        formatted_context = "\n".join(context_blocks)

        # 2. Enforce strict system instructions
        system_prompt = (
            "You are a strict context-grounded assistant.\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. Answer the user's question using ONLY the provided context below.\n"
            "2. Every factual assertion or claim MUST be tagged with its corresponding source ID in brackets, e.g., [doc_id#c0].\n"
            "3. Do NOT use outside knowledge or assume unmentioned facts.\n"
            "4. If the context is insufficient, state that you cannot answer."
        )

        # 3. Build user prompt block
        user_prompt = f"Context:\n{formatted_context}\n\nQuestion: {query}\n\nAnswer:"

        retrieved_ids = [res.chunk.chunk_id for res in search_results]

        return AugmentedPrompt(
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            is_answerable=True,
            retrieved_ids=retrieved_ids
        )

class LLMGenerator:
    def __init__(self, client=None, model: str = "gpt-4o-mini"):
        self.client = client
        self.model = model

    def generate(self, prompt: AugmentedPrompt, original_query: str) -> RAGResponse:
        start_time = time.perf_counter()

        # Handle unanswerable query immediately without calling LLM
        if not prompt.is_answerable:
            elapsed_ms = (time.perf_counter() - start_time) * 1000
            return RAGResponse(
                query=original_query,
                answer=prompt.user_prompt,
                is_answerable=False,
                citations=[],
                retrieved_chunk_ids=[],
                latency_ms=round(elapsed_ms, 2)
            )

        # Execute LLM call or fallback mock for testing
        if self.client:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt.system_prompt},
                    {"role": "user", "content": prompt.user_prompt}
                ],
                temperature=0.0
            )
            raw_answer = response.choices[0].message.content
        else:
            # Safely format citations from whatever chunk IDs were retrieved
            primary_id = prompt.retrieved_ids[0] if prompt.retrieved_ids else "unknown"
            secondary_id = prompt.retrieved_ids[1] if len(prompt.retrieved_ids) > 1 else primary_id

            raw_answer = (
                f"When context relevance drops below a threshold, the system gracefully "
                f"declines to answer [{secondary_id}]. This is managed by modular RAG "
                f"pipelines that separate retrieval and context preparation [{primary_id}]."
            )

        elapsed_ms = (time.perf_counter() - start_time) * 1000

        # Extract citations using Regex matching [doc_id#c0] pattern
        citations = re.findall(r"\[([a-zA-Z0-9_\-]+#c\d+)\]", raw_answer)

        return RAGResponse(
            query=original_query,
            answer=raw_answer,
            is_answerable=True,
            citations=list(set(citations)),
            retrieved_chunk_ids=prompt.retrieved_ids,
            latency_ms=round(elapsed_ms, 2)
        )

class ModularRAGPipeline:
    def __init__(self, score_threshold: float = 0.35, llm_client=None):
        self.ingestor = DocumentIngestor()
        self.retriever = VectorRetriever(score_threshold=score_threshold)
        self.augmenter = PromptAugmenter()
        self.generator = LLMGenerator(client=llm_client)

    def ingest(self, doc_id: str, text: str):
        chunks = self.ingestor.process_document(doc_id, text)
        self.retriever.add_chunks(chunks)

    def query(self, user_query: str) -> RAGResponse:
        search_results = self.retriever.search(user_query)
        augmented_prompt = self.augmenter.build_prompt(user_query, search_results)
        response = self.generator.generate(augmented_prompt, user_query)
        return response

if __name__ == "__main__":
    pipeline = ModularRAGPipeline(score_threshold=0.35)

    doc_text = (
        "Modular RAG pipelines separate retrieval, context preparation, and generation. "
        "By enforcing strict citation rules, hallucinated claims can be caught instantly. "
        "When context relevance drops below a threshold, the system gracefully declines to answer."
    )

    pipeline.ingest("policy_v1", doc_text)

    print("\n================ END-TO-END QUERY 1 ================")
    res1 = pipeline.query("How do RAG pipelines handle low context relevance?")
    print(f"Query: {res1.query}")
    print(f"Answerable: {res1.is_answerable}")
    print(f"Answer: {res1.answer}")
    print(f"Citations: {res1.citations}")
    print(f"Latency: {res1.latency_ms} ms")

    print("\n================ END-TO-END QUERY 2 ================")
    res2 = pipeline.query("Who won the 2022 World Cup?")
    print(f"Query: {res2.query}")
    print(f"Answerable: {res2.is_answerable}")
    print(f"Answer: {res2.answer}")
    print(f"Citations: {res2.citations}")
    print(f"Latency: {res2.latency_ms} ms")