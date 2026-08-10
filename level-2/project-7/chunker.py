import math
import re

def fixed_size_chunk(text:str, chunk_size:int =200,overlap:int = 40) -> list[str]:
    if overlap >= chunk_size:
            raise ValueError("Overlap must be strictly smaller than chunk_size.")
    
    step = chunk_size - overlap
    return [text[i:i+chunk_size] for i in range(0, len(text), step)]


def recursive_chunk(
    text:str,
    max_chunk_size:int =200,
    separators: list[str] = None
) -> list[str]:
    if separators is None:
        separators = ["\n\n","\n"," ",""]
        
    if len(text) <= max_chunk_size or not separators:
        return [text] if text else []
    
    sep = separators[0]
    next_separators = separators[1:]
    
    splits = text.split(sep) if sep != "" else list(text)
    
    chunks = []
    current_chunk = ""
    
    for piece in splits:
        piece_with_sep = piece + sep if sep != "" else piece
        
        if len(piece_with_sep) > max_chunk_size:
            if current_chunk:
                chunks.append(current_chunk.strip())
                current_chunk =""
            chunks.extend(recursive_chunk(piece, max_chunk_size,next_separators))
            continue
        
        if len(current_chunk) + len(piece_with_sep) <= max_chunk_size:
            current_chunk += piece_with_sep
        else:
            chunks.append(current_chunk.strip())
            current_chunk = piece_with_sep
    
    if current_chunk:
        chunks.append(current_chunk.strip())
        
    return [c for c in chunks if c]
    


def split_into_sentences(text: str) -> list[str]:
    """Splits text into sentences based on punctuation."""
    sentences = re.split(r'(?<=[.!?]) +', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def simple_embedding(sentence: str, vocabulary: list[str]) -> list[float]:
    """
    Creates a simple term-frequency vector representing the sentence's meaning.
    """
    words = re.findall(r'\w+', sentence.lower())
    return [words.count(word) for word in vocabulary]

def cosine_distance(vec1: list[float], vec2: list[float]) -> float:
    """Calculates cosine distance (1 - cosine similarity) between two vectors."""
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = math.sqrt(sum(a * a for a in vec1))
    norm2 = math.sqrt(sum(b * b for b in vec2))
    
    if norm1 == 0 or norm2 == 0:
        return 1.0  # Maximum distance if a vector is empty
        
    similarity = dot_product / (norm1 * norm2)
    return 1.0 - similarity

def semantic_chunk(text: str, distance_threshold: float = 0.5) -> list[str]:
    """
    Chunks text dynamically when the semantic distance between adjacent sentences
    exceeds the distance_threshold.
    """
    sentences = split_into_sentences(text)
    if len(sentences) <= 1:
        return sentences

    # Build vocabulary for simple vectorization across all sentences
    all_words = sorted(list(set(re.findall(r'\w+', text.lower()))))
    
    # Generate vectors for every sentence
    vectors = [simple_embedding(s, all_words) for s in sentences]

    chunks = []
    current_chunk = [sentences[0]]

    for i in range(len(sentences) - 1):
        dist = cosine_distance(vectors[i], vectors[i+1])
        
        # If distance spikes above threshold, break into a new chunk
        if dist > distance_threshold:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sentences[i+1]]
        else:
            current_chunk.append(sentences[i+1])

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks

sample_text = (
    "Retrieval-Augmented Generation (RAG) enhances Large Language Models by fetching relevant "
    "documents from an external knowledge base. Proper chunking is critical because if chunks are too "
    "large, they dilute the signal with irrelevant noise. Conversely, if chunks are too small, they "
    "lack the necessary semantic context needed to answer complex queries accurately. Fixed-size "
    "chunking uses a fixed character length with an overlapping window to preserve boundary context."
)

multi_topic_text = (
    "Retrieval-Augmented Generation enhances Large Language Models by fetching relevant external knowledge. "
    "Proper chunking ensures high vector similarity during dense passage retrieval. "
    "Baking artisan sourdough bread requires flour, water, salt, and active wild yeast. "
    "The fermentation process usually takes 12 to 24 hours depending on room temperature."
)


if __name__ == "__main__":
    print("=== FIXED SIZE CHUNKING ===")
    fixed = fixed_size_chunk(sample_text, chunk_size=150, overlap=30)
    for i, c in enumerate(fixed):
        print(f"Chunk {i+1}: {repr(c)}\n")

    print("\n=== RECURSIVE CHUNKING ===")
    rec = recursive_chunk(sample_text, max_chunk_size=150)
    for i, c in enumerate(rec):
        print(f"Chunk {i+1}: {repr(c)}\n")
        
    print("=== SEMANTIC CHUNKING ===")
    sem_chunks = semantic_chunk(multi_topic_text, distance_threshold=0.6)
    for i, c in enumerate(sem_chunks):
        print(f"Chunk {i+1}:\n  {repr(c)}\n")