import math
import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client()

def get_embedding(text: str) -> list[float]:
    result = client.models.embed_content(
        model="gemini-embedding-2",
        contents=text
    )
    return result.embeddings[0].values

def cosine_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    dot_product = sum(a * b for a,b in zip(vec_a, vec_b))
    norm_a = math.sqrt(sum(a ** 2 for a in vec_a))
    norm_b = math.sqrt(sum(b ** 2 for b in vec_b))
    
    if norm_a == 0 or norm_b == 0:
        return 0.0
    
    return dot_product / (norm_a * norm_b)

if __name__ == "__main__":
    base_sentence = "The cat sits on the mat."
    similar_sentence = "A feline is resting on the rug"
    unrelated_sentence = "Quantum computing uses qubits to process data."
    
    print("Fetching Coordinates...")
    vec_base = get_embedding(base_sentence)
    vec_similar = get_embedding(similar_sentence)
    vec_unrelated = get_embedding(unrelated_sentence)
    
    score_similar = cosine_similarity(vec_base, vec_similar)
    score_unrelated = cosine_similarity(vec_base, vec_unrelated)
    
    print("\n--- Results ---")
    print(f"Base: '{base_sentence}")
    print(f"Vs. Similar: '{similar_sentence}' -> Score: {score_similar:.4f}")
    print(f"Vs. Unrelated: '{unrelated_sentence}' -> Score: {score_unrelated:.4f}")
    