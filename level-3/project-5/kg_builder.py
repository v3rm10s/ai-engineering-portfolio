import os
from dotenv import load_dotenv
from google import genai
from google.genai import types
from pydantic import BaseModel, Field
from typing import List
import networkx as nx

load_dotenv()
api_key = os.getenv("GEMINI-API-KEY") or os.getenv("GEMINI_API_KEY")

class Triplet(BaseModel):
    subject: str = Field(description="The source entity (e.g., 'Apple Inc.', 'Steve Jobs')")
    predicate: str = Field(description="The relationship in UPPER_SNAKE_CASE (e.g., 'FOUNDED_BY', 'LOCATED_IN')")
    object: str = Field(description="The target entity (e.g., 'California', 'iPhone')")
    
class KnowledgeGraphSchema(BaseModel):
    triplets: List[Triplet] = Field(description="List of extracted subject-predicate-object triplets.")

def extract_triplets(text: str) -> KnowledgeGraphSchema:
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=text,
        config=types.GenerateContentConfig(
            system_instruction="You are a Knowledge Graph extraction engine. Extract all entities and relations as clean (subject, predicate, object) triplets from the input text.",
            response_mime_type="application/json",
            response_schema=KnowledgeGraphSchema,
        )
    )
    
    # The SDK automatically parses the JSON directly into your Pydantic model!
    return response.parsed

def build_graph(data: KnowledgeGraphSchema) -> nx.DiGraph:
    G = nx.DiGraph()

    for triplet in data.triplets:
        G.add_edge(
            triplet.subject,
            triplet.object,
            relation = triplet.predicate
        )
    return G

def find_multi_hop_path(G: nx.DiGraph, start_node:str, end_node:str):
    print(f"🔍 Searching paths from '{start_node}' ──► '{end_node}'...\n")
    
    if not G.has_node(start_node) or not G.has_node(end_node):
        print("One or both nodes do not exist in the graph.")
        return
    
    try:
        paths = list(nx.all_simple_paths(G, source=start_node, target=end_node))
        if not paths:
            print("No path found.")
            return

        for idx, path in enumerate(paths, 1):
            print(f"Path {idx} ({len(path) - 1} hops):")
            for i in range(len(path) - 1):
                u = path[i]
                v = path[i + 1]
                rel = G[u][v]['relation']
                print(f"  Hop {i+1}: ({u}) ───[{rel}]───► ({v})")
            print()
    except nx.NetworkXNoPath:
        print("No path exists between these entities.")
    
if __name__ == "__main__":
    sample_text = """
    Satya Nadella is the CEO of Microsoft. Microsoft acquired GitHub in 2018. 
    GitHub is headquartered in San Francisco. Microsoft also invested heavily in OpenAI, 
    which developed ChatGPT.
    """
    
    print("1. Extracting triplets...")
    graph_data = extract_triplets(sample_text)
    
    print("2. Building graph...")
    G = build_graph(graph_data)
    
    print("3. Executing Multi-Hop Traversal:\n")
    # Query 1: 3-Hop Traversal (Satya Nadella -> San Francisco)
    find_multi_hop_path(G, "Satya Nadella", "San Francisco")
    
    # Query 2: 3-Hop Traversal (Satya Nadella -> ChatGPT)
    find_multi_hop_path(G, "Satya Nadella", "ChatGPT")