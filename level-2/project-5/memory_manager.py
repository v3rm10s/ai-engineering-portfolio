from pydantic import BaseModel, Field
from collections import deque
from typing import List, Dict, Optional
import json
import uuid
from datetime import datetime,timezone

class ConversationBuffer:
    def __init__(self, max_turns: int =6):
        self.max_turns = max_turns
        self.buffer = deque(maxlen=max_turns)
        
    def add_messages(self, role: str, content: str) -> None:
        self.buffer.append({"role": role, "content":content})
        
    def get_history(self) -> List[Dict[str,str]]:
        return list(self.buffer)
    
class ExtractedFact(BaseModel):
    fact: str = Field(description="An atomic, self-contained statement of fact or preference about the user.")
    category: str = Field(description="Category of fact (e.g., 'preference', 'identity', 'work', 'technical').")
    confidence: float = Field(default=1.0, description="Confidence score between 0.0 and 1.0.")
    
class FactExtractionResult(BaseModel):
    facts: List[ExtractedFact] = Field(default_factory=list, description="List of new facts extracted from the interaction.")

class FactExtractor:
    def __init__(self, llm_client = None):
        self.llm_client=llm_client
    
    def extract_facts(self, user_message: str) -> List[ExtractedFact]:
        extracted = []
        msg_lower = user_message.lower()

        if "alex" in msg_lower:
            extracted.append(ExtractedFact(fact="User's name is Alex", category="identity", confidence=0.95))
        if "backend engineer" in msg_lower or "developer" in msg_lower:
            extracted.append(ExtractedFact(fact="User works as a backend engineer", category="work", confidence=0.9))
        if "pytorch" in msg_lower:
            extracted.append(ExtractedFact(fact="User prefers PyTorch over TensorFlow", category="preference", confidence=0.9))
        if "chicago" in msg_lower:
            extracted.append(ExtractedFact(fact="User is based in Chicago", category="location", confidence=0.85))

        return extracted

class EpistemicMemoryStore:
    def __init__(self):
        self.records: List[Dict] = []
        
    def upsert_fact(self, extracted_fact: ExtractedFact, source: str = "user_chat") -> Dict:
        for record in self.records:
            if record["fact"].lower() == extracted_fact.fact.lower():
                record["metadate"]["updated_at"] = datetime.now(timezone.utc).isoformat()
                return record
    
        record = {
            "id": str(uuid.uuid4())[:8],
            "fact": extracted_fact.fact,
            "metadata": {
                "category": extracted_fact.category,
                "confidence": extracted_fact.confidence,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "source": source
            }
        }
        self.records.append(record)
        return record
    
    def search_facts(self, query:str, top_k: int=3) -> List[Dict]:
        query_words = set(query.lower().split())
        scored_records = []
        
        for record in self.records:
            fact_words = set(record["fact"].lower().split())
            category_words = set(record["metadata"]["category"].lower().split())
            
            score = len(query_words.intersection(fact_words | category_words))
            scored_records.append((score, record))
        
        scored_records.sort(key=lambda x: x[0], reverse=True)
        
        return [item[1] for item in scored_records[:top_k]]

class MemoryManager:
    def __init__(self, max_turns: int =4):
        self.short_term = ConversationBuffer(max_turns=max_turns)
        self.extractor = FactExtractor()
        self.long_term = EpistemicMemoryStore()
        
    def process_turn(self, user_message: str) -> Dict:
        extracted = self.extractor.extract_facts(user_message)
        for fact in extracted:
            self.long_term.upsert_fact(fact)
            
        self.short_term.add_messages(role="user", content=user_message)
        relevant_facts = self.long_term.search_facts(user_message, top_k=3)
        system_prompt = self._build_system_prompt(relevant_facts)
        
        return {
            "system_prompt": system_prompt,
            "short_term_history": self.short_term.get_history(),
            "newly_extracted_facts_count": len(extracted)
        }
        
    def add_assistant_response(self, assistant_message: str) -> None:
        self.short_term.add_messages(role="assistant",content=assistant_message)    

    def _build_system_prompt(self, retrieved_facts: List[Dict]) -> str:
        prompt_lines = [
            "You are an intelligent, personalized AI assistant.",
            "",
            "=== RELEVANT LONG-TERM (EPISTEMIC) MEMORY ==="
        ]
        
        if not retrieved_facts:
            prompt_lines.append("No relevant long-term facts found for this turn.")
        else:
            for item in retrieved_facts:
                cat = item['metadata']['category'].upper()
                prompt_lines.append(f"- [{cat}] {item['fact']}")
                
        prompt_lines.append("==================================")
        return "\n".join(prompt_lines)


if __name__ == "__main__":
    memory = MemoryManager(max_turns=4)

    print("--- TURN 1: User Introduces Self ---")
    turn1 = memory.process_turn("Hi! I'm Alex, a backend engineer in Chicago. I prefer PyTorch.")
    memory.add_assistant_response("Nice to meet you Alex!")

    print("\n--- TURNS 2 to 5: Filler Chat (Flushing Short-Term Buffer) ---")
    for i in range(4):
        memory.process_turn(f"Filler user question {i+1}")
        memory.add_assistant_response(f"Filler response {i+1}")

    print("\n--- TURN 6: Recalling Info After Short-Term Buffer Flush ---")
    turn6 = memory.process_turn("What model framework do I prefer for machine learning?")

    print("\n[Current Short-Term Buffer Content]:")
    for msg in turn6["short_term_history"]:
        print(f"  {msg['role']}: {msg['content']}")

    print("\n[Synthesized System Prompt Injected to LLM]:")
    print(turn6["system_prompt"])