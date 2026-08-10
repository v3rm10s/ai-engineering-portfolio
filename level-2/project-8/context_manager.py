import os
from dotenv import load_dotenv
from google import genai
from google.genai import types

class ContextManager:
    def __init__(self, client: genai.Client, model_name: str= "gemini-2.5-flash", max_tokens:int = 4000, safety_buffer: int = 1000):
        self.client = client
        self.model_name = model_name
        self.max_tokens = max_tokens
        self.safety_buffer = safety_buffer
        self.effective_limit = max_tokens - safety_buffer
        
        self.summary:str = ""
        self.history: list[dict] = []
        
    def count_tokens(self, contents: list | str) -> int:
        response = self.client.models.count_tokens(
            model=self.model_name,
            contents=contents
        )
        return response.total_tokens
    
    def add_message(self, role: str, text:str):
        self.history.append({"role": role, "text": text})
        
    def build_payload(self) -> list[dict]:
        """
        Assembles the full context payload:
        1. Inject summary as pre-conditioned context (if present)
        2. Append uncompressed chat history turns
        """
        payload = []
        
        # If we have a rolling summary, prepend it as initial context
        if self.summary:
            payload.append({
                "role": "user",
                "parts": [{"text": f"[PREVIOUS CONVERSATION SUMMARY]:\n{self.summary}"}]
            })
            payload.append({
                "role": "model",
                "parts": [{"text": "Understood. I have locked this previous context into memory."}]
            })
            
        # Append active history
        for msg in self.history:
            payload.append({
                "role": msg["role"],
                "parts": [{"text": msg["text"]}]
            })
            
        return payload

    def get_total_tokens(self) -> int:
        """Calculates total tokens across the full payload."""
        payload = self.build_payload()
        if not payload:
            return 0
        return self.count_tokens(payload)

    def is_over_budget(self) -> bool:
        """Checks if current context payload breaches the effective token limit."""
        return self.get_total_tokens() > self.effective_limit
    
    def summarize_and_trim(self, num_to_trim: int = 2):
        """
        Takes the oldest `num_to_trim` messages, summarizes them with Gemini,
        updates `self.summary`, and removes them from active history.
        """
        if len(self.history) < num_to_trim:
            return  # Not enough history to trim
            
        old_messages = self.history[:num_to_trim]
        
        # Build prompt for summarization
        prompt = f"Existing Summary:\n{self.summary if self.summary else 'None'}\n\n"
        prompt += "Old Conversation Segment to Integrate:\n"
        for msg in old_messages:
            prompt += f"{msg['role'].upper()}: {msg['text']}\n"
            
        prompt += "\nTask: Synthesize the existing summary and the old conversation segment into a concise, unified summary paragraph preserving critical facts, names, and context."
        
        # Call Gemini to create the condensed summary
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt
        )
        
        # Update rolling summary state
        self.summary = response.text.strip()
        
        # Drop trimmed messages from local history list
        self.history = self.history[num_to_trim:]
        print(f"\n[SYSTEM]: Context Compressed! Trimmed {num_to_trim} messages. New Summary Length: {len(self.summary)} chars.\n")

    def auto_manage(self):
        """Trims context incrementally until we are safely back under budget."""
        while self.is_over_budget() and len(self.history) >= 2:
            self.summarize_and_trim(num_to_trim=2)
        
if __name__ == "__main__":
    load_dotenv()
    gen_ai = genai.Client()
    
    # Low budget to trigger compression easily
    cm = ContextManager(gen_ai, "gemini-2.5-flash", max_tokens=70, safety_buffer=20)
    
    conversation = [
        ("user", "Hi, my name is Alice and I am a software engineer building an AI agent."),
        ("model", "Nice to meet you, Alice! What kind of AI agent are you building?"),
        ("user", "It is a dynamic context management system that uses Gemini to summarize old chat logs."),
        ("model", "That sounds awesome! Token management is super important for scaling conversational AI."),
        ("user", "Exactly. Now can you remind me what my job title and project were?")
    ]
    
    for role, text in conversation:
        cm.add_message(role, text)
        print(f"Added [{role}] message. Tokens: {cm.get_total_tokens()} | Over Budget? {cm.is_over_budget()}")
        
        if cm.is_over_budget():
            print(">>> Budget Breached! Executing auto_manage()...")
            cm.auto_manage()
            print(f"Post-compression Tokens: {cm.get_total_tokens()}")

    print("\n" + "="*40)
    print("FINAL MANAGER STATE:")
    print(f"Rolling Summary: {cm.summary}")
    print(f"Active History Items: {len(cm.history)}")
    print("="*40)