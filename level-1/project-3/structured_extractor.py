import os
from dotenv import load_dotenv
from typing import List, Optional
from pydantic import BaseModel, Field
from google import genai
from google.genai import types

class EventInfo(BaseModel):
    title: str = Field(description="A concise summary or name of the event")
    date: str = Field(description="Date of the event in YYYY=MM-DD if identifiable")
    time: Optional[str] = Field(default=None, description="Start time (e.g., '14:00) if mentioned")
    participants: List[str] = Field(default=[], description="Names of individuals who attended the event")
    location: Optional[str] = Field(default=None, description="Physical venue or virtual link if available")
    is_urgent: bool = Field(description="Set to true if the text implies immediate or high priority")
    
UNSTRUCTURED_EMAIL = """
Hey team, quick heads up! We need urgent planning for the Q3 Strategy Sync. 
Sarah and Marcus will be joining us. Let's meet at Room 4B on 2026-08-12 at 14:30. 
Please bring your project updates!
"""

def extract_event_info(text:str) -> EventInfo:
    load_dotenv()
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("Error: Gemini API Key not found")
    
    client = genai.Client(api_key=api_key)
    
    response = client.models.generate_content(
        model = "gemini-2.5-flash",
        contents= f"Extract event information from the following text:\n\n{text}",
        config = types.GenerateContentConfig(
            response_mime_type = "application/json",
            response_schema=EventInfo,
        ),
    )
    
    event = EventInfo.model_validate_json(response.text)
    return event

if __name__ == "__main__":
    print("Extracint event details...")
    extracted_event = extract_event_info(UNSTRUCTURED_EMAIL)
    
    print("\n--- Extracted Strongly-Typed Pydantic Object ---")
    print(f"Title: {extracted_event.title}")
    print(f"Date: {extracted_event.date}")
    print(f"Time: {extracted_event.time}")
    print(f"Participants: {extracted_event.participants}")
    print(f"Location: {extracted_event.location}")
    print(f"Is Urgent: {extracted_event.is_urgent}")
    print(f"\nPython Data Type: {type(extracted_event)}")