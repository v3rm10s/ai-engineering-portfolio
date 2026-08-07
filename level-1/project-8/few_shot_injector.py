import os
from dotenv import load_dotenv
import google.genai as genai

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

client = genai.Client()

TRICKY_QUERY = (
    "The screen on my Apex Watch 3 flickers, and honestly your support team has been pretty unhelpful. "
    "Thinking about swapping it for the SoundPulse Earbuds."
)

EXAMPLES = [
    {
        "input": "Hey, the battery on my Apex Watch 3 died after 2 hours. I want a refund ASAP!",
        "output": '{"intent": "refund_request", "sentiment": "negative", "urgency": "high", "products_mentioned": ["Apex Watch 3"]}'
    },
    {
        "input": "Just wanted to say the custom leather band for the Apex Watch fits super nicely. Great work!",
        "output": '{"intent": "product_praise", "sentiment": "positive", "urgency": "low", "products_mentioned": ["Apex Watch custom leather band"]}'
    },
    {
        "input": "Can I use the SoundPulse Earbuds under water? Looking at taking them swimming.",
        "output": '{"intent": "product_inquiry", "sentiment": "neutral", "urgency": "medium", "products_mentioned": ["SoundPulse Earbuds"]}'
    }
]

def format_few_shot_prompt(query: str, examples: list) -> str:
    prompt = (
        "You are a customer support triage assistant. Analyze the input text and return ONLY"
        "valid JSON matching the schema and keys shown in the examples below.\n\n"
    )
    
    for i, example in enumerate(examples, 1):
        prompt += f"--- Example {i} === \n"
        prompt += f"Input: {example['input']}\n"
        prompt += f"Output: {example['output']}\n"
        
    prompt += f"--- Real Task ---\nInput: {query}\nOutput:"
    return prompt

def run_zero_shot(query: str):
    zero_shot_prompt = {
        f"Analyze this customer message and extract intent, sentiment, urgency and products_mentioned into JSON:\n\n{query}"
    }
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=zero_shot_prompt,
    )
    return response.text

def run_few_shot(query: str, examples: list):
    few_shot_prompt = format_few_shot_prompt(query, examples)
    response = client.models.generate_content(
        model="gemini-2.5-flash",
        contents=few_shot_prompt,
    )
    return response.text

if __name__ == "__main__":
    print("===Zero-shot response===")
    print(run_zero_shot(TRICKY_QUERY))
    print("\n" + "="*30 + "\n")
    print("===Few-shot response===")
    print(run_few_shot(TRICKY_QUERY,EXAMPLES))