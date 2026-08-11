import os
from dotenv import load_dotenv
from google import genai
from query_expander import QueryExpander

# Call function with ()
load_dotenv()

api_key = os.getenv("GEMINI_API_KEY")

# Initialize client
client = genai.Client(api_key=api_key)

expander = QueryExpander(client=client, num_variations=3)

test_query = "How do I fix a database connection timeout in Python?"
print(f"Original Query: '{test_query}'\n")

variations = expander.generate_variations(test_query)

print("Generated Variations:")
for i, var in enumerate(variations):
    print(f"{i+1}. {var}")