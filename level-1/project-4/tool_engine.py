import os
from dotenv import load_dotenv
from google import genai
from google.genai import types


def get_current_weather(location: str, unit: str = "celcius") -> str:
    """
    Gets the current weather for a given location.
    
    Args:
        location: The city and state/country, e.g. 'San Francisco, CA' or 'Tokyo, Japan'.
        unit: The temperature unit, either 'celcius' or ''fahrenheit.
    """
    
    mock_data = {
        "tokyo": f"18 degrees {unit}, rainy",
        "san francisco": f"21 degree {unit}, sunny",
        "paris": f"15 degrees {unit}, cloudy"
    }
    
    loc_lower = location.lower()
    for city, weather in mock_data.items():
        if city in loc_lower:
            return f"Weather in {location}: {weather}"
        
    return f"Weather in {location}: 20 degrees {unit}, partly cloudy"

def run_manual_tool_loop():
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY")
    
    if not api_key:
        raise ValueError("Error: GEMINI_API_KEY not found. Please check your .env file")
    
    client = genai.Client(api_key=api_key)
    
    chat = client.chats.create(
        model="gemini-2.5-flash",
        config=types.GenerateContentConfig(
            tools=[get_current_weather],
            automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True)
        )
    )
    
    prompt = "What's the weather like in Tokyo right now?"
    print(f"User Prompt: '{prompt}")
    
    response = chat.send_message(prompt)
    
    if response.function_calls:
        call = response.function_calls[0]
        print(f"\n[1] Model requested tool: {call.name}")
        print(f"[2] Tool arguements: {call.args}")
        
        if call.name == "get_current_weather":
            tool_result = get_current_weather(**call.args)
            print(f"[3] Local tool execution result: '{tool_result}'")
            
            final_response = chat.send_message(
                types.Part.from_function_response(
                    name=call.name,
                    response={"result": tool_result}
                )
            )
            
            print("\n---Final Synthesized Response---")
            print(final_response.text)
            
    else:
        print("No function call requested")
        print(response.text)    

if __name__ == "__main__":
    
    run_manual_tool_loop()