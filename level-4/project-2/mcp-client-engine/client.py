import asyncio
import os
import sys
from pathlib import Path
from dotenv import load_dotenv

from google import genai
from google.genai import types

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client


load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Anchor resolution relative to this file: client.py -> mcp-client-engine -> level-4 -> project-1
SERVER_SCRIPT_PATH = (
    Path(__file__).resolve().parent.parent.parent
    / "project-1"
    / "mcp-server-scratch"
    / "server.py"
)

def map_mcp_to_genai_tool(mcp_tool) -> types.FunctionDeclaration:
    """Converts an MCP tool definition into a Google GenAI FunctionDeclaration."""
    # Ensure title field is stripped if present to keep schema minimal
    schema = dict(mcp_tool.input_schema)
    schema.pop("title", None)
    
    return types.FunctionDeclaration(
        name=mcp_tool.name,
        description=mcp_tool.description or "",
        parameters=schema,
    )

async def execute_mcp_tool(session: ClientSession, name: str, args: dict) -> str:
    """Executes a tool on the MCP server and returns the aggregated text response."""
    try:
        result = await session.call_tool(name, arguments=args)
        
        # Consolidate all text blocks returned by the MCP tool
        text_outputs = []
        for content in result.content:
            if content.type == "text":
                text_outputs.append(content.text)
            else:
                text_outputs.append(str(content))
        
        return "\n".join(text_outputs) if text_outputs else "Success (no output)"
    except Exception as e:
        # Graceful error capture returned to LLM
        return f"Error executing tool '{name}': {str(e)}"

async def query_llm_with_mcp(prompt: str):
    if not SERVER_SCRIPT_PATH.exists():
        raise FileNotFoundError(f"Server script not found at: {SERVER_SCRIPT_PATH}")

    # Initialize Google GenAI client
    ai_client = genai.Client()
    model_name = "gemini-3.1-flash-lite"

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_SCRIPT_PATH)],
        env={**os.environ}
    )

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            print("[+] Connected to MCP Server.")

            # 1. Discover and map tools
            tools_response = await session.list_tools()
            genai_functions = [map_mcp_to_genai_tool(t) for t in tools_response.tools]
            tools_config = [types.Tool(function_declarations=genai_functions)]
            
            print(f"[+] Mapped {len(genai_functions)} tools into GenAI declarations.")
            print(f"\n[User Query]: {prompt}\n")

            # 2. First turn: Send user prompt + available tools
            config = types.GenerateContentConfig(
                tools=tools_config,
                temperature=0.0
            )

            contents = [prompt]
            response = ai_client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )

            # 3. Check for tool call requests
            if response.function_calls:
                print(f"[LLM Action] Model requested {len(response.function_calls)} tool call(s):")
                
                # Append model's turn (with function calls) to history
                contents.append(response.candidates[0].content)

                # Execute requested tools against MCP server
                tool_response_parts = []
                for call in response.function_calls:
                    print(f" └─ Executing '{call.name}' with args: {call.args}")
                    raw_result = await execute_mcp_tool(session, call.name, call.args)
                    print(f"    Raw Output: {raw_result.strip()}")

                    tool_response_parts.append(
                        types.Part.from_function_response(
                            name=call.name,
                            response={"result": raw_result}
                        )
                    )

                # Append tool execution results as a user-role response turn
                contents.append(types.Content(role="user", parts=tool_response_parts))

                # 4. Final synthesis turn
                print("\n[LLM Synthesis] Generating final response...")
                final_response = ai_client.models.generate_content(
                    model=model_name,
                    contents=contents,
                    config=config
                )
                print(f"\n[Final Answer]:\n{final_response.text}")
            else:
                print(f"\n[Direct Answer (No Tools Used)]:\n{response.text}")

if __name__ == "__main__":
    test_prompt = "What OS is this machine running, and what is the mean and standard deviation of 12, 45, 78, 90, 105?"
    asyncio.run(query_llm_with_mcp(test_prompt))