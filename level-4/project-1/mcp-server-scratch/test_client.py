import asyncio
import os
import sys
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

async def main():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    server_script = os.path.join(current_dir, "server.py")

    server_params = StdioServerParameters(
        command=sys.executable,
        args=[server_script],
        env=dict(os.environ)
    )

    print(f"[*] Launching MCP Server: {server_script}")
    print("[*] Connecting via stdio...")

    async with stdio_client(server_params) as (read_stream, write_stream):
        async with ClientSession(read_stream, write_stream) as session:
            # 1. Initialize session handshake
            await session.initialize()
            print("[+] Session initialized successfully.\n")

            # 2. Tool Discovery: tools/list
            tools_response = await session.list_tools()
            print("=== Discovered Tools ===")
            for tool in tools_response.tools:
                print(f"-> Tool: {tool.name}")
                print(f"   Description: {tool.description}")
                print(f"   Input Schema: {tool.input_schema}\n")

            # 3. Tool Invocation: tools/call (get_system_info)
            print("=== Invoking: get_system_info ===")
            res_sys = await session.call_tool("get_system_info", arguments={})
            print(f"Output: {res_sys.content[0].text}\n")

            # 4. Tool Invocation: tools/call (compute_statistics)
            print("=== Invoking: compute_statistics ===")
            res_calc = await session.call_tool(
                "compute_statistics",
                arguments={"numbers": [12.5, 45.0, 78.2, 33.1, 99.4], "operation": "mean"}
            )
            print(f"Output: {res_calc.content[0].text}\n")

            # 5. Tool Invocation: tools/call (format_text)
            print("=== Invoking: format_text ===")
            res_format = await session.call_tool(
                "format_text",
                arguments={"text": "hello mcp world from fastmcp", "style": "titlecase"}
            )
            print(f"Output: {res_format.content[0].text}\n")

if __name__ == "__main__":
    asyncio.run(main())