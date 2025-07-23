import asyncio
from fastmcp import Client
import homework


async def main():
    async with Client(homework.mcp) as client:
        tools = await client.list_tools()
        print(tools)

        # Call a tool
        result = await client.call_tool("get_weather", {"city": "berlin"})
        print("Berlin weather:", result)


if __name__ == "__main__":
    asyncio.run(main())