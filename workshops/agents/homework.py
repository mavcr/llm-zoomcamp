from fastmcp import FastMCP
import random

mcp = FastMCP("Weather Server")

known_weather_data = {'berlin': 20.0}

@mcp.tool
def get_weather(city: str) -> float:
    """Get weather for a city"""
    city = city.strip().lower()
    if city in known_weather_data:
        return known_weather_data[city]
    return round(random.uniform(-5, 35), 1)

@mcp.tool
def set_weather(city: str, temp: float) -> None:
    city = city.strip().lower()
    known_weather_data[city] = temp
    return 'OK'
