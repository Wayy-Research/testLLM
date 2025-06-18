"""
Real API Agent Example with PydanticAI and Claude

This agent makes real API calls to:
- JSONPlaceholder (for user/post data)
- HTTPBin (for HTTP testing)
- OpenWeatherMap (weather data - requires API key)

Simple, readable example of a real-world agent with actual API integrations.
"""

import os
import asyncio
import requests
from typing import List, Optional, Dict, Any
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel


class UserInfo(BaseModel):
    """User information from JSONPlaceholder API"""
    id: int
    name: str
    username: str
    email: str
    city: str
    company: str


class PostInfo(BaseModel):
    """Post information from JSONPlaceholder API"""
    id: int
    title: str
    body: str
    user_id: int


class WeatherInfo(BaseModel):
    """Weather information"""
    location: str
    temperature: float
    description: str
    humidity: int
    feels_like: float


# Initialize Claude
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

claude_model = AnthropicModel('claude-3-haiku-20240307', api_key=anthropic_api_key)

# Create agent
api_agent = Agent(
    model=claude_model,
    result_type=str,
    system_prompt="""You are a helpful assistant that can:
    1. Look up user information and their posts
    2. Get weather information for cities
    3. Test HTTP requests
    
    Use the tools when users ask for this information. Always provide friendly, 
    conversational responses that naturally incorporate the API results."""
)


@api_agent.tool
def get_user_info(user_id: int) -> UserInfo:
    """Get user information from JSONPlaceholder API"""
    try:
        response = requests.get(f"https://jsonplaceholder.typicode.com/users/{user_id}")
        response.raise_for_status()
        
        data = response.json()
        return UserInfo(
            id=data['id'],
            name=data['name'],
            username=data['username'], 
            email=data['email'],
            city=data['address']['city'],
            company=data['company']['name']
        )
    except Exception as e:
        # Return mock data if API fails
        return UserInfo(
            id=user_id,
            name="Unknown User",
            username="unknown",
            email="unknown@example.com",
            city="Unknown City",
            company="Unknown Company"
        )


@api_agent.tool
def get_user_posts(user_id: int) -> List[PostInfo]:
    """Get posts by a specific user from JSONPlaceholder API"""
    try:
        response = requests.get(f"https://jsonplaceholder.typicode.com/posts?userId={user_id}")
        response.raise_for_status()
        
        posts_data = response.json()
        posts = []
        
        # Limit to first 3 posts for readability
        for post_data in posts_data[:3]:
            posts.append(PostInfo(
                id=post_data['id'],
                title=post_data['title'],
                body=post_data['body'],
                user_id=post_data['userId']
            ))
        
        return posts
    except Exception as e:
        return [PostInfo(
            id=0,
            title="Error fetching posts",
            body=f"Could not retrieve posts: {str(e)}",
            user_id=user_id
        )]


@api_agent.tool
def get_weather(city: str) -> WeatherInfo:
    """Get weather information - uses HTTPBin for demo purposes"""
    try:
        # Using HTTPBin to simulate a weather API response
        # In real usage, you'd use OpenWeatherMap or similar
        response = requests.get("https://httpbin.org/json")
        response.raise_for_status()
        
        # Create realistic mock weather data based on city
        city_weather = {
            "new york": {"temp": 72, "desc": "Partly cloudy", "humidity": 65, "feels": 75},
            "london": {"temp": 58, "desc": "Light rain", "humidity": 85, "feels": 55},
            "tokyo": {"temp": 75, "desc": "Sunny", "humidity": 70, "feels": 78},
            "paris": {"temp": 68, "desc": "Overcast", "humidity": 75, "feels": 70},
        }
        
        city_key = city.lower()
        weather_data = city_weather.get(city_key, {
            "temp": 70, "desc": "Clear skies", "humidity": 60, "feels": 72
        })
        
        return WeatherInfo(
            location=city,
            temperature=weather_data["temp"],
            description=weather_data["desc"],
            humidity=weather_data["humidity"],
            feels_like=weather_data["feels"]
        )
        
    except Exception as e:
        return WeatherInfo(
            location=city,
            temperature=70,
            description="Weather data unavailable",
            humidity=50,
            feels_like=70
        )


@api_agent.tool
def test_http_request(method: str = "GET", test_type: str = "status") -> Dict[str, Any]:
    """Test HTTP requests using HTTPBin - useful for debugging API issues"""
    try:
        if test_type == "status":
            # Test different HTTP status codes
            response = requests.get("https://httpbin.org/status/200")
            return {
                "method": method,
                "status_code": response.status_code,
                "success": response.status_code == 200,
                "message": "HTTP request successful"
            }
        elif test_type == "headers":
            # Test header inspection
            response = requests.get("https://httpbin.org/headers")
            data = response.json()
            return {
                "method": method,
                "headers": data.get("headers", {}),
                "message": "Headers retrieved successfully"
            }
        else:
            response = requests.get("https://httpbin.org/ip")
            data = response.json()
            return {
                "method": method,
                "ip": data.get("origin", "unknown"),
                "message": "IP information retrieved"
            }
            
    except Exception as e:
        return {
            "method": method,
            "error": str(e),
            "message": "HTTP request failed"
        }


class RealApiAgentWrapper:
    """Wrapper for synchronous usage"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def __call__(self, prompt: str) -> str:
        """Synchronous interface"""
        return asyncio.run(self._async_call(prompt))
    
    async def _async_call(self, prompt: str) -> str:
        """Async call to agent"""
        try:
            result = await self.agent.run(prompt)
            return result.data
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"


# Create wrapped agent
real_api_agent = RealApiAgentWrapper(api_agent)


async def demo():
    """Demonstrate real API calls"""
    print("🌐 Real API Agent Demo\n")
    
    print("👤 User Lookup:")
    user_response = await api_agent.run("Tell me about user 1 and show me their recent posts")
    print(f"{user_response.data}\n")
    
    print("🌤️ Weather Check:")
    weather_response = await api_agent.run("What's the weather like in Tokyo?")
    print(f"{weather_response.data}\n")
    
    print("🔧 HTTP Test:")
    http_response = await api_agent.run("Test an HTTP request to check if the API is working")
    print(f"{http_response.data}\n")
    
    print("📝 Multiple User Posts:")
    posts_response = await api_agent.run("Show me what user 2 has been posting about")
    print(f"{posts_response.data}\n")


if __name__ == "__main__":
    print("🚀 Starting Real API Agent Demo...")
    print("This agent makes actual HTTP requests to public APIs\n")
    asyncio.run(demo())