"""
Test with real Claude/Anthropic agent
"""

import pytest
import os
import requests
from testllm import load_test_file, run_test_from_yaml, LocalAgent

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass


class ClaudeAgent:
    """Simple Anthropic/Claude API wrapper"""
    
    def __init__(self, api_key: str, model: str = "claude-3-haiku-20240307"):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.anthropic.com/v1/messages"
    
    def __call__(self, content: str) -> str:
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model,
            "max_tokens": 150,
            "messages": [{"role": "user", "content": content}]
        }
        
        try:
            response = requests.post(self.base_url, json=payload, headers=headers, timeout=30)
            response.raise_for_status()
            return response.json()["content"][0]["text"].strip()
        except Exception as e:
            return f"Error: {str(e)}"


@pytest.fixture
def claude_agent():
    """Real Claude agent fixture"""
    api_key = os.getenv("ANTHROPIC_API_KEY")
    if not api_key:
        pytest.skip("ANTHROPIC_API_KEY not found - skipping real agent tests")
    
    return LocalAgent(model=ClaudeAgent(api_key))


def test_yaml_with_claude(claude_agent):
    """Test YAML definition with real Claude agent"""
    # Quick test to see if API is working  
    test_response = claude_agent.send_message("test")
    if test_response.startswith("Error:"):
        pytest.skip(f"Claude API not working: {test_response}")
        
    test_def = load_test_file("test_yaml/test_greeting_yaml.yaml")
    result = run_test_from_yaml(test_def, claude_agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_examples_with_claude(claude_agent):
    """Test examples with real Claude agent"""
    test_def = load_test_file("examples/basic_greeting.yaml")
    result = run_test_from_yaml(test_def, claude_agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_sentiment_with_claude(claude_agent):
    """Test sentiment analysis with Claude"""
    # Test that Claude can express positive sentiment
    response = claude_agent.send_message("Say something positive and happy")
    
    # Should contain positive keywords
    positive_words = ['good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 
                      'awesome', 'brilliant', 'perfect', 'love', 'happy', 'glad', 'pleased',
                      'satisfied', 'delighted', 'thrilled', 'excited', 'yes', 'sure', 'absolutely']
    
    response_lower = response.lower()
    has_positive = any(word in response_lower for word in positive_words)
    assert has_positive, f"Claude response should contain positive sentiment: {response}"