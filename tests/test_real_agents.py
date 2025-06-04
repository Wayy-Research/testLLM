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


def test_greeting_with_claude(claude_agent):
    """Test greeting with real Claude agent using inline approach"""
    from testllm import ConversationTest, AgentAssertion
    
    # Quick test to see if API is working  
    test_response = claude_agent.send_message("test")
    if test_response.startswith("Error:"):
        pytest.skip(f"Claude API not working: {test_response}")
    
    test = ConversationTest("claude_greeting", "Claude should respond to greeting appropriately")
    
    test.add_turn(
        "Hello there",
        AgentAssertion.any_of(
            AgentAssertion.contains("hello"),
            AgentAssertion.contains("hi"),
            AgentAssertion.contains("greetings")
        ),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.max_length(300)  # More realistic limit for Claude
    )
    
    result = test.execute(claude_agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_weather_with_claude(claude_agent):
    """Test weather query with real Claude agent"""
    from testllm import ConversationTest, AgentAssertion
    
    # First, let's see what Claude actually responds with
    direct_response = claude_agent.send_message("What's the weather like?")
    print(f"\n=== DIRECT CLAUDE RESPONSE ===")
    print(f"Response: '{direct_response}'")
    print(f"Length: {len(direct_response)} characters")
    print("=" * 50)
    
    test = ConversationTest("claude_weather", "Claude should handle weather questions")
    
    test.add_turn(
        "What's the weather like?",
        AgentAssertion.any_of(
            AgentAssertion.contains("weather"),
            AgentAssertion.contains("information"),
            AgentAssertion.contains("data"),
            AgentAssertion.contains("location"),
            AgentAssertion.contains("assistant"),
            AgentAssertion.contains("access")
        ),
        AgentAssertion.any_of(
            AgentAssertion.sentiment("positive"),
            AgentAssertion.sentiment("neutral")  # Accept neutral for helpful but limitation-explaining responses
        ),
        AgentAssertion.max_length(600)  # Increased limit for Claude's verbose responses
    )
    
    result = test.execute(claude_agent)
    
    # Debug: print detailed assertion results
    print(f"\n=== TEST RESULT DETAILS ===")
    print(f"Test passed: {result.passed}")
    print(f"Errors: {result.errors}")
    
    for convo in result.conversations:
        for turn in convo.get('turns', []):
            if turn.get('role') == 'agent':
                print(f"\nAgent response: '{turn.get('content', 'No content')}'")
                print(f"Response length: {len(turn.get('content', ''))}")
                if 'assertions' in turn:
                    for i, assertion in enumerate(turn['assertions']):
                        # Handle both dict and AssertionResult object formats
                        if hasattr(assertion, 'passed'):
                            # AssertionResult object
                            status = "PASS" if assertion.passed else "FAIL"
                            print(f"  Assertion {i+1} [{status}]: {assertion.assertion_type}")
                            if not assertion.passed:
                                print(f"    Expected: {assertion.expected}")
                                print(f"    Actual: {assertion.actual}")
                                print(f"    Message: {assertion.message}")
                        else:
                            # Dict format
                            status = "PASS" if assertion.get('passed', False) else "FAIL"
                            print(f"  Assertion {i+1} [{status}]: {assertion.get('assertion_type', 'unknown')}")
                            if not assertion.get('passed', False):
                                print(f"    Expected: {assertion.get('expected', 'N/A')}")
                                print(f"    Actual: {assertion.get('actual', 'N/A')}")
                                print(f"    Message: {assertion.get('message', 'N/A')}")
    print("=" * 50)
    
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