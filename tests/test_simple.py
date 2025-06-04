"""
Simple tests that work without complex mocks
"""

import pytest
from testllm import load_test_file, run_test_from_yaml, LocalAgent, ConversationTest, UserTurn, AgentAssertion


class BasicAgent:
    """Basic agent that gives reasonable responses"""
    
    def __call__(self, content):
        content_lower = content.lower()
        
        if any(word in content_lower for word in ['hello', 'hi', 'hey']):
            return "Hello! I'm here to help you with any questions or tasks you might have."
        elif 'weather' in content_lower:
            return "I'd be happy to help with weather information. Could you please specify which city you're interested in?"
        else:
            return "I understand your question. How can I assist you today?"


@pytest.fixture
def agent():
    """Simple working agent"""
    return LocalAgent(model=BasicAgent())


def test_basic_conversation(agent):
    """Test basic conversation without YAML"""
    test = ConversationTest("basic_test", "Basic conversation test")
    
    test.add_turn(
        UserTurn("Hello"),
        AgentAssertion.contains("Hello"),
        AgentAssertion.max_length(200)
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_greeting_inline(agent):
    """Test greeting using inline definition"""
    test = ConversationTest("inline_greeting", "Test inline greeting")
    
    test.add_turn(
        "Hello there",
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.max_length(150)
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_weather_inline(agent):
    """Test weather query using inline definition"""
    test = ConversationTest("inline_weather", "Test inline weather query")
    
    test.add_turn(
        "What's the weather like?",
        AgentAssertion.contains("weather"),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.max_length(200)
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"