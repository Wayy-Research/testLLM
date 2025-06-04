"""
Test agent conversations using inline test definitions
"""

import pytest
from testllm import LocalAgent, ConversationTest, AgentAssertion


@pytest.fixture
def agent():
    """Simple agent fixture for testing"""
    def simple_agent_function(prompt):
        """Simple function that acts like a real agent"""
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm here to help you with any questions or tasks you might have."
        elif "weather" in prompt_lower:
            return "I'd be happy to help with weather information. Could you please specify which city you're interested in?"
        else:
            return "I understand your question. How can I assist you today?"
    
    class SimpleAgent:
        def __call__(self, content):
            return simple_agent_function(content)
    
    return LocalAgent(model=SimpleAgent())


def test_basic_greeting(agent):
    """Test basic greeting interaction"""
    test = ConversationTest("basic_greeting", "Agent should respond to greeting appropriately")
    
    test.add_turn(
        "Hello there",
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.max_length(150)
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_weather_inquiry(agent):
    """Test weather-related inquiry"""
    test = ConversationTest("weather_inquiry", "Agent should handle weather questions appropriately")
    
    test.add_turn(
        "What's the weather like?",
        AgentAssertion.contains("weather"),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.max_length(200)
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_multi_turn_conversation(agent):
    """Test multi-turn conversation"""
    test = ConversationTest("multi_turn", "Agent should handle multiple conversation turns")
    
    # First turn - greeting
    test.add_turn(
        "Hi there!",
        AgentAssertion.contains("hello", case_sensitive=False),
        AgentAssertion.sentiment("positive")
    )
    
    # Second turn - follow-up question  
    test.add_turn(
        "How are you today?",
        AgentAssertion.any_of(
            AgentAssertion.contains("good"),
            AgentAssertion.contains("help"),
            AgentAssertion.contains("assist")
        ),
        AgentAssertion.min_length(10)
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_assertion_combinations(agent):
    """Test various assertion combinations"""
    test = ConversationTest("assertion_test", "Test different assertion types")
    
    test.add_turn(
        "Hello",
        AgentAssertion.all_of(
            AgentAssertion.contains("hello"),
            AgentAssertion.sentiment("positive"),
            AgentAssertion.min_length(5),
            AgentAssertion.max_length(200)
        )
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"