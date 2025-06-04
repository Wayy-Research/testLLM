"""
Example test file showing how to use testLLM with pytest using inline test definitions
"""

import pytest
from testllm import ApiAgent, LocalAgent, ConversationTest, AgentAssertion


# Example 1: Testing an API-based agent
@pytest.fixture
def api_agent():
    """Fixture for an agent that communicates via HTTP API"""
    return ApiAgent(
        endpoint="https://api.youragent.com/chat",
        headers={"Authorization": "Bearer your-api-key"},
        timeout=10
    )


# Example 2: Testing a local agent implementation
@pytest.fixture  
def local_agent():
    """Fixture for a local agent implementation"""
    # This is a mock implementation - replace with your actual agent
    class MockAgent:
        def __call__(self, prompt):
            if "hello" in prompt.lower():
                return "Hello! How can I help you today?"
            elif "weather" in prompt.lower():
                if "new york" in prompt.lower():
                    return "The weather in New York is currently 72°F and sunny."
                else:
                    return "I'd be happy to help with weather information. Which city are you interested in?"
            elif "python" in prompt.lower():
                return "Python is a great programming language for many applications!"
            else:
                return "I'm here to help! What would you like to know?"
    
    return LocalAgent(model=MockAgent())


# Example 3: Basic greeting test
def test_greeting_behavior(local_agent):
    """Test basic greeting functionality"""
    test = ConversationTest("greeting_test", "Agent should greet users appropriately")
    
    test.add_turn(
        "Hello there!",
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.max_length(100)
    )
    
    result = test.execute(local_agent)
    assert result.passed, f"Test failed: {result.errors}"


# Example 4: Weather query test
def test_weather_queries(local_agent):
    """Test weather query handling"""
    test = ConversationTest("weather_test", "Agent should handle weather questions")
    
    test.add_turn(
        "What's the weather in New York?",
        AgentAssertion.contains("weather"),
        AgentAssertion.contains("New York"),
        AgentAssertion.sentiment("positive")
    )
    
    result = test.execute(local_agent)
    assert result.passed, f"Test failed: {result.errors}"


# Example 5: Multi-turn conversation test
def test_conversation_flow(local_agent):
    """Test a multi-turn conversation"""
    test = ConversationTest("conversation_flow", "Test conversation flow")
    
    # Turn 1: Greeting
    test.add_turn(
        "Hi there!",
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive")
    )
    
    # Turn 2: Follow-up question  
    test.add_turn(
        "Can you help me with something?",
        AgentAssertion.contains("help"),
        AgentAssertion.max_length(200)
    )
    
    result = test.execute(local_agent)
    assert result.passed, f"Test failed: {result.errors}"


# Example 6: Testing error handling
def test_error_handling(local_agent):
    """Test how agent handles unclear requests"""
    test = ConversationTest("error_handling", "Test unclear input handling")
    
    test.add_turn(
        "asdfkjasdfkj",  # Gibberish input
        AgentAssertion.excludes("error"),  # Should not say "error"
        AgentAssertion.min_length(10)      # Should still provide a response
    )
    
    result = test.execute(local_agent)
    assert result.passed, f"Test failed: {result.errors}"


# Example 7: Testing with complex assertions
def test_complex_assertions(local_agent):
    """Test with multiple assertion types using all_of and any_of"""
    test = ConversationTest("complex_assertions", "Test various assertion combinations")
    
    test.add_turn(
        "Tell me about Python programming",
        AgentAssertion.all_of(
            AgentAssertion.contains("Python"),
            AgentAssertion.excludes("Java"),
            AgentAssertion.max_length(500),
            AgentAssertion.sentiment("positive")
        )
    )
    
    result = test.execute(local_agent)
    assert result.passed, f"Test failed: {result.errors}"


# Example 8: Testing with any_of assertions
def test_flexible_responses(local_agent):
    """Test that agent can respond with any of several acceptable responses"""
    test = ConversationTest("flexible_test", "Test flexible response matching")
    
    test.add_turn(
        "Hello",
        AgentAssertion.any_of(
            AgentAssertion.contains("hello"),
            AgentAssertion.contains("hi"),
            AgentAssertion.contains("greetings")
        ),
        AgentAssertion.sentiment("positive")
    )
    
    result = test.execute(local_agent)
    assert result.passed, f"Test failed: {result.errors}"


# Example 9: Testing response length constraints
def test_response_length(local_agent):
    """Test response length constraints"""
    test = ConversationTest("length_test", "Test response length requirements")
    
    test.add_turn(
        "Give me a brief hello",
        AgentAssertion.min_length(5),   # At least 5 characters
        AgentAssertion.max_length(50),  # No more than 50 characters
        AgentAssertion.contains("hello")
    )
    
    result = test.execute(local_agent)
    assert result.passed, f"Test failed: {result.errors}"


if __name__ == "__main__":
    # Run tests with: python -m pytest test_agent.py -v
    pytest.main([__file__, "-v"])