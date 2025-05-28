"""
Example test file showing how to use testLLM with pytest
"""

import pytest
from testllm import agent_test, ApiAgent, LocalAgent, ConversationTest, UserTurn, AgentAssertion


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
        def process_prompt(self, prompt):
            if "hello" in prompt.lower():
                return "Hello! How can I help you today?"
            elif "weather" in prompt.lower():
                if "new york" in prompt.lower():
                    return "The weather in New York is currently 72°F and sunny."
                else:
                    return "I'd be happy to help with weather information. Which city are you interested in?"
            else:
                return "I'm here to help! What would you like to know?"
    
    return LocalAgent(model=MockAgent())


# Example 3: Using YAML test definitions
@agent_test("examples/basic_greeting.yaml")
def test_greeting_behavior(local_agent):
    """Test basic greeting using YAML definition"""
    pass  # The decorator handles everything


@agent_test("examples/weather_query.yaml") 
def test_weather_queries(local_agent):
    """Test weather query handling using YAML definition"""
    pass


# Example 4: Programmatic test definition
def test_conversation_flow(local_agent):
    """Test a multi-turn conversation programmatically"""
    test = ConversationTest("multi_turn_conversation", "Test conversation flow")
    
    # Turn 1: Greeting
    test.add_turn(
        UserTurn("Hi there!"),
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive")
    )
    
    # Turn 2: Follow-up question  
    test.add_turn(
        UserTurn("Can you help me with something?"),
        AgentAssertion.contains("help"),
        AgentAssertion.max_length(200)
    )
    
    result = test.execute(local_agent)
    assert result.passed, f"Test failed: {result.errors}"


# Example 5: Testing JSON responses
def test_json_response(local_agent):
    """Test that agent can return valid JSON when requested"""
    test = ConversationTest("json_response", "Test JSON output")
    
    test.add_turn(
        UserTurn("Return your response as JSON with a 'message' field"),
        AgentAssertion.is_valid_json()
    )
    
    result = test.execute(local_agent)
    assert result.passed


# Example 6: Testing error handling
def test_error_handling(local_agent):
    """Test how agent handles unclear requests"""
    test = ConversationTest("error_handling", "Test unclear input handling")
    
    test.add_turn(
        UserTurn("asdfkjasdfkj"),  # Gibberish input
        AgentAssertion.excludes("error"),  # Should not say "error"
        AgentAssertion.min_length(10)      # Should still provide a response
    )
    
    result = test.execute(local_agent)
    assert result.passed


# Example 7: Testing with custom assertions
def test_custom_behavior(local_agent):
    """Test with multiple assertion types"""
    test = ConversationTest("custom_behavior", "Test various assertion types")
    
    test.add_turn(
        UserTurn("Tell me about Python programming"),
        AgentAssertion.all_of([
            AgentAssertion.contains("Python"),
            AgentAssertion.excludes("Java"),
            AgentAssertion.max_length(500),
            AgentAssertion.sentiment("positive")
        ])
    )
    
    result = test.execute(local_agent)
    assert result.passed


if __name__ == "__main__":
    # Run tests with: python -m pytest test_agent.py --testllm
    pytest.main([__file__, "--testllm", "-v"])