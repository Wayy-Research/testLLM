"""
Example of writing tests using the inline approach with testLLM
This example shows how to write agent tests directly in Python without YAML files
"""

import pytest
from testllm import LocalAgent, ConversationTest, AgentAssertion


@pytest.fixture
def my_agent():
    """Example agent fixture - replace with your actual agent"""
    def example_agent(prompt):
        """Simple example agent that responds to basic queries"""
        prompt_lower = prompt.lower()
        
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! How can I help you today?"
        elif "weather" in prompt_lower:
            return "I can help with weather information. Which city are you interested in?"
        elif "goodbye" in prompt_lower or "bye" in prompt_lower:
            return "Goodbye! Have a great day!"
        else:
            return "I understand your question. Let me help you with that."
    
    # Wrap your agent function in a simple callable class
    class SimpleAgent:
        def __call__(self, content):
            return example_agent(content)
    
    return LocalAgent(model=SimpleAgent())


def test_greeting_conversation(my_agent):
    """Test basic greeting functionality"""
    test = ConversationTest("greeting_test", "Agent should greet users appropriately")
    
    test.add_turn(
        "Hello there!",
        AgentAssertion.contains("hello"),  # Response should contain "hello"
        AgentAssertion.sentiment("positive"),  # Response should be positive
        AgentAssertion.max_length(100)  # Response should be under 100 characters
    )
    
    result = test.execute(my_agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_weather_inquiry(my_agent):
    """Test weather-related queries"""
    test = ConversationTest("weather_test", "Agent should handle weather questions")
    
    test.add_turn(
        "What's the weather like?",
        AgentAssertion.contains("weather"),
        AgentAssertion.contains("city"),  # Should ask for city
        AgentAssertion.sentiment("positive")
    )
    
    result = test.execute(my_agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_multi_turn_conversation(my_agent):
    """Test a complete conversation flow"""
    test = ConversationTest("conversation_flow", "Test multi-turn conversation")
    
    # Turn 1: Greeting
    test.add_turn(
        "Hi!",
        AgentAssertion.contains("hello", case_sensitive=False),
        AgentAssertion.sentiment("positive")
    )
    
    # Turn 2: Question
    test.add_turn(
        "Can you help me with weather?",
        AgentAssertion.any_of(
            AgentAssertion.contains("weather"),
            AgentAssertion.contains("help"),
            AgentAssertion.contains("city")
        )
    )
    
    # Turn 3: Goodbye
    test.add_turn(
        "Thanks, goodbye!",
        AgentAssertion.contains("goodbye", case_sensitive=False),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.min_length(5)
    )
    
    result = test.execute(my_agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_complex_assertions(my_agent):
    """Example of more complex assertion combinations"""
    test = ConversationTest("complex_test", "Test complex assertion logic")
    
    test.add_turn(
        "Hello",
        AgentAssertion.all_of(
            AgentAssertion.contains("hello"),
            AgentAssertion.sentiment("positive"),
            AgentAssertion.min_length(10),
            AgentAssertion.max_length(200)
        )
    )
    
    result = test.execute(my_agent)
    assert result.passed, f"Test failed: {result.errors}"


# If you want to run this example directly
if __name__ == "__main__":
    # Create an agent instance
    def example_agent(prompt):
        prompt_lower = prompt.lower()
        if "hello" in prompt_lower:
            return "Hello! How can I help you today?"
        return "I understand your question. Let me help you with that."
    
    class SimpleAgent:
        def __call__(self, content):
            return example_agent(content)
    
    agent = LocalAgent(model=SimpleAgent())
    
    # Run a simple test
    test = ConversationTest("demo", "Demo test")
    test.add_turn(
        "Hello!",
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive")
    )
    
    result = test.execute(agent)
    print(f"Test passed: {result.passed}")
    if not result.passed:
        print(f"Errors: {result.errors}")