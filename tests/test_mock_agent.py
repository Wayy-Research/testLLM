"""
Test file with a mock agent to demonstrate testLLM framework
Run with: pytest --testllm tests/test_mock_agent.py -v
"""

import pytest
from testllm import LocalAgent, ConversationTest, UserTurn, AgentAssertion


class MockLLMAgent:
    """Mock agent that simulates realistic LLM responses"""
    
    def __init__(self):
        self.conversation_history = []
    
    def generate_response(self, content, **kwargs):
        """Process prompt and return realistic response"""
        prompt_lower = content.lower()
        
        # Greeting responses - use word boundaries to avoid false matches
        import re
        greeting_pattern = r'\b(?:hello|hi|hey|greetings)\b'
        if re.search(greeting_pattern, prompt_lower):
            return "Hello! I'm here to help you with any questions you might have. How can I assist you today?"
        
        # Weather responses
        elif 'weather' in prompt_lower:
            if 'new york' in prompt_lower or 'nyc' in prompt_lower:
                return "The current weather in New York City is 72°F with partly cloudy skies. There's a light breeze from the southwest at 8 mph."
            elif any(city in prompt_lower for city in ['seattle', 'washington']):
                return "Seattle is experiencing typical Pacific Northwest weather - 58°F with light rain and overcast skies."
            else:
                return "I'd be happy to help you with weather information! Could you please specify which city or location you're interested in?"
        
        # Code generation responses
        elif 'function' in prompt_lower and ('fibonacci' in prompt_lower or 'fib' in prompt_lower):
            return """Here's a Python function to calculate Fibonacci numbers:

```python
def fibonacci(n):
    if n <= 1:
        return n
    return fibonacci(n-1) + fibonacci(n-2)

# More efficient iterative version:
def fibonacci_iterative(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b
```"""
        
        # JSON response
        elif 'json' in prompt_lower and 'data' in prompt_lower:
            return '{"mean": 3.0, "median": 3.0, "mode": null, "sum": 15, "count": 5}'
        
        # Help/assistance responses
        elif any(word in prompt_lower for word in ['help', 'assist', 'support']):
            return "I'm here to help! I can assist with various tasks including answering questions, providing information, helping with code, and much more. What specific topic would you like help with?"
        
        # Error handling for unclear input
        elif len(content.strip()) < 3 or any(char in content for char in ['asdf', 'qwer', 'zxcv']):
            return "I didn't quite understand that. Could you please rephrase your question or provide more details about what you're looking for?"
        
        # Default response
        else:
            return f"Thank you for your message: '{content}'. I understand you're asking about this topic. Let me help you with that. Could you provide a bit more context so I can give you the most relevant information?"


@pytest.fixture
def mock_agent():
    """Fixture providing a mock LLM agent for testing"""
    return LocalAgent(model=MockLLMAgent())


@pytest.fixture
def agent(mock_agent):
    """Default agent fixture for YAML tests"""
    return mock_agent


# Test using YAML definition - simplified approach
def test_basic_greeting_yaml(mock_agent):
    """Test basic greeting using YAML test definition"""
    from testllm import load_test_file, run_test_from_yaml
    test_def = load_test_file("examples/basic_greeting.yaml")
    result = run_test_from_yaml(test_def, mock_agent)
    assert result.passed, f"Test failed: {result.errors}"


# Test using programmatic definition
def test_greeting_programmatic(mock_agent):
    """Test greeting using programmatic test definition"""
    test = ConversationTest("programmatic_greeting", "Test greeting with code")
    
    test.add_turn(
        UserTurn("Hi there!"),
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.max_length(200)
    )
    
    result = test.execute(mock_agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_weather_query(mock_agent):
    """Test weather query handling"""
    test = ConversationTest("weather_test", "Test weather information requests")
    
    # Test specific city
    test.add_turn(
        UserTurn("What's the weather in New York?"),
        AgentAssertion.contains("New York"),
        AgentAssertion.contains("°F"),
        AgentAssertion.excludes("I don't know"),
        AgentAssertion.max_length(500)
    )
    
    result = test.execute(mock_agent)
    assert result.passed, f"Weather test failed: {result.errors}"


def test_multi_turn_conversation(mock_agent):
    """Test multi-turn conversation flow"""
    test = ConversationTest("multi_turn", "Test conversation flow")
    
    # Turn 1: Greeting
    test.add_turn(
        UserTurn("Hello!"),
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive")
    )
    
    # Turn 2: Ask for help
    test.add_turn(
        UserTurn("Can you help me with something?"),
        AgentAssertion.contains("help"),
        AgentAssertion.excludes("sorry"),
        AgentAssertion.min_length(20)
    )
    
    result = test.execute(mock_agent)
    assert result.passed, f"Multi-turn test failed: {result.errors}"


def test_code_generation(mock_agent):
    """Test code generation capabilities"""
    test = ConversationTest("code_gen", "Test code generation")
    
    test.add_turn(
        UserTurn("Write a function to calculate Fibonacci numbers"),
        AgentAssertion.contains("def"),
        AgentAssertion.contains("fibonacci"),
        AgentAssertion.regex(r"def\s+\w+\s*\("),  # Function definition pattern
        AgentAssertion.max_length(1000)
    )
    
    result = test.execute(mock_agent)
    assert result.passed, f"Code generation test failed: {result.errors}"


def test_json_response(mock_agent):
    """Test JSON response validation"""
    test = ConversationTest("json_test", "Test JSON output")
    
    test.add_turn(
        UserTurn("Analyze this data and return JSON: [1,2,3,4,5]"),
        AgentAssertion.is_valid_json(),
        AgentAssertion.contains("mean")
    )
    
    result = test.execute(mock_agent)
    
    # Debug: print the actual response
    if not result.passed:
        for convo in result.conversations:
            for turn in convo.get('turns', []):
                if turn.get('role') == 'agent':
                    print(f"Agent response: {turn.get('content', 'No content')}")
    
    assert result.passed, f"JSON test failed: {result.errors}"


def test_error_handling(mock_agent):
    """Test how agent handles unclear input"""
    test = ConversationTest("error_handling", "Test unclear input handling")
    
    test.add_turn(
        UserTurn("asdfkjasdfkj"),  # Gibberish input
        AgentAssertion.excludes("error"),  # Should not say "error"
        AgentAssertion.min_length(10),     # Should still provide a response
        AgentAssertion.contains("understand")  # Should indicate confusion politely
    )
    
    result = test.execute(mock_agent)
    assert result.passed, f"Error handling test failed: {result.errors}"


def test_composite_assertions(mock_agent):
    """Test composite assertion types"""
    test = ConversationTest("composite", "Test composite assertions")
    
    test.add_turn(
        UserTurn("Hello, can you help me?"),
        AgentAssertion.all_of([
            AgentAssertion.contains("hello"),
            AgentAssertion.contains("help"),
            AgentAssertion.sentiment("positive"),
            AgentAssertion.max_length(300)
        ])
    )
    
    result = test.execute(mock_agent)
    assert result.passed, f"Composite assertion test failed: {result.errors}"


def test_any_of_assertion(mock_agent):
    """Test any_of assertion type"""
    test = ConversationTest("any_of", "Test any_of assertion")
    
    test.add_turn(
        UserTurn("What's the weather?"),  # No location specified
        AgentAssertion.any_of([
            AgentAssertion.contains("location"),
            AgentAssertion.contains("where"),
            AgentAssertion.contains("city"),
            AgentAssertion.contains("specify")
        ])
    )
    
    result = test.execute(mock_agent)
    assert result.passed, f"Any_of assertion test failed: {result.errors}"


if __name__ == "__main__":
    # Run tests directly
    import sys
    import subprocess
    
    print("Running testLLM demonstration tests...")
    print("=" * 50)
    
    # Run with pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "--testllm", 
        "-v",
        "--tb=short"
    ], capture_output=False)
    
    sys.exit(result.returncode)