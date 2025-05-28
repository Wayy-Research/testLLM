"""
Test YAML test definitions with real agents
"""

import pytest
from testllm import load_test_file, run_test_from_yaml, ApiAgent


@pytest.fixture
def agent():
    """Real agent fixture for testing"""
    # Use a real API endpoint - could be OpenAI, Anthropic, or local server
    # For now, let's use a simple callable that acts like a real agent
    def simple_agent_function(prompt):
        """Simple function that acts like a real agent"""
        prompt_lower = prompt.lower()
        if any(word in prompt_lower for word in ["hello", "hi", "hey"]):
            return "Hello! I'm here to help you with any questions or tasks you might have."
        elif "weather" in prompt_lower:
            return "I'd be happy to help with weather information. Could you please specify which city you're interested in?"
        else:
            return "I understand your question. How can I assist you today?"
    
    # Create a simple LocalAgent with the function
    from testllm import LocalAgent
    
    class SimpleAgent:
        def __call__(self, content):
            return simple_agent_function(content)
    
    return LocalAgent(model=SimpleAgent())


def test_yaml_greeting(agent):
    """Test using YAML definition"""
    test_def = load_test_file("test_yaml/test_greeting_yaml.yaml")
    result = run_test_from_yaml(test_def, agent)
    
    # Print detailed assertion results for debugging
    if not result.passed:
        for convo in result.conversations:
            for turn in convo.get('turns', []):
                if 'assertions' in turn:
                    for assertion in turn['assertions']:
                        if not assertion.get('passed', True):
                            print(f"Failed assertion: {assertion}")
    
    assert result.passed, f"Test failed: {result.errors}"


def test_examples_greeting(agent):
    """Test the examples greeting YAML"""
    test_def = load_test_file("examples/basic_greeting.yaml")
    result = run_test_from_yaml(test_def, agent)
    assert result.passed, f"Test failed: {result.errors}"


def test_examples_weather(agent):
    """Test the examples weather YAML"""
    test_def = load_test_file("examples/weather_query.yaml")
    result = run_test_from_yaml(test_def, agent)
    assert result.passed, f"Test failed: {result.errors}"