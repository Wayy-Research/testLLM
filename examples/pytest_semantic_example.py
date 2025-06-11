"""
Example of using pytest with semantic testing
"""

import pytest
from testllm import LocalAgent
from testllm.pytest_semantic import semantic_test, SemanticTestRunner


@pytest.fixture
def example_agent():
    """Example agent fixture"""
    def agent_function(prompt):
        prompt_lower = prompt.lower()
        
        if "hello" in prompt_lower or "hi" in prompt_lower:
            return "Hello! How can I help you today?"
        elif "weather" in prompt_lower:
            return "I can help with weather information. Which city are you interested in?"
        elif "goodbye" in prompt_lower or "bye" in prompt_lower:
            return "Goodbye! Have a great day!"
        elif "math" in prompt_lower or "calculate" in prompt_lower:
            return "I can help with basic math calculations. What would you like me to calculate?"
        else:
            return "I understand your question. Let me help you with that."
    
    class SimpleAgent:
        def __call__(self, content):
            return agent_function(content)
    
    return LocalAgent(model=SimpleAgent())


@semantic_test("greeting_test", "Test greeting functionality with semantic evaluation")
async def test_greeting_semantic(agent):
    """Test agent greeting using semantic evaluation"""
    from testllm import SemanticTest
    
    test = SemanticTest("greeting", "Test greeting responses")
    test.add_test_case(
        "Hello there!",
        "The response should be a friendly greeting",
        "The response should offer to help the user",
        "The tone should be welcoming and professional"
    )
    
    return await test.execute(agent)


@semantic_test("weather_test", "Test weather inquiries")
async def test_weather_semantic(agent):
    """Test weather-related queries"""
    from testllm import SemanticTest
    
    test = SemanticTest("weather", "Weather query handling")
    test.add_test_case(
        "What's the weather like?",
        "Response should acknowledge the weather question",
        "Response should ask for location if needed",
        "Response should be helpful and relevant"
    )
    
    return await test.execute(agent)


# Alternative approach using SemanticTestRunner fixture
async def test_math_inquiry_with_runner(example_agent, semantic_runner):
    """Test math inquiries using the semantic runner fixture"""
    test = semantic_runner.create_test("math_test", "Test math inquiry handling")
    
    test.add_test_case(
        "I need help with some calculations",
        "Response should recognize this as a math request",
        "Response should offer to help with calculations",
        "Response should be professional and competent"
    )
    
    results = await semantic_runner.run_test(example_agent, test)
    semantic_runner.assert_all_passed(results)


async def test_multi_criteria_evaluation(example_agent, semantic_runner):
    """Test with multiple semantic criteria"""
    test = semantic_runner.create_test("multi_criteria", "Multiple criteria test")
    
    test.add_test_case(
        "Hi, can you help me?",
        "Response should acknowledge the greeting",
        "Response should confirm willingness to help",
        "Response should be concise and not overly verbose",
        "Response should maintain a helpful tone"
    )
    
    results = await semantic_runner.run_test(example_agent, test)
    
    # Custom assertions on results
    assert len(results) == 1
    result = results[0]
    assert result.consensus_score > 0.5, f"Expected consensus score > 0.5, got {result.consensus_score}"
    
    # Check individual criteria
    assert len(result.evaluation_results) == 4
    for eval_result in result.evaluation_results:
        print(f"Criterion: {eval_result['criterion']} - Score: {eval_result['consensus_score']:.2f}")