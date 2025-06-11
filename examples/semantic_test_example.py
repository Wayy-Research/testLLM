"""
Example of the new semantic testing interface for testLLM
This demonstrates LLM-evaluated testing as the primary testing paradigm
"""

import pytest
from testllm import LocalAgent, SemanticTest, semantic_test, pytest_semantic_test


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
        elif "math" in prompt_lower or "calculate" in prompt_lower:
            return "I can help with mathematical calculations. What would you like me to compute?"
        else:
            return "I understand your question. Let me help you with that."
    
    class SimpleAgent:
        def __call__(self, content):
            return example_agent(content)
    
    return LocalAgent(model=SimpleAgent())


# Example 1: Basic semantic test using the SemanticTest class
def test_greeting_behavior_semantic(my_agent):
    """Test agent greeting behavior using semantic evaluation"""
    test = SemanticTest(
        test_id="greeting_test",
        description="Test how the agent handles greetings",
        evaluator_models=["gpt-4o-mini"],  # Can specify multiple models
        consensus_threshold=0.7
    )
    
    # Add test cases with natural language criteria
    test.add_case(
        "Hello there!",
        "Response should be a friendly greeting",
        "Response should offer help or assistance",
        "Response should be professional but warm"
    )
    
    test.add_case(
        "Hi!",
        "Response should acknowledge the greeting",
        "Response should invite further interaction"
    )
    
    # Execute the test (async)
    results = test.execute_sync(my_agent)
    
    # All test cases should pass
    assert all(result.passed for result in results), \
        f"Some test cases failed: {[r.test_id for r in results if not r.passed]}"


# Example 2: Using the semantic_test factory function
def test_weather_inquiry_semantic(my_agent):
    """Test weather-related queries using semantic evaluation"""
    test = semantic_test(
        "weather_test", 
        "Test agent's handling of weather questions"
    )
    
    test.add_case(
        "What's the weather like?",
        "Response should acknowledge the weather inquiry",
        "Response should ask for location information",
        "Response should be helpful and informative"
    )
    
    test.add_case(
        "Will it rain tomorrow?",
        "Response should understand this is a weather prediction question",
        "Response should indicate need for location or offer general guidance"
    )
    
    results = test.execute_sync(my_agent)
    assert all(result.passed for result in results)


# Example 3: Using the pytest decorator for cleaner syntax
@pytest_semantic_test(
    "math_capability_test",
    "Test agent's mathematical capabilities",
    consensus_threshold=0.8
)
def test_math_capabilities(my_agent):
    """Test mathematical query handling"""
    return [
        ("Can you help me with math?", [
            "Response should acknowledge the math request",
            "Response should offer to help with calculations",
            "Response should be encouraging and supportive"
        ]),
        ("I need to calculate something", [
            "Response should show readiness to help with calculations",
            "Response should ask what to calculate or offer assistance"
        ])
    ]


# Example 4: Multi-turn conversation testing
def test_conversation_flow_semantic(my_agent):
    """Test a complete conversation flow with semantic evaluation"""
    test = SemanticTest("conversation_flow", "Test multi-turn conversation")
    
    # First turn: Greeting
    test.add_case(
        "Hi there!",
        "Response should be a warm greeting",
        "Response should invite further interaction"
    )
    
    # Second turn: Request
    test.add_case(
        "Can you help me with weather information?",
        "Response should acknowledge the weather request",
        "Response should ask for specifics like location",
        "Response should maintain helpful tone"
    )
    
    # Third turn: Goodbye
    test.add_case(
        "Thanks, goodbye!",
        "Response should acknowledge the thanks",
        "Response should provide a polite farewell",
        "Response should be positive and closing"
    )
    
    results = test.execute_sync(my_agent)
    
    # Check overall conversation quality
    overall_score = sum(r.overall_score for r in results) / len(results)
    assert overall_score >= 0.7, f"Overall conversation score too low: {overall_score}"
    assert all(result.passed for result in results)


# Example 5: Testing with different evaluation models
def test_with_multiple_evaluators(my_agent):
    """Test using multiple LLM evaluators for consensus"""
    test = SemanticTest(
        "multi_evaluator_test",
        "Test with multiple LLM evaluators",
        evaluator_models=["gpt-4o-mini", "claude-3-haiku-20240307"],  # Multiple models
        consensus_threshold=0.75
    )
    
    test.add_case(
        "Hello! How are you doing today?",
        "Response should be friendly and engaging",
        "Response should reciprocate the greeting appropriately",
        "Response should maintain professional boundaries while being warm"
    )
    
    results = test.execute_sync(my_agent)
    
    # Check that we got evaluations from multiple models
    for result in results:
        for criterion_result in result.criterion_results:
            evaluator_models = [eval_res["evaluator"] for eval_res in criterion_result["evaluations"]]
            assert len(set(evaluator_models)) > 1, "Should have multiple evaluators"
    
    assert all(result.passed for result in results)


# Example 6: Advanced criteria testing
def test_advanced_semantic_criteria(my_agent):
    """Test with more sophisticated semantic criteria"""
    test = semantic_test("advanced_criteria", "Test advanced semantic evaluation")
    
    test.add_case(
        "I'm feeling confused about something",
        "Response should show empathy and understanding",
        "Response should offer support without being presumptuous",
        "Response should invite the user to share more details",
        "Response should avoid giving premature advice",
        "Tone should be supportive but not overly emotional"
    )
    
    results = test.execute_sync(my_agent)
    
    # Print detailed results for inspection
    for result in results:
        print(f"Test: {result.test_id}")
        print(f"Input: {result.user_input}")
        print(f"Response: {result.agent_response}")
        print(f"Overall Score: {result.overall_score:.2f}")
        print(f"Passed: {result.passed}")
        
        for criterion_result in result.criterion_results:
            print(f"  Criterion: {criterion_result['criterion']}")
            print(f"  Score: {criterion_result['consensus_score']:.2f}")
            print(f"  Passed: {criterion_result['passed']}")
        print()
    
    assert all(result.passed for result in results)


if __name__ == "__main__":
    # Run a simple demo
    def demo_agent(prompt):
        if "hello" in prompt.lower():
            return "Hello! How can I help you today?"
        return "I understand your question. Let me help you with that."
    
    class SimpleAgent:
        def __call__(self, content):
            return demo_agent(content)
    
    agent = LocalAgent(model=SimpleAgent())
    
    # Create and run a semantic test
    test = semantic_test("demo", "Demo semantic test")
    test.add_case(
        "Hello!",
        "Response should be a friendly greeting",
        "Response should offer help"
    )
    
    results = test.execute_sync(agent)
    
    print("Demo Results:")
    for result in results:
        print(f"Passed: {result.passed}")
        print(f"Score: {result.overall_score:.2f}")
        print(f"Input: {result.user_input}")
        print(f"Response: {result.agent_response}")