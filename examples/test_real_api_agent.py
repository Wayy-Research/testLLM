"""
Tests for Real API Agent - Comprehensive Testing Example

These tests showcase testLLM's testing capabilities by evaluating both
traditional assertions and semantic understanding of agent responses.

Run with: pytest test_real_api_agent.py -v
"""

import pytest
from testllm import LocalAgent, ConversationTest, AgentAssertion, SemanticTest
from real_api_agent import real_api_agent


@pytest.fixture
def api_agent():
    """Fixture for the real API agent"""
    return LocalAgent(model=real_api_agent)


def test_user_information_traditional(api_agent):
    """Traditional test for user information retrieval"""
    test = ConversationTest(
        "user_lookup_traditional", 
        "Agent should fetch and present user information from JSONPlaceholder API"
    )
    
    test.add_turn(
        "Tell me about user 1",
        AgentAssertion.contains("user"),
        AgentAssertion.any_of(
            AgentAssertion.contains("name"),
            AgentAssertion.contains("email"),
            AgentAssertion.contains("city")
        ),
        AgentAssertion.min_length(50)
    )
    
    result = test.execute(api_agent)
    assert result.passed, f"User lookup test failed: {result.errors}"


def test_user_information_semantic(api_agent):
    """Semantic test for user information understanding"""
    test = SemanticTest(
        "user_lookup_semantic",
        "Agent should provide meaningful user information from API"
    )
    
    test.add_scenario(
        user_input="Tell me about user 1",
        criteria=[
            "Response includes the user's personal details like name and contact information",
            "Response is conversational and friendly, not just raw data",
            "Information appears to come from an external data source"
        ]
    )
    
    results = test.execute_sync(api_agent)
    assert all(r.passed for r in results), f"Semantic user lookup test failed: {[r.errors for r in results if not r.passed]}"


def test_user_posts_traditional(api_agent):
    """Traditional test for user posts retrieval"""
    test = ConversationTest(
        "posts_traditional",
        "Agent should retrieve and present user posts"
    )
    
    test.add_turn(
        "What has user 2 been posting about?",
        AgentAssertion.contains("post"),
        AgentAssertion.any_of(
            AgentAssertion.contains("title"),
            AgentAssertion.contains("wrote"),
            AgentAssertion.contains("published")
        ),
        AgentAssertion.min_length(100)
    )
    
    result = test.execute(api_agent)
    assert result.passed, f"User posts test failed: {result.errors}"


def test_user_posts_semantic(api_agent):
    """Semantic test for user posts understanding"""
    test = SemanticTest(
        "posts_semantic",
        "Agent should understand and summarize user's posting activity"
    )
    
    test.add_scenario(
        user_input="What has user 2 been writing about lately?",
        criteria=[
            "Response summarizes the themes or topics of the user's posts",
            "Response demonstrates understanding of post content, not just listing titles",
            "Response is helpful for someone wanting to understand the user's interests"
        ]
    )
    
    results = test.execute_sync(api_agent)
    assert all(result.passed for result in results), f"Semantic posts test failed: {[r.errors for r in results if not r.passed]}"


def test_weather_information_traditional(api_agent):
    """Traditional test for weather information retrieval"""
    test = ConversationTest(
        "weather_traditional",
        "Agent should provide weather information for cities"
    )
    
    test.add_turn(
        "What's the weather like in New York?",
        AgentAssertion.contains("New York"),
        AgentAssertion.any_of(
            AgentAssertion.contains("temperature"),
            AgentAssertion.contains("°"),
            AgentAssertion.contains("humidity")
        ),
        AgentAssertion.min_length(30)
    )
    
    result = test.execute(api_agent)
    assert result.passed, f"Weather test failed: {result.errors}"


def test_weather_semantic_usefulness(api_agent):
    """Semantic test for weather information usefulness"""
    test = SemanticTest(
        "weather_semantic",
        "Weather responses should be actionable and contextual"
    )
    
    test.add_scenario(
        user_input="What's the weather like in New York today?",
        criteria=[
            "Response provides current weather conditions that would help someone plan their day",
            "Response includes specific temperature and weather description",
            "Response is more helpful than just stating raw numbers"
        ]
    )
    
    results = test.execute_sync(api_agent)
    assert all(result.passed for result in results), f"Semantic weather test failed: {[r.errors for r in results if not r.passed]}"


def test_http_testing_traditional(api_agent):
    """Traditional test for HTTP request testing functionality"""
    test = ConversationTest(
        "http_traditional",
        "Agent should be able to test HTTP requests"
    )
    
    test.add_turn(
        "Test an HTTP request for me",
        AgentAssertion.any_of(
            AgentAssertion.contains("HTTP"),
            AgentAssertion.contains("request"),
            AgentAssertion.contains("status"),
            AgentAssertion.contains("successful")
        )
    )
    
    result = test.execute(api_agent)
    assert result.passed, f"HTTP test failed: {result.errors}"


def test_http_testing_semantic(api_agent):
    """Semantic test for HTTP testing explanation"""
    test = SemanticTest(
        "http_semantic",
        "Agent should explain technical processes in user-friendly ways"
    )
    
    test.add_scenario(
        user_input="Can you test if the API is working properly?",
        criteria=[
            "Response explains what the API test involves in understandable terms",
            "Response reports results in a way that non-technical users can understand",
            "Response demonstrates whether the system is functioning correctly"
        ]
    )
    
    results = test.execute_sync(api_agent)
    assert all(r.passed for r in results), f"Semantic HTTP test failed: {[r.errors for r in results if not r.passed]}"


def test_multi_turn_conversation_traditional(api_agent):
    """Traditional test for multi-turn conversation with different API calls"""
    test = ConversationTest(
        "multi_turn_traditional",
        "Agent should handle multiple different API requests in sequence"
    )
    
    # Turn 1: User lookup
    test.add_turn(
        "Tell me about user 3",
        AgentAssertion.contains("user"),
        AgentAssertion.min_length(30)
    )
    
    # Turn 2: Weather check
    test.add_turn(
        "Now tell me the weather in London",
        AgentAssertion.contains("London"),
        AgentAssertion.any_of(
            AgentAssertion.contains("weather"),
            AgentAssertion.contains("temperature")
        )
    )
    
    result = test.execute(api_agent)
    assert result.passed, f"Multi-turn test failed: {result.errors}"


def test_multi_turn_semantic_context(api_agent):
    """Semantic test for conversational context maintenance"""
    test = SemanticTest(
        "multi_turn_semantic",
        "Agent should maintain semantic context across multiple requests"
    )
    
    test.add_scenario(
        user_input="Tell me about user 4 and then what they've been writing about",
        criteria=[
            "Response provides user information and connects it to their posting activity",
            "Response maintains conversational flow throughout the response",
            "Response demonstrates understanding of the compound request"
        ]
    )
    
    results = test.execute_sync(api_agent)
    assert all(r.passed for r in results), f"Semantic context test failed: {[r.errors for r in results if not r.passed]}"


def test_error_handling_traditional(api_agent):
    """Traditional test for error handling with invalid inputs"""
    test = ConversationTest(
        "error_traditional",
        "Agent should handle invalid user IDs gracefully"
    )
    
    test.add_turn(
        "Tell me about user 999",  # Non-existent user
        AgentAssertion.excludes("error"),  # Should not expose raw errors
        AgentAssertion.min_length(20)  # Should still provide a response
    )
    
    result = test.execute(api_agent)
    assert result.passed, f"Error handling test failed: {result.errors}"


def test_error_handling_semantic(api_agent):
    """Semantic test for error handling quality"""
    test = SemanticTest(
        "error_semantic",
        "Agent should handle errors in a semantically appropriate way"
    )
    
    test.add_scenario(
        user_input="Tell me about user 999",
        criteria=[
            "Response acknowledges the limitation without being technical or exposing errors",
            "Response offers alternative help or suggestions",
            "Response maintains a helpful and professional tone despite the error"
        ]
    )
    
    results = test.execute_sync(api_agent)
    assert all(r.passed for r in results), f"Semantic error handling test failed: {[r.errors for r in results if not r.passed]}"


def test_weather_comparison_semantic(api_agent):
    """Semantic test for weather comparison understanding"""
    test = SemanticTest(
        "weather_comparison",
        "Agent should semantically understand and compare weather between cities"
    )
    
    test.add_scenario(
        user_input="Compare the weather in Tokyo and London for me",
        criteria=[
            "Response compares weather conditions between both cities meaningfully",
            "Response highlights significant differences or similarities",
            "Response helps the user understand which location has better/different weather"
        ]
    )
    
    results = test.execute_sync(api_agent)
    assert all(r.passed for r in results), f"Semantic weather comparison test failed: {[r.errors for r in results if not r.passed]}"


def test_data_interpretation_traditional(api_agent):
    """Traditional test for data interpretation"""
    test = ConversationTest(
        "data_interpretation_traditional",
        "Agent should present API data clearly"
    )
    
    test.add_turn(
        "What can you tell me about user 4's online activity?",
        AgentAssertion.contains("user"),
        AgentAssertion.excludes("null"),  # Should not show raw null values
        AgentAssertion.excludes("undefined"),
        AgentAssertion.min_length(40)
    )
    
    result = test.execute(api_agent)
    assert result.passed, f"Data interpretation test failed: {result.errors}"


def test_data_interpretation_semantic(api_agent):
    """Semantic test for meaningful data interpretation"""
    test = SemanticTest(
        "data_interpretation_semantic",
        "Agent should interpret and contextualize API data meaningfully"
    )
    
    test.add_scenario(
        user_input="What can you tell me about user 4's online activity?",
        criteria=[
            "Response synthesizes information from multiple data points about the user",
            "Response provides insights beyond just listing facts",
            "Response demonstrates understanding of what makes someone's online activity interesting or notable"
        ]
    )
    
    results = test.execute_sync(api_agent)
    assert all(r.passed for r in results), f"Semantic data interpretation test failed: {[r.errors for r in results if not r.passed]}"


def test_conversational_helpfulness_traditional(api_agent):
    """Traditional test for general helpfulness"""
    test = ConversationTest(
        "helpfulness_traditional",
        "Agent should provide helpful responses"
    )
    
    test.add_turn(
        "Hello! Can you help me with some information?",
        AgentAssertion.any_of(
            AgentAssertion.contains("help"),
            AgentAssertion.contains("assist"),
            AgentAssertion.contains("information")
        ),
        AgentAssertion.max_length(200)
    )
    
    result = test.execute(api_agent)
    assert result.passed, f"Helpfulness test failed: {result.errors}"


def test_conversational_helpfulness_semantic(api_agent):
    """Semantic test for contextual helpfulness"""
    test = SemanticTest(
        "helpfulness_semantic",
        "Agent should provide contextually appropriate help"
    )
    
    test.add_scenario(
        user_input="I'm looking for someone who works in tech - can you help me find them?",
        criteria=[
            "Response understands the user is looking for people with tech backgrounds",
            "Response attempts to help within the constraints of available data",
            "Response suggests practical ways to find the information the user needs"
        ]
    )
    
    results = test.execute_sync(api_agent)
    assert all(r.passed for r in results), f"Semantic helpfulness test failed: {[r.errors for r in results if not r.passed]}"


def test_api_integration_showcase(api_agent):
    """Test showcasing real API integration capabilities"""
    test = ConversationTest(
        "api_integration",
        "Agent should demonstrate its real API capabilities"
    )
    
    test.add_turn(
        "What kind of information can you look up for me?",
        AgentAssertion.any_of(
            AgentAssertion.contains("user"),
            AgentAssertion.contains("weather"),
            AgentAssertion.contains("post"),
            AgentAssertion.contains("HTTP")
        ),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.min_length(60),
        AgentAssertion.excludes("I don't know")
    )
    
    result = test.execute(api_agent)
    assert result.passed, f"API integration test failed: {result.errors}"