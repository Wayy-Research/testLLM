"""
Tests for Zapier Gmail Agent - Comprehensive Testing Example

This test suite demonstrates the full range of testLLM testing capabilities:
- Semantic testing for meaning and context using LLM evaluation
- Traditional assertions for specific content
- Combined approaches for comprehensive validation
- Multi-turn conversations
- Error handling scenarios

Run with: pytest test_zapier_gmail_agent.py -v

Note: Set ZAPIER_MCP_URL and TEST_EMAIL environment variables for actual email testing
"""

import pytest
import os
from testllm import LocalAgent, ConversationTest, AgentAssertion, SemanticTest
from zapier_gmail_agent import zapier_gmail_agent


@pytest.fixture
def gmail_agent():
    """Fixture for the Zapier Gmail agent"""
    return LocalAgent(model=zapier_gmail_agent)


def test_email_sending_traditional_assertions(gmail_agent):
    """Traditional assertion-based test for email sending"""
    test = ConversationTest(
        "email_sending_traditional",
        "Agent should send emails with proper confirmation and details"
    )
    
    test_email = os.getenv('TEST_EMAIL', 'test@example.com')
    
    test.add_turn(
        f"Send an email to {test_email} with subject 'testLLM Framework Test' and tell them this is a test of our email integration",
        # Traditional assertions for specific content
        AgentAssertion.contains(test_email),
        AgentAssertion.contains("testLLM Framework Test"),
        # Quality assertions
        AgentAssertion.min_length(50),
        AgentAssertion.sentiment("positive"),
        # Exclusion assertions for error handling
        AgentAssertion.excludes("error occurred"),
        AgentAssertion.excludes("failed to")
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Traditional email test failed: {result.errors}"


def test_email_sending_semantic_evaluation(gmail_agent):
    """Semantic test for email sending understanding using LLM evaluation"""
    test_email = os.getenv('TEST_EMAIL', 'test@example.com')
    
    test = SemanticTest(
        "email_sending_semantic",
        "Agent should understand email requests and handle sending appropriately"
    )
    
    test.add_scenario(
        user_input=f"Send an email to {test_email} with subject 'testLLM Framework Test' and tell them this is a test of our email integration",
        criteria=[
            "Response indicates that an email sending attempt was made",
            "Response communicates the success or failure clearly", 
            "Response includes confirmation of the email details (recipient, subject)",
            "Response is helpful and professional in tone"
        ]
    )
    
    results = test.execute_sync(gmail_agent)
    assert all(r.passed for r in results), f"Semantic email test failed: {[r.errors for r in results if not r.passed]}"


def test_email_validation_traditional(gmail_agent):
    """Test email validation using traditional assertions"""
    test = ConversationTest(
        "email_validation_traditional",
        "Agent should validate email addresses correctly"
    )
    
    test.add_turn(
        "Is 'invalid-email-format' a valid email address?",
        # Traditional content checks
        AgentAssertion.any_of(
            AgentAssertion.contains("invalid"),
            AgentAssertion.contains("not valid"),
            AgentAssertion.contains("incorrect")
        ),
        # Should not contain positive validation terms
        AgentAssertion.excludes("valid email"),
        AgentAssertion.excludes("correct format"),
        AgentAssertion.min_length(20)
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Email validation test failed: {result.errors}"


def test_email_validation_semantic(gmail_agent):
    """Semantic test for email validation understanding"""
    test = SemanticTest(
        "email_validation_semantic",
        "Agent should understand and explain email validation properly"
    )
    
    test.add_scenario(
        user_input="Is 'invalid-email-format' a valid email address?",
        criteria=[
            "Response clearly identifies the email format as invalid",
            "Response explains what makes an email address valid or invalid",
            "Response is helpful for someone learning about email formats"
        ]
    )
    
    results = test.execute_sync(gmail_agent)
    assert all(r.passed for r in results), f"Semantic email validation test failed: {[r.errors for r in results if not r.passed]}"


def test_valid_email_recognition(gmail_agent):
    """Test recognition of valid email addresses"""
    test = ConversationTest(
        "valid_email_recognition",
        "Agent should recognize valid emails properly"
    )
    
    test.add_turn(
        "Check if 'user@example.com' is a valid email address",
        AgentAssertion.contains("user@example.com"),
        AgentAssertion.any_of(
            AgentAssertion.contains("valid"),
            AgentAssertion.contains("correct"),
            AgentAssertion.contains("proper")
        ),
        AgentAssertion.max_length(200),
        AgentAssertion.sentiment("positive")
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Valid email test failed: {result.errors}"


def test_multi_turn_email_conversation(gmail_agent):
    """Test multi-turn conversation with email functionality"""
    test = ConversationTest(
        "multi_turn_email",
        "Agent should handle multi-turn email conversations"
    )
    
    # Turn 1: Initial email request
    test.add_turn(
        "I need to send an email to my colleague",
        AgentAssertion.any_of(
            AgentAssertion.contains("what"),
            AgentAssertion.contains("details"),
            AgentAssertion.contains("tell me"),
            AgentAssertion.contains("information")
        ),
        AgentAssertion.sentiment("positive")
    )
    
    # Turn 2: Provide details
    test.add_turn(
        "Send it to john@company.com with subject 'Meeting Tomorrow' and ask if he can attend the 3pm meeting",
        AgentAssertion.contains("john@company.com"),  
        AgentAssertion.contains("Meeting Tomorrow"),
        AgentAssertion.min_length(40)
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Multi-turn conversation test failed: {result.errors}"


def test_multi_turn_semantic_context(gmail_agent):
    """Semantic test for multi-turn conversation context"""
    test = SemanticTest(
        "multi_turn_semantic",
        "Agent should maintain context across conversation turns"
    )
    
    # Note: SemanticTest handles multi-turn differently - this is a simplified version
    test.add_scenario(
        user_input="I need help sending an email to my colleague about our project meeting",
        criteria=[
            "Response acknowledges the email request",
            "Response asks for necessary details like recipient and content",
            "Response is helpful and conversational"
        ]
    )
    
    results = test.execute_sync(gmail_agent)
    assert all(r.passed for r in results), f"Semantic multi-turn test failed: {[r.errors for r in results if not r.passed]}"


def test_error_handling_comprehensive(gmail_agent):
    """Test error handling with multiple assertion types"""
    test = ConversationTest(
        "error_handling_comprehensive",
        "Agent should handle errors gracefully"
    )
    
    test.add_turn(
        "Send an email to 'not-an-email' with no subject",
        # Traditional error detection
        AgentAssertion.any_of(
            AgentAssertion.contains("invalid"),
            AgentAssertion.contains("error"),
            AgentAssertion.contains("problem"),
            AgentAssertion.contains("format")
        ),
        # Should not crash or give unhelpful responses
        AgentAssertion.excludes("Exception"),
        AgentAssertion.excludes("traceback"),
        AgentAssertion.min_length(30)
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Error handling test failed: {result.errors}"


def test_error_handling_semantic(gmail_agent):
    """Semantic test for error handling quality"""
    test = SemanticTest(
        "error_handling_semantic",
        "Agent should handle errors with helpful explanations"
    )
    
    test.add_scenario(
        user_input="Send an email to 'not-an-email' with subject 'Test'",
        criteria=[
            "Response identifies the email address problem without being technical",
            "Response provides helpful guidance on what constitutes a valid email",
            "Response maintains a helpful tone despite the error"
        ]
    )
    
    results = test.execute_sync(gmail_agent)
    assert all(r.passed for r in results), f"Semantic error handling test failed: {[r.errors for r in results if not r.passed]}"


def test_professional_email_composition(gmail_agent):
    """Test professional email composition capabilities"""
    test = ConversationTest(
        "professional_composition",
        "Agent should help with professional email composition"
    )
    
    test.add_turn(
        "Help me send a professional follow-up email to client@business.com about our proposal discussion",
        AgentAssertion.contains("client@business.com"),
        AgentAssertion.any_of(
            AgentAssertion.contains("proposal"),
            AgentAssertion.contains("discussion"),
            AgentAssertion.contains("follow-up")
        ),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.min_length(100),
        # Should avoid overly casual language
        AgentAssertion.excludes("hey"),
        AgentAssertion.excludes("sup")
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Professional composition test failed: {result.errors}"


def test_professional_email_semantic(gmail_agent):
    """Semantic test for professional email understanding"""
    test = SemanticTest(
        "professional_email_semantic",
        "Agent should understand professional email context and tone"
    )
    
    test.add_scenario(
        user_input="Help me send a professional follow-up email to client@business.com about our proposal discussion",
        criteria=[
            "Response understands the professional context and follow-up nature",
            "Response generates or suggests appropriate professional content",
            "Response demonstrates understanding of business communication tone"
        ]
    )
    
    results = test.execute_sync(gmail_agent)
    assert all(r.passed for r in results), f"Semantic professional email test failed: {[r.errors for r in results if not r.passed]}"


def test_cc_functionality(gmail_agent):
    """Test CC functionality with traditional assertions"""
    test = ConversationTest(
        "cc_functionality",
        "Agent should handle CC recipients properly"
    )
    
    test.add_turn(
        "Send an email to manager@company.com and CC assistant@company.com about the project update",
        AgentAssertion.all_of(
            AgentAssertion.contains("manager@company.com"),
            AgentAssertion.contains("assistant@company.com"),
            AgentAssertion.any_of(
                AgentAssertion.contains("CC"),
                AgentAssertion.contains("cc"),
                AgentAssertion.contains("copy")
            )
        ),
        AgentAssertion.min_length(60)
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"CC functionality test failed: {result.errors}"


def test_integration_capabilities(gmail_agent):
    """Test showcasing integration capabilities"""
    test = ConversationTest(
        "integration_showcase",
        "Agent should demonstrate its integration capabilities"
    )
    
    test.add_turn(
        "What can you do with emails? Show me your capabilities",
        AgentAssertion.any_of(
            AgentAssertion.contains("send"),
            AgentAssertion.contains("email"),
            AgentAssertion.contains("Zapier")
        ),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.min_length(80),
        AgentAssertion.max_length(500),  # Not too verbose
        AgentAssertion.excludes("I don't know"),
        AgentAssertion.excludes("I can't")
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Integration showcase test failed: {result.errors}"


def test_user_experience_quality(gmail_agent):
    """Test overall user experience quality"""
    test = ConversationTest(
        "user_experience",
        "Agent should provide excellent user experience"
    )
    
    test.add_turn(
        "I'm new to this - can you help me send my first email?",
        AgentAssertion.any_of(
            AgentAssertion.contains("help"),
            AgentAssertion.contains("guide"),
            AgentAssertion.contains("show")
        ),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.excludes("complicated"),
        AgentAssertion.excludes("difficult"),
        AgentAssertion.min_length(50),
        AgentAssertion.max_length(300)
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"User experience test failed: {result.errors}"


def test_user_experience_semantic(gmail_agent):
    """Semantic test for user experience quality"""
    test = SemanticTest(
        "user_experience_semantic",
        "Agent should provide welcoming experience for new users"
    )
    
    test.add_scenario(
        user_input="I'm new to this - can you help me send my first email?",
        criteria=[
            "Response is welcoming and supportive for a new user",
            "Response provides clear guidance without being overwhelming",
            "Response encourages the user and builds confidence"
        ]
    )
    
    results = test.execute_sync(gmail_agent)
    assert all(r.passed for r in results), f"Semantic user experience test failed: {[r.errors for r in results if not r.passed]}"