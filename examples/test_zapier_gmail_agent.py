"""
Tests for Zapier Gmail Agent - Comprehensive Testing Example

This test suite demonstrates the full range of testLLM testing capabilities:
- Semantic assertions for meaning and context
- Traditional assertions for specific content
- Combined assertions for comprehensive validation
- Multi-turn conversations
- Error handling scenarios

Run with: pytest test_zapier_gmail_agent.py -v

Note: Set ZAPIER_MCP_URL and TEST_EMAIL environment variables for actual email testing
"""

import pytest
import os
from testllm import LocalAgent, ConversationTest, AgentAssertion
from zapier_gmail_agent import zapier_gmail_agent


@pytest.fixture
def gmail_agent():
    """Fixture for the Zapier Gmail agent"""
    return LocalAgent(model=zapier_gmail_agent)


def test_email_sending_comprehensive(gmail_agent):
    """Comprehensive test combining semantic and traditional assertions"""
    test = ConversationTest(
        "email_sending_comprehensive",
        "Agent should send emails with proper confirmation and details"
    )
    
    test_email = os.getenv('TEST_EMAIL', 'test@example.com')
    
    test.add_turn(
        f"Send an email to {test_email} with subject 'testLLM Framework Test' and tell them this is a test of our email integration",
        # Semantic assertions for understanding and context
        AgentAssertion.semantic("Response indicates that an email sending attempt was made"),
        AgentAssertion.semantic("Response communicates the success or failure clearly"),
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
    assert result.passed, f"Comprehensive email test failed: {result.errors}"


def test_email_validation_mixed_assertions(gmail_agent):
    """Test email validation using multiple assertion types"""
    test = ConversationTest(
        "email_validation_mixed",
        "Agent should validate email addresses using multiple criteria"
    )
    
    test.add_turn(
        "Is 'invalid-email-format' a valid email address?",
        # Semantic understanding
        AgentAssertion.semantic("Response clearly identifies the email format as invalid"),
        # Traditional content checks
        AgentAssertion.any_of(
            AgentAssertion.contains("invalid"),
            AgentAssertion.contains("not valid"),
            AgentAssertion.contains("incorrect")
        ),
        # Should not contain positive validation terms
        AgentAssertion.excludes("valid email"),
        AgentAssertion.excludes("correct format")
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Email validation test failed: {result.errors}"


def test_valid_email_with_all_assertions(gmail_agent):
    """Test valid email recognition with comprehensive assertions"""
    test = ConversationTest(
        "valid_email_comprehensive",
        "Agent should recognize valid emails with proper confirmation"
    )
    
    test.add_turn(
        "Check if 'user@example.com' is a valid email address",
        # Semantic validation
        AgentAssertion.semantic("Response confirms the email address is valid"),
        # Content validation
        AgentAssertion.contains("user@example.com"),
        AgentAssertion.any_of(
            AgentAssertion.contains("valid"),
            AgentAssertion.contains("correct"),
            AgentAssertion.contains("proper")
        ),
        # Quality checks
        AgentAssertion.max_length(200),
        AgentAssertion.sentiment("positive")
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Valid email test failed: {result.errors}"


def test_multi_turn_email_conversation(gmail_agent):
    """Test multi-turn conversation with various assertion types"""
    test = ConversationTest(
        "multi_turn_email",
        "Agent should handle multi-turn email conversations"
    )
    
    # Turn 1: Initial email request
    test.add_turn(
        "I need to send an email to my colleague",
        AgentAssertion.semantic("Response asks for more details about the email"),
        AgentAssertion.any_of(
            AgentAssertion.contains("what"),
            AgentAssertion.contains("details"),
            AgentAssertion.contains("tell me")
        ),
        AgentAssertion.sentiment("helpful")
    )
    
    # Turn 2: Provide details
    test.add_turn(
        "Send it to john@company.com with subject 'Meeting Tomorrow' and ask if he can attend the 3pm meeting",
        AgentAssertion.semantic("Response processes the email details and attempts to send"),
        AgentAssertion.contains("john@company.com"),  
        AgentAssertion.contains("Meeting Tomorrow"),
        AgentAssertion.min_length(40)
    )
    
    # Turn 3: Follow-up question
    test.add_turn(
        "Was the email sent successfully?",
        AgentAssertion.semantic("Response provides status information about the email"),
        AgentAssertion.any_of(
            AgentAssertion.contains("sent"),
            AgentAssertion.contains("delivered"),
            AgentAssertion.contains("success")
        )
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Multi-turn conversation test failed: {result.errors}"


def test_error_handling_comprehensive(gmail_agent):
    """Test error handling with multiple assertion types"""
    test = ConversationTest(
        "error_handling_comprehensive",
        "Agent should handle errors gracefully across multiple scenarios"
    )
    
    test.add_turn(
        "Send an email to 'not-an-email' with no subject",
        # Semantic error handling
        AgentAssertion.semantic("Response identifies problems with the email request"),
        AgentAssertion.semantic("Response provides helpful guidance"),
        # Traditional error detection
        AgentAssertion.any_of(
            AgentAssertion.contains("invalid"),
            AgentAssertion.contains("error"),
            AgentAssertion.contains("problem")
        ),
        # Should not crash or give unhelpful responses
        AgentAssertion.excludes("Exception"),
        AgentAssertion.excludes("traceback"),
        AgentAssertion.min_length(30)
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Error handling test failed: {result.errors}"


def test_professional_email_composition(gmail_agent):
    """Test professional email composition with quality assertions"""
    test = ConversationTest(
        "professional_composition",
        "Agent should help compose professional emails"
    )
    
    test.add_turn(
        "Help me send a professional follow-up email to client@business.com about our proposal discussion",
        # Semantic understanding of professional context
        AgentAssertion.semantic("Response understands the professional context and follow-up nature"),
        AgentAssertion.semantic("Response generates or suggests appropriate professional content"),
        # Content requirements
        AgentAssertion.contains("client@business.com"),
        AgentAssertion.any_of(
            AgentAssertion.contains("proposal"),
            AgentAssertion.contains("discussion"),
            AgentAssertion.contains("follow-up")
        ),
        # Quality and tone
        AgentAssertion.sentiment("professional"),
        AgentAssertion.min_length(100),
        # Should avoid overly casual language
        AgentAssertion.excludes("hey"),
        AgentAssertion.excludes("sup")
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Professional composition test failed: {result.errors}"


def test_cc_functionality_assertions(gmail_agent):
    """Test CC functionality with multiple assertion types"""
    test = ConversationTest(
        "cc_functionality",
        "Agent should handle CC recipients properly"
    )
    
    test.add_turn(
        "Send an email to manager@company.com and CC assistant@company.com about the project update",
        # Semantic understanding
        AgentAssertion.semantic("Response demonstrates understanding of CC functionality"),
        AgentAssertion.semantic("Response handles multiple recipients appropriately"),
        # Content validation
        AgentAssertion.all_of(
            AgentAssertion.contains("manager@company.com"),
            AgentAssertion.contains("assistant@company.com"),
            AgentAssertion.any_of(
                AgentAssertion.contains("CC"),
                AgentAssertion.contains("cc"),
                AgentAssertion.contains("copy")
            )
        ),
        # Quality checks
        AgentAssertion.min_length(60)
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"CC functionality test failed: {result.errors}"


def test_integration_capabilities_showcase(gmail_agent):
    """Test showcasing integration capabilities with varied assertions"""
    test = ConversationTest(
        "integration_showcase",
        "Agent should demonstrate its integration capabilities"
    )
    
    test.add_turn(
        "What can you do with emails? Show me your capabilities",
        # Semantic capability demonstration
        AgentAssertion.semantic("Response explains email capabilities clearly"),
        AgentAssertion.semantic("Response mentions integration with external services"),
        # Feature coverage
        AgentAssertion.any_of(
            AgentAssertion.contains("send"),
            AgentAssertion.contains("email"),
            AgentAssertion.contains("Zapier")
        ),
        # Helpful and informative
        AgentAssertion.sentiment("informative"),
        AgentAssertion.min_length(80),
        AgentAssertion.max_length(500),  # Not too verbose
        # Professional presentation
        AgentAssertion.excludes("I don't know"),
        AgentAssertion.excludes("I can't")
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Integration showcase test failed: {result.errors}"


def test_edge_cases_comprehensive(gmail_agent):
    """Test edge cases with comprehensive assertion coverage"""
    test = ConversationTest(
        "edge_cases_comprehensive", 
        "Agent should handle edge cases gracefully"
    )
    
    test.add_turn(
        "Send an email with a very long subject: " + "A" * 200,
        # Semantic handling of edge case
        AgentAssertion.semantic("Response handles the unusually long subject appropriately"),
        # Should not crash
        AgentAssertion.min_length(20),
        # Should either process or warn about length
        AgentAssertion.any_of(
            AgentAssertion.contains("long"),
            AgentAssertion.contains("subject"),
            AgentAssertion.contains("sent")
        ),
        # Quality response
        AgentAssertion.sentiment("helpful")
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"Edge cases test failed: {result.errors}"


def test_user_experience_quality(gmail_agent):
    """Test overall user experience quality with mixed assertions"""
    test = ConversationTest(
        "user_experience",
        "Agent should provide excellent user experience"
    )
    
    test.add_turn(
        "I'm new to this - can you help me send my first email?",
        # Semantic user experience
        AgentAssertion.semantic("Response is welcoming and supportive for a new user"),
        AgentAssertion.semantic("Response provides clear guidance without being overwhelming"),
        # Helpful content
        AgentAssertion.any_of(
            AgentAssertion.contains("help"),
            AgentAssertion.contains("guide"),
            AgentAssertion.contains("show")
        ),
        # Positive interaction
        AgentAssertion.sentiment("encouraging"),
        AgentAssertion.excludes("complicated"),
        AgentAssertion.excludes("difficult"),
        # Appropriate length
        AgentAssertion.min_length(50),
        AgentAssertion.max_length(300)
    )
    
    result = test.execute(gmail_agent)
    assert result.passed, f"User experience test failed: {result.errors}"