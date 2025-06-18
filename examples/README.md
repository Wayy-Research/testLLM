# testLLM Examples

This directory contains comprehensive examples demonstrating how to use testLLM to test AI agents with both semantic and traditional assertions.

## Overview

The examples showcase two real-world agent implementations and their corresponding test suites:

1. **Real API Agent** - A PydanticAI agent that makes actual HTTP requests to public APIs
2. **Zapier Gmail Agent** - A PydanticAI agent that integrates with Zapier's MCP server for Gmail functionality

Both examples demonstrate testLLM's key capabilities:
- **Semantic assertions** - Test the meaning and quality of responses
- **Traditional assertions** - Test for specific content and patterns
- **Multi-turn conversations** - Test conversational flow and context
- **Error handling** - Test graceful failure scenarios
- **Real-world integration** - Test agents that interact with external services

## Setup

### Prerequisites

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Or install individual packages:

```bash
pip install pydantic-ai fastmcp pytest requests python-dotenv
```

### Environment Variables

Set up your environment variables:

```bash
# Required for both examples
export ANTHROPIC_API_KEY="your-claude-api-key"

# Optional for Zapier example (uses demo URL if not set)
export ZAPIER_MCP_URL="your-zapier-mcp-url"

# Optional for testing actual email delivery
export TEST_EMAIL="your-email@example.com"
```

## Example 1: Real API Agent

### Files
- `real_api_agent.py` - The agent implementation
- `test_real_api_agent.py` - Comprehensive test suite

### What it does
The Real API Agent demonstrates integration with public APIs:
- **JSONPlaceholder API** - Fetches user information and posts
- **HTTPBin API** - Tests HTTP requests and responses
- **Mock weather data** - Simulates weather API responses

### Key Features
- Makes actual HTTP requests to real APIs
- Handles API errors gracefully
- Provides conversational responses with API data
- Demonstrates tool use with external services

### Running the Agent

```bash
cd examples
python real_api_agent.py
```

### Running the Tests

```bash
# Run all tests
pytest test_real_api_agent.py -v

# Run specific test
pytest test_real_api_agent.py::test_user_information_semantic -v

# Run with detailed output
pytest test_real_api_agent.py -v -s
```

### Test Examples Explained

#### Semantic Assertions
```python
AgentAssertion.semantic("Response includes the user's personal details like name and contact information")
AgentAssertion.semantic("Response is conversational and friendly, not just raw data")
```
These test the *meaning* and *quality* of responses rather than specific keywords.

#### Traditional Assertions
```python
AgentAssertion.contains("user")
AgentAssertion.min_length(50)
AgentAssertion.excludes("null")
```
These test for specific content, length, and exclusions.

#### Combined Approach
```python
test.add_turn(
    "Tell me about user 1",
    AgentAssertion.semantic("Response provides meaningful user information"),
    AgentAssertion.contains("user"),
    AgentAssertion.min_length(50)
)
```
Combines semantic understanding with specific content validation.

## Example 2: Zapier Gmail Agent

### Files
- `zapier_gmail_agent.py` - The agent implementation
- `test_zapier_gmail_agent.py` - Comprehensive test suite

### What it does
The Zapier Gmail Agent demonstrates real email integration:
- **Send emails** through Zapier MCP Gmail integration
- **Validate email addresses** with proper formatting
- **List available tools** from Zapier MCP server
- **Handle CC recipients** and complex email scenarios

### Key Features
- Real integration with Zapier's MCP server
- Actual email sending capabilities
- Professional email composition assistance
- Comprehensive error handling

### Running the Agent

```bash
cd examples
python zapier_gmail_agent.py
```

### Running the Tests

```bash
# Run all tests
pytest test_zapier_gmail_agent.py -v

# Run specific test category
pytest test_zapier_gmail_agent.py -k "semantic" -v

# Run with your email for actual delivery testing
TEST_EMAIL="your-email@example.com" pytest test_zapier_gmail_agent.py::test_email_confirmation_semantic -v
```

### Test Examples Explained

#### Email Sending Validation
```python
test.add_turn(
    "Send an email to test@example.com with subject 'Test Email'",
    # Semantic: Understanding of email sending
    AgentAssertion.semantic("Response indicates that an email sending attempt was made"),
    # Traditional: Specific content validation
    AgentAssertion.contains("test@example.com"),
    AgentAssertion.contains("Test Email"),
    # Quality: Response characteristics
    AgentAssertion.min_length(50),
    AgentAssertion.sentiment("positive")
)
```

#### Multi-turn Conversation Testing
```python
# Turn 1: Initial request
test.add_turn(
    "I need to send an email to my colleague",
    AgentAssertion.semantic("Response asks for more details about the email")
)

# Turn 2: Provide details
test.add_turn(
    "Send it to john@company.com with subject 'Meeting Tomorrow'",
    AgentAssertion.semantic("Response processes the email details and attempts to send"),
    AgentAssertion.contains("john@company.com")
)
```

#### Error Handling Testing
```python
test.add_turn(
    "Send an email to 'not-an-email'",
    AgentAssertion.semantic("Response identifies problems with the email request"),
    AgentAssertion.any_of(
        AgentAssertion.contains("invalid"),
        AgentAssertion.contains("error")
    ),
    AgentAssertion.excludes("Exception")  # Should handle errors gracefully
)
```

## Test Assertion Types

### Semantic Assertions
Test the meaning and quality of responses:
- `AgentAssertion.semantic("description")` - Tests conceptual understanding
- Best for testing conversation quality, context understanding, and appropriateness

### Traditional Assertions
Test specific content and patterns:
- `AgentAssertion.contains("text")` - Must contain specific text
- `AgentAssertion.excludes("text")` - Must not contain specific text
- `AgentAssertion.min_length(n)` - Minimum character length
- `AgentAssertion.max_length(n)` - Maximum character length
- `AgentAssertion.sentiment("positive")` - Overall sentiment

### Logical Assertions
Combine multiple conditions:
- `AgentAssertion.all_of(assertion1, assertion2)` - All must pass
- `AgentAssertion.any_of(assertion1, assertion2)` - At least one must pass

## Running All Tests

```bash
# Run all example tests
pytest examples/ -v

# Run with coverage
pytest examples/ --cov=examples

# Run specific test patterns
pytest examples/ -k "semantic" -v
pytest examples/ -k "email" -v
pytest examples/ -k "error" -v
```

## Real-World Testing

### Testing Actual Email Delivery
To test actual email delivery, set your email address and run:

```bash
export TEST_EMAIL="your-email@example.com"
pytest test_zapier_gmail_agent.py::test_email_confirmation_semantic -v
```

The test will attempt to send a real email to your address, allowing you to verify:
1. The agent correctly processes the email request
2. The email is actually delivered
3. The content matches what was requested

### Testing with Real APIs
The Real API Agent makes actual HTTP requests to:
- `jsonplaceholder.typicode.com` - For user and post data
- `httpbin.org` - For HTTP testing

These tests demonstrate how testLLM can validate agents that interact with real external services.

## Best Practices

1. **Combine Semantic and Traditional** - Use both types of assertions for comprehensive validation
2. **Test Error Scenarios** - Always test how agents handle invalid inputs
3. **Use Multi-turn Tests** - Test conversational context and memory
4. **Validate External Integrations** - Test with real APIs when possible
5. **Test User Experience** - Use semantic assertions to validate helpfulness and clarity

## Troubleshooting

### Common Issues

**Import Errors**
```bash
pip install pydantic-ai fastmcp
```

**API Key Issues**
```bash
export ANTHROPIC_API_KEY="your-key-here"
```

**Zapier MCP Connection**
- Ensure your ZAPIER_MCP_URL is correct
- Check network connectivity
- Verify Zapier MCP server is running

**Test Failures**
- Check that APIs are accessible
- Verify environment variables are set
- Review test assertions for accuracy

For more information, see the main testLLM documentation.