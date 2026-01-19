# testllm

**The first testing framework designed specifically for LLM-based agents.**

testllm uses fast, accurate LLM evaluators to test your AI agents semantically, not with brittle string matching. Write natural language test criteria that evaluate meaning, intent, and behavior rather than exact outputs.

## Quick Start

### Installation

```bash
pip install testkitLLM
```

### Setup (1 minute, free)

**Option 1: Interactive Setup (Recommended)**
```bash
python -m testllm.setup
```
This opens Google AI Studio where you can get a free API key (no credit card required).

**Option 2: Manual Setup**

Add an API key to your environment or `.env` file:
```bash
# Google Gemini (RECOMMENDED) - Free tier, fast
GOOGLE_API_KEY=your_google_api_key_here

# Anthropic Claude (OPTIONAL) - More thorough
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Mistral (OPTIONAL)
MISTRAL_API_KEY=your_mistral_api_key_here
```

### 30-Second Example

**Write a semantic test** (`test_my_agent.py`):
```python
import pytest
from testllm import LocalAgent, semantic_test

@pytest.fixture
def my_agent():
    class WeatherAgent:
        def __call__(self, prompt):
            if "weather" in prompt.lower():
                return "I'll check the current weather conditions for you."
            return "I understand your request. How can I help?"

    return LocalAgent(model=WeatherAgent())

def test_weather_query_response(my_agent):
    """Test weather query handling"""
    test = semantic_test("weather_test", "Weather query handling")

    test.add_scenario(
        user_input="What's the weather in Seattle?",
        criteria=[
            "Response should acknowledge the weather question",
            "Response should mention checking or retrieving weather data",
            "Response should be helpful and professional"
        ]
    )

    results = test.execute_sync(my_agent)
    assert all(r.passed for r in results), "Weather test failed"
```

**Run it**:
```bash
pytest test_my_agent.py -v
```

That's it! testLLM evaluates your agent's response semantically, understanding meaning rather than requiring exact text matches.

## Core Features

### 1. Semantic Testing (Single Turn)

Test individual agent responses with natural language criteria:

```python
from testllm import semantic_test

def test_customer_support(agent):
    """Test customer support responses"""
    test = semantic_test("support_test", "Customer support testing")

    test.add_scenario(
        user_input="I need help with my account",
        criteria=[
            "Response should offer assistance",
            "Response should be empathetic and professional",
            "Response should not dismiss the request"
        ]
    )

    results = test.execute_sync(agent)
    assert all(r.passed for r in results)
```

### 2. Conversation Flow Testing

Test multi-step conversations with context retention:

```python
from testllm import conversation_flow

def test_customer_onboarding(agent):
    """Test customer onboarding workflow"""
    flow = conversation_flow("onboarding", "Customer onboarding process")

    # Step 1: Initial contact
    flow.step(
        "Hello, I'm a new customer",
        criteria=[
            "Response should acknowledge new customer status",
            "Response should begin onboarding process"
        ]
    )

    # Step 2: Information gathering with context retention
    flow.step(
        "My name is Sarah and I need a business account",
        criteria=[
            "Response should acknowledge the name Sarah",
            "Response should understand business account requirement"
        ],
        expect_context_retention=True
    )

    # Step 3: Memory validation
    flow.context_check(
        "What type of account was I requesting?",
        context_criteria=[
            "Response should remember business account request"
        ]
    )

    result = flow.execute_sync(agent)
    assert result.passed
    assert result.context_retention_score >= 0.7
```

### 3. Tool Testing Framework

testllm provides comprehensive tool testing with three complementary approaches:

#### Tool Expectation DSL

Declarative API for specifying tool expectations within conversation flows:

```python
from testllm import conversation_flow, expect_tools

flow = (
    conversation_flow("flight_booking", "Test flight booking")
    .tool_step(
        "Find me flights from SFO to NYC",
        criteria=["Should present flight options"],
        tool_expectations=expect_tools()
            .expect_call("search_flights")
            .with_arguments_containing(origin="SFO", destination="NYC")
            .returning({"flights": [{"id": "F1", "price": 299}]})
    )
)
```

#### Tool Interceptor

Capture and mock tool calls for any agent:

```python
from testllm import ToolInterceptor, InterceptedAgent

interceptor = ToolInterceptor()
interceptor.register_mock("search_flights", {
    "flights": [{"id": "F1", "airline": "United", "price": 299}]
})
interceptor.register_schema("book_flight", {
    "properties": {
        "flight_id": {"type": "string"},
        "passenger_name": {"type": "string"}
    },
    "required": ["flight_id", "passenger_name"]
})

# Wrap your agent
wrapped_agent = InterceptedAgent(my_agent, interceptor)

# After running tests, verify tool usage
assert interceptor.was_called("search_flights")
assert interceptor.call_count("search_flights") == 1
print(interceptor.get_call_sequence())  # ['search_flights', 'book_flight']
```

#### Response Simulation

Test agent behavior under various tool response scenarios:

```python
from testllm import simulate_tool, tool_response_suite, ScenarioType

# Configure multiple scenarios for a tool
flight_search = (
    simulate_tool("search_flights")
    .on_success({"flights": [{"id": "F1", "price": 299}]})
    .on_failure("Flight search service unavailable")
    .on_timeout(30000)
    .on_empty({"flights": [], "message": "No flights found"})
    .on_rate_limited()
)

# Create a test suite for resilience testing
suite = (
    tool_response_suite("resilience_tests", "Test error handling")
    .add_simulator(flight_search)
    .test_all_scenarios(
        "search_flights",
        user_input="Find flights to NYC",
        criteria=["Agent should handle the response gracefully"]
    )
)
```

#### Pre-built Scenarios

Common tool patterns ready to use:

```python
from testllm import SearchScenarios, APIScenarios, CRUDScenarios

# Weather API with success, failure, timeout scenarios
weather_sim = APIScenarios.weather_api()

# Hotel search with various result scenarios
hotel_sim = SearchScenarios.hotel_search()

# Database operations with CRUD scenarios
db_sim = CRUDScenarios.database_operations()
```

### 4. Behavioral Pattern Testing

Pre-built patterns for common agent behaviors:

```python
from testllm import ToolUsagePatterns, BusinessLogicPatterns

def test_agent_patterns(agent):
    """Test using pre-built behavioral patterns"""

    # Test API integration behavior
    api_flow = ToolUsagePatterns.api_integration_pattern(
        "Get current stock price of AAPL",
        "financial"
    )

    # Test business workflow
    auth_flow = BusinessLogicPatterns.user_authentication_flow("premium")

    # Execute patterns
    api_result = api_flow.execute_sync(agent)
    auth_result = auth_flow.execute_sync(agent)

    assert api_result.passed
    assert auth_result.passed
```

### 5. Universal Agent Support

testLLM works with **any** agent:

```python
# Local model
from testllm import LocalAgent
agent = LocalAgent(model=your_local_model)

# API endpoint
from testllm import ApiAgent
agent = ApiAgent(endpoint="https://your-api.com/chat")

# With tool interception
from testllm import InterceptedAgent, ToolInterceptor
interceptor = ToolInterceptor()
agent = InterceptedAgent(your_agent, interceptor)

# Custom implementation
from testllm import AgentUnderTest

class MyAgent(AgentUnderTest):
    def send_message(self, content, context=None):
        return your_custom_logic(content)

    def reset_conversation(self):
        pass
```

## Configuration

### Testing Modes

```python
# Fast mode (default) - optimized for development
flow = conversation_flow("test_id", config_mode="fast")
# Uses: Gemini, ~15-30 seconds per test

# Production mode - balanced reliability and performance
flow = conversation_flow("test_id", config_mode="production")
# Uses: Gemini + Claude validation, ~30-60 seconds per test

# Thorough mode - comprehensive testing
flow = conversation_flow("test_id", config_mode="thorough")
# Uses: Multiple evaluators, multiple iterations, ~45-90 seconds per test
```

### Custom Configuration

```python
test = semantic_test(
    "custom_test",
    evaluator_models=["gemini-2.0-flash"],  # Gemini-only for max speed
    consensus_threshold=0.8
)
```

## pytest Integration

### Run Tests with Detailed Output

```bash
# Show detailed evaluation output
pytest -v -s

# Run specific test files
pytest test_weather.py -v -s

# Run tests matching a pattern
pytest -k "test_greeting" -v -s
```

The `-s` flag shows detailed LLM evaluation output with reasoning and scoring.

## Writing Effective Tests

### Good Semantic Criteria

| Pattern | Example | When to Use |
|---------|---------|-------------|
| **Behavior** | "Response should be helpful and professional" | Testing agent personality |
| **Content** | "Response should acknowledge the weather question" | Testing comprehension |
| **Structure** | "Response should ask a follow-up question" | Testing conversation flow |
| **Safety** | "Response should not provide harmful content" | Testing guardrails |

### Performance Tips

```python
# FAST: Use fewer, focused criteria
test.add_scenario(
    "Hello",
    ["Response should be friendly"]  # 1 criterion = faster
)

# SLOW: Too many criteria
test.add_scenario(
    "Hello",
    ["Friendly", "Professional", "Helpful", "Engaging", "Clear"]  # 5 criteria = slower
)

# FAST: Use fast mode for development
flow = conversation_flow("test", config_mode="fast")

# BALANCED: Use production mode for CI/CD
flow = conversation_flow("test", config_mode="production")
```

## Requirements

- Python 3.9+
- pytest 7.0+
- At least one API key (Google, Anthropic, or Mistral)

## Support

- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: For detailed guides and examples

## Development & Release Process

### Making a Release

1. **Bump version and create release:**
   ```bash
   # For patch release (0.1.0 -> 0.1.1)
   python scripts/bump_version.py patch

   # For minor release (0.1.0 -> 0.2.0)
   python scripts/bump_version.py minor

   # For major release (0.1.0 -> 1.0.0)
   python scripts/bump_version.py major
   ```

2. **Push changes and tag:**
   ```bash
   git push origin main
   git push origin v0.2.0  # Replace with your version
   ```

3. **Create GitHub Release:**
   - Go to https://github.com/Wayy-Research/testLLM/releases
   - Click "Create a new release"
   - Select your tag
   - Add release notes
   - Publish release

This will automatically:
- Run tests across Python 3.9-3.12
- Deploy to PyPI on GitHub releases

---

**Ready to test your LLM agents properly?**

```bash
pip install testkitLLM
```

Start building reliable AI systems today!
