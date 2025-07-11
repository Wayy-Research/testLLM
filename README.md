# testLLM

**The first testing framework designed specifically for LLM-based agents.**

testLLM uses fast, accurate LLM evaluators (Mistral Large and Claude Sonnet 4) to test your AI agents semantically, not with brittle string matching. Write natural language test criteria that evaluate meaning, intent, and behavior rather than exact outputs.

## 🚀 Quick Start

### Installation

```bash
pip install testllm
```

### Setup

Add API keys to `.env`:
```bash
# Mistral (RECOMMENDED) - 3-5x faster than Claude
MISTRAL_API_KEY=your_mistral_api_key_here

# Claude (OPTIONAL) - More thorough but slower
ANTHROPIC_API_KEY=your_anthropic_api_key_here
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

That's it! testLLM evaluates your agent's response semantically, understanding meaning rather than requiring exact text matches. 🎉

## 🎯 Core Features

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

### 3. Behavioral Pattern Testing

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

### 4. Universal Agent Support

testLLM works with **any** agent:

```python
# Local model
from testllm import LocalAgent  
agent = LocalAgent(model=your_local_model)

# API endpoint
from testllm import ApiAgent
agent = ApiAgent(endpoint="https://your-api.com/chat")

# Custom implementation
from testllm import AgentUnderTest

class MyAgent(AgentUnderTest):
    def send_message(self, content, context=None):
        return your_custom_logic(content)
    
    def reset_conversation(self):
        pass
```

## ⚡ Performance & Configuration

### Testing Modes

```python
# Fast mode (default) - optimized for development
flow = conversation_flow("test_id", config_mode="fast")
# Uses: Mistral only, ~15-30 seconds per test

# Production mode - balanced reliability and performance  
flow = conversation_flow("test_id", config_mode="production")
# Uses: Mistral + Claude validation, ~30-60 seconds per test

# Thorough mode - comprehensive testing
flow = conversation_flow("test_id", config_mode="thorough") 
# Uses: Mistral + Claude, multiple iterations, ~45-90 seconds per test
```

### Custom Configuration

```python
test = semantic_test(
    "custom_test",
    evaluator_models=["mistral-large-latest"],  # Mistral-only for max speed
    consensus_threshold=0.8
)
```

## 🔧 pytest Integration

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

### Example Test Structure

```python
import pytest
from testllm import LocalAgent, semantic_test

@pytest.fixture(scope="session")
def agent():
    """Setup agent once per session"""
    return LocalAgent(model=your_model)

@pytest.fixture(autouse=True)
def reset_agent(agent):
    """Reset agent state before each test"""
    agent.reset_conversation()

def test_greeting_behavior(agent):
    """Test agent greeting behavior"""
    test = semantic_test("greeting_test", "Greeting behavior")
    
    test.add_scenario(
        user_input="Hello!",
        criteria=[
            "Response should be friendly",
            "Response should offer to help"
        ]
    )
    
    results = test.execute_sync(agent)
    assert all(r.passed for r in results)
```

## 📊 Real-World Examples

### E-commerce Agent Testing

```python
def test_purchase_flow(ecommerce_agent):
    """Test complete purchase workflow"""
    flow = conversation_flow("purchase", "E-commerce purchase flow")
    
    # Product search
    flow.tool_usage_check(
        "I'm looking for a laptop for machine learning",
        expected_tools=["product_search", "specification_filter"],
        criteria=[
            "Response should search product catalog",
            "Response should understand ML requirements"
        ]
    )
    
    # Purchase process
    flow.business_logic_check(
        "I want the Dell XPS with 32GB RAM",
        business_rules=["inventory_check", "pricing"],
        criteria=[
            "Response should check availability",
            "Response should provide pricing",
            "Response should offer purchasing options"
        ]
    )
    
    result = flow.execute_sync(ecommerce_agent)
    assert result.passed
```

### Customer Support Testing

```python
def test_support_escalation(support_agent):
    """Test support escalation workflow"""
    flow = conversation_flow("escalation", "Support escalation")
    
    # Initial complaint
    flow.step(
        "I've been having issues for three days with no help",
        criteria=[
            "Response should acknowledge frustration",
            "Response should show empathy",
            "Response should offer immediate assistance"
        ]
    )
    
    # Escalation trigger
    flow.business_logic_check(
        "This is the fourth time I'm contacting support",
        business_rules=["escalation_trigger", "case_history"],
        criteria=[
            "Response should recognize escalation need",
            "Response should offer higher-level support"
        ]
    )
    
    result = flow.execute_sync(support_agent)
    assert result.passed
    assert result.business_logic_score >= 0.8
```

## 🏗️ Writing Effective Tests

### Good Semantic Criteria

| Pattern | Example | When to Use |
|---------|---------|-------------|
| **Behavior** | "Response should be helpful and professional" | Testing agent personality |
| **Content** | "Response should acknowledge the weather question" | Testing comprehension |
| **Structure** | "Response should ask a follow-up question" | Testing conversation flow |
| **Safety** | "Response should not provide harmful content" | Testing guardrails |

### Performance Tips

```python
# ✅ FAST: Use fewer, focused criteria
test.add_scenario(
    "Hello",
    ["Response should be friendly"]  # 1 criterion = faster
)

# ❌ SLOW: Too many criteria
test.add_scenario(
    "Hello", 
    ["Friendly", "Professional", "Helpful", "Engaging", "Clear"]  # 5 criteria = slower
)

# ✅ FAST: Use fast mode for development
flow = conversation_flow("test", config_mode="fast")

# ✅ BALANCED: Use production mode for CI/CD
flow = conversation_flow("test", config_mode="production")
```

## 📋 Requirements

- Python 3.8+
- pytest 7.0+
- At least one API key (Mistral or Anthropic)

## 🆘 Support

- **GitHub Issues**: For bug reports and feature requests
- **Documentation**: For detailed guides and examples

## 🚀 Development & Release Process

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
   git push origin v0.1.1  # Replace with your version
   ```

3. **Create GitHub Release:**
   - Go to https://github.com/Wayy-Research/testLLM/releases
   - Click "Create a new release"
   - Select your tag (e.g., v0.1.1)
   - Add release notes
   - Publish release

This will automatically:
- Run tests across Python 3.8-3.12
- Deploy to TestPyPI on every push to main
- Deploy to PyPI on GitHub releases

### Continuous Deployment

The project uses GitHub Actions for continuous deployment:

- **Every push to main**: Automatically uploads to TestPyPI
- **Every GitHub release**: Automatically uploads to PyPI
- **All commits**: Run core tests (full test suite runs locally with API keys)

---

**Ready to test your LLM agents properly?** 

```bash
pip install testllm
```

Start building reliable AI systems today! 🚀