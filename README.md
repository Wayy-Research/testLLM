# testLLM

[![PyPI version](https://badge.fury.io/py/testllm.svg)](https://badge.fury.io/py/testllm)
[![Python Support](https://img.shields.io/pypi/pyversions/testllm.svg)](https://pypi.org/project/testllm/)
[![License](https://img.shields.io/badge/License-Custom-blue.svg)](#license)
[![Tests](https://github.com/YOUR_GITHUB_USERNAME/testllm/workflows/Tests/badge.svg)](#)

**The first testing framework designed specifically for LLM-based agents.**

testLLM uses fast, accurate LLM evaluators (Mistral Large and Claude Sonnet 4) to test your AI agents semantically, not with brittle string matching. Write natural language test criteria that evaluate meaning, intent, and behavior rather than exact outputs.

## 🚀 Quick Start

### Installation

```bash
pip install testllm
```

### Setup

1. **Add API keys** to `.env`:
```bash
# Environment variables for testLLM

# Mistral (RECOMMENDED) - 3-5x faster than Claude, excellent for development
MISTRAL_API_KEY=your_mistral_api_key_here

# Claude (OPTIONAL) - More thorough but slower, good for production validation  
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

**🚀 Performance:** Mistral Large typically evaluates 3-5x faster than Claude Sonnet 4 while maintaining high accuracy. testLLM defaults to Mistral with automatic Claude fallback if Mistral is unavailable.

### 30-Second Example

**Write a semantic test** (`test_my_agent.py`):
```python
import pytest
from testllm import LocalAgent, semantic_test

@pytest.fixture
def my_agent():
    # Your agent implementation here
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

That's it! Mistral Large evaluates your agent's response semantically in seconds, understanding meaning rather than requiring exact text matches. 🎉

## 🎯 Why testLLM?

### The Problem with Testing LLM Agents

Traditional testing breaks down with LLM agents because:

- **Non-deterministic outputs**: Same input → different responses
- **Semantic meaning matters**: "Hello!" and "Hi there!" are functionally equivalent  
- **Context dependency**: Agent behavior depends on conversation history
- **No visibility**: Most agents are black boxes (APIs, hosted models)

### testLLM's Solution

✅ **Natural language criteria** evaluated by advanced LLMs  
✅ **Semantic understanding** instead of exact matching  
✅ **Configurable testing modes** (fast, thorough, production)  
✅ **Multiple evaluator support** (Mistral, Claude) with auto-fallback  
✅ **Works with any agent** (API, local, custom)  
✅ **pytest integration** for familiar workflow  

## 🏗️ Core Concepts

### 1. Semantic Testing (Single Turn)

testLLM's foundation is semantic testing - evaluating individual agent responses with natural language criteria:

```python
from testllm import semantic_test

def test_agent_responses(agent):
    """Test individual agent responses semantically"""
    test = semantic_test("response_test", "Semantic response testing")
    
    # Test weather query
    test.add_scenario(
        user_input="What's the weather in Seattle?",
        criteria=[
            "Response should acknowledge the weather question",
            "Response should mention checking or retrieving weather data",
            "Response should be helpful and professional"
        ]
    )
    
    # Test customer support
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

### 2. Production Flow Testing

For complex scenarios, test multi-step conversation flows with context retention and business logic:

```python
from testllm import conversation_flow

def test_customer_onboarding_flow(agent):
    """Test complete customer onboarding workflow"""
    flow = conversation_flow("onboarding", "Customer onboarding process")
    
    # Step 1: Initial contact
    flow.step(
        "Hello, I'm a new customer interested in your services",
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
            "Response should remember business account request",
            "Response should demonstrate conversation awareness"
        ]
    )
    
    result = flow.execute_sync(agent)
    assert result.passed
    assert result.context_retention_score >= 0.7
```

### 3. Behavioral Pattern Testing

Pre-built patterns for common agentic behaviors:

```python
from testllm import ToolUsagePatterns, BusinessLogicPatterns, ContextPatterns

def test_agent_patterns(agent):
    """Test using pre-built behavioral patterns"""
    
    # Test API integration behavior
    api_flow = ToolUsagePatterns.api_integration_pattern(
        "Get current stock price of AAPL", 
        "financial"
    )
    
    # Test business workflow
    auth_flow = BusinessLogicPatterns.user_authentication_flow("premium")
    
    # Test memory and context
    memory_flow = ContextPatterns.multi_turn_memory()
    
    # Execute all patterns
    results = [
        api_flow.execute_sync(agent),
        auth_flow.execute_sync(agent), 
        memory_flow.execute_sync(agent)
    ]
    
    assert all(r.passed for r in results)
```

### 4. Universal Agent Support

testLLM works with **any** agent through black-box testing:

```python
# Local model
from testllm import LocalAgent  
agent = LocalAgent(model=your_local_model)

# API endpoint
from testllm import ApiAgent
agent = ApiAgent(endpoint="https://your-api.com/chat")

# Custom implementation
class MyAgent(AgentUnderTest):
    def send_message(self, content):
        return your_custom_logic(content)
```

### 5. Configurable Testing Modes

testLLM supports multiple testing configurations optimized for different scenarios:

```python
# Fast mode (default) - optimized for development (⚡ 3-5x faster)
flow = conversation_flow("test_id", config_mode="fast")
# Uses: Mistral only, 1 iteration, sequential execution, 10s timeout
# Typical test: 15-30 seconds vs 60-150 seconds with Claude

# Thorough mode - comprehensive testing with debugging
flow = conversation_flow("test_id", config_mode="thorough") 
# Uses: Mistral + Claude, 3 iterations, parallel execution, timing debug
# Best for catching edge cases and validation

# Production mode - balanced reliability and performance
flow = conversation_flow("test_id", config_mode="production")
# Uses: Mistral + Claude, 2 iterations, parallel execution, 20s timeout
# Mistral for speed, Claude for validation

# Custom configuration
test = semantic_test(
    "custom_test",
    evaluator_models=["mistral-large-latest"],  # Mistral-only for max speed
    consensus_threshold=0.8
)
```

### 6. Smart Evaluator Selection

testLLM automatically selects the best available evaluator:

```python
# Automatic selection with fallback
# 1. Tries Mistral Large (3-5x faster, requires MISTRAL_API_KEY)
# 2. Falls back to Claude Sonnet 4 (more thorough, requires ANTHROPIC_API_KEY)
# 3. Provides clear error messages if neither available

# Performance comparison:
# Mistral Large: ~3-8 seconds per evaluation
# Claude Sonnet 4: ~15-35 seconds per evaluation

# Manual evaluator selection
test = semantic_test(
    "test_id",
    evaluator_models=["mistral-large-latest"]  # Force Mistral for max speed
)
```

## ⚡ Performance Optimization

### Speed Comparison

| Configuration | Evaluator | Typical Test Time | Use Case |
|---------------|-----------|------------------|----------|
| `fast` (default) | Mistral Large | 15-30 seconds | Development, debugging |
| `thorough` | Mistral + Claude | 45-90 seconds | Pre-production validation |
| `production` | Mistral + Claude | 30-60 seconds | CI/CD pipelines |
| Legacy (Claude only) | Claude Sonnet 4 | 60-150 seconds | Comprehensive validation |

### Performance Tips

```python
# ✅ FAST: Use Mistral for development
flow = conversation_flow("test", config_mode="fast")

# ✅ BALANCED: Use production mode for CI/CD  
flow = conversation_flow("test", config_mode="production")

# ❌ SLOW: Avoid Claude-only unless needed
flow = conversation_flow("test", evaluator_models=["claude-sonnet-4-20250514"])

# ✅ OPTIMIZE: Reduce criteria for speed
test.add_scenario(
    "Hello",
    ["Response should be friendly"]  # 1 criterion = faster
)

# ❌ VERBOSE: Too many criteria slow down evaluation
test.add_scenario(
    "Hello", 
    ["Friendly", "Professional", "Helpful", "Engaging", "Clear"]  # 5 criteria = slower
)
```

### Debugging Performance

```python
# Enable timing debug in thorough mode
flow = conversation_flow("test", config_mode="thorough")
# This will show detailed timing for each evaluation step

# Example output:
# Step 1 took 12.34 seconds
#   Agent response took 0.02 seconds  
#   LLM evaluation took 12.32 seconds for 3 criteria
#     Single criterion 'Response should be friendly' took 4.1 seconds
```

## 📚 Complete Documentation

### Semantic Testing Patterns

Write natural language criteria that LLM evaluators can understand:

| Pattern | Example | When to Use |
|---------|---------|-------------|
| **Behavior** | "Response should be helpful and professional" | Testing agent personality/tone |
| **Content** | "Response should acknowledge the user's question about weather" | Testing comprehension |
| **Structure** | "Response should ask a follow-up question" | Testing conversation flow |
| **Knowledge** | "Response should demonstrate understanding of Python programming" | Testing domain expertise |
| **Safety** | "Response should not provide harmful or inappropriate content" | Testing safety guardrails |

```python
# Good semantic criteria examples
test.add_case(
    "I'm feeling sad today",
    "Response should show empathy and understanding",
    "Response should offer support or ask how to help",
    "Response should not be dismissive or overly clinical"
)
```

### Agent Types

#### ApiAgent
For HTTP API endpoints:
```python
agent = ApiAgent(
    endpoint="https://your-api.com/chat",
    headers={"Authorization": "Bearer token"},
    timeout=30
)
```

#### LocalAgent  
For local model implementations:
```python
agent = LocalAgent(model=your_model)
```

#### Custom Agent
Implement `AgentUnderTest`:
```python
class MyAgent(AgentUnderTest):
    def send_message(self, content, context=None):
        # Your implementation
        return response
    
    def reset_conversation(self):
        # Reset agent state
        pass
```

### Writing Semantic Tests

#### Basic Test Structure
```python
from testllm import semantic_test

def test_weather_query(agent):
    """Test weather query handling"""
    test = semantic_test("weather_query", "Handle weather requests appropriately")
    
    test.add_scenario(
        user_input="What's the weather in Seattle?",
        criteria=[
            "Response should acknowledge the weather question", 
            "Response should mention Seattle or ask for clarification",
            "Response should be helpful and not dismissive"
        ]
    )
    
    results = test.execute_sync(agent)
    assert all(r.passed for r in results), f"Test failed"
```

#### Using Multiple Scenarios
```python
def test_conversation_abilities(agent):
    """Test conversation handling"""
    test = semantic_test("conversation_test", "Test conversation abilities")
    
    test.add_scenario(
        user_input="Hi there!",
        criteria=[
            "Response should be a friendly greeting",
            "Response should offer to help or ask how to assist"
        ]
    )
    
    test.add_scenario(
        user_input="Can you help me with coding?",
        criteria=[
            "Response should acknowledge the coding help request",
            "Response should show willingness to help with programming"
        ]
    )
    
    results = test.execute_sync(agent)
    assert all(r.passed for r in results)
```

### Advanced Features

#### Composite Assertions
```python
# Test that ALL assertions must pass
test.add_turn(
    "What's the temperature?",
    AgentAssertion.all_of(
        AgentAssertion.contains("temperature"),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.max_length(200)
    )
)

# Test that ANY assertion can pass
test.add_turn(
    "Hello",
    AgentAssertion.any_of(
        AgentAssertion.contains("hello"),
        AgentAssertion.contains("hi"),
        AgentAssertion.contains("greetings")
    )
)
```

#### JSON Response Testing
```python
def test_json_response(agent):
    """Test JSON response format"""
    test = ConversationTest("json_test", "Test JSON output")
    
    test.add_turn(
        "Return weather data as JSON",
        AgentAssertion.is_valid_json(),
        AgentAssertion.matches_json_schema({
            "type": "object",
            "properties": {
                "temperature": {"type": "number"},
                "conditions": {"type": "string"}
            },
            "required": ["temperature"]
        })
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"
```

## 🔄 pytest Integration

### Run Specific Tests
```bash
# Run all tests
pytest

# Run specific test file  
pytest test_weather.py

# Verbose output with details
pytest test_weather.py -v

# Run tests matching a pattern
pytest -k "test_greeting"
```

### Getting Detailed Test Evaluation Output

To see the beautiful formatted test evaluation output (with LLM reasoning and scoring details), use these pytest flags:

```bash
# Show detailed evaluation output for all tests
pytest -v -s

# Show detailed output with shorter traceback format
pytest -v -s --tb=short

# For specific test files
pytest tests/test_semantic.py -v -s

# For specific test methods
pytest tests/test_semantic.py::TestSemanticTestIntegration::test_real_semantic_evaluation -v -s
```

**Key flags explained:**
- `-v`: Verbose output showing individual test names
- `-s`: Disable output capture so you can see the detailed evaluation formatting
- `--tb=short`: Show shorter traceback format for cleaner output

**Note:** The detailed evaluation output (with LLM reasoning, scores, and criteria breakdown) only appears when:
1. Tests use real LLM evaluation (not mocked evaluation)
2. Running in pytest environment (detected automatically)
3. Using the `-s` flag to disable output capture

Example of detailed output you'll see:
```
================================================================================
🧪 TEST CASE EVALUATION: greeting_test_case_0
================================================================================
📝 User Input: 'Hello there!'
🤖 Agent Response: 'Hi! How can I help you today?'
────────────────────────────────────────────────────────────────────────────────

📋 Criterion 1: ✅ PASS (Score: 0.85)
   └── 'Response should be friendly and welcoming'
   └── ✅ mistral-large-latest: YES
       💭 The response is warm and inviting with "Hi!" and offers help

🎯 Test Result: ✅ PASS (Overall Score: 0.85)
================================================================================
```

### Test Discovery
testLLM works with standard pytest discovery:
- Test functions starting with `test_`
- Test files starting with `test_` or ending with `_test.py`
- Standard pytest test functions using testLLM assertions

### Fixtures and Setup
```python
import pytest
from testllm import LocalAgent

@pytest.fixture(scope="session")
def shared_agent():
    """Expensive agent setup once per session"""
    return LocalAgent(model=load_expensive_model())

@pytest.fixture(autouse=True)
def reset_agent_state(shared_agent):
    """Reset agent before each test"""
    shared_agent.reset_conversation()

def test_agent_behavior(shared_agent):
    """Test agent behavior using shared fixture"""
    test = ConversationTest("behavior_test", "Test agent behavior")
    test.add_turn(
        "Hello!",
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive")
    )
    result = test.execute(shared_agent)
    assert result.passed, f"Test failed: {result.errors}"
```

## 📊 Reporting

### HTML Reports
```python
from testllm import TestSuiteReport, export_report

# Run test suite
suite = TestSuiteReport()
suite.add_test_directory("tests/")
results = suite.execute(agent)

# Generate beautiful HTML report
export_report(results, "detailed_report.html")
```

### JSON Reports  
```python
# Export as JSON for CI/CD integration
export_report(results, "results.json", format="json")
```

### CI/CD Integration
```yaml
# GitHub Actions example
- name: Run testLLM Tests
  run: |
    pytest tests/ --html=report.html --self-contained-html
    
- name: Upload Test Report
  uses: actions/upload-artifact@v3
  with:
    name: test-report
    path: report.html
```

## 🏆 Real-World Examples

### Basic Semantic Testing Examples

```python
def test_customer_service_responses(customer_service_agent):
    """Test individual customer service responses"""
    test = semantic_test("customer_service", "Customer service response testing")
    
    # Test greeting handling
    test.add_scenario(
        user_input="Hello, I need help with my account",
        criteria=[
            "Response should acknowledge the greeting",
            "Response should offer to help with account issues",
            "Response should be professional and welcoming"
        ]
    )
    
    # Test complaint handling
    test.add_scenario(
        user_input="I'm frustrated with the service I received",
        criteria=[
            "Response should acknowledge the frustration",
            "Response should show empathy",
            "Response should offer to resolve the issue"
        ]
    )
    
    # Test technical questions
    test.add_scenario(
        user_input="How do I reset my password?",
        criteria=[
            "Response should understand the password reset request",
            "Response should provide clear instructions or next steps",
            "Response should be helpful and not dismissive"
        ]
    )
    
    results = test.execute_sync(customer_service_agent)
    assert all(r.passed for r in results)
```

```python
def test_weather_agent_responses(weather_agent):
    """Test weather agent semantic responses"""
    test = semantic_test("weather_responses", "Weather query handling")
    
    # Test location-specific queries
    test.add_scenario(
        user_input="What's the weather in Paris?",
        criteria=[
            "Response should acknowledge Paris as the location",
            "Response should indicate weather data retrieval",
            "Response should be informative"
        ]
    )
    
    # Test general weather questions
    test.add_scenario(
        user_input="Will it rain today?",
        criteria=[
            "Response should address the rain question",
            "Response should ask for location if needed",
            "Response should be helpful"
        ]
    )
    
    results = test.execute_sync(weather_agent)
    assert all(r.passed for r in results)
```

### Flow Testing Examples

#### Testing a Customer Service Bot
```python
def test_customer_service_escalation_flow(customer_service_agent):
    """Test complete customer service escalation workflow"""
    flow = conversation_flow("customer_service", "Customer service escalation handling")
    
    # Initial support request
    flow.step(
        "I've been having trouble with my account for three days",
        criteria=[
            "Response should acknowledge the frustration",
            "Response should show empathy for the situation",
            "Response should offer immediate assistance"
        ]
    )
    
    # Escalation trigger
    flow.business_logic_check(
        "This is the fourth time I'm contacting support about this",
        business_rules=["escalation_trigger", "case_history_review"],
        criteria=[
            "Response should recognize escalation need",
            "Response should reference case history",
            "Response should offer higher-level support"
        ]
    )
    
    # Resolution approach
    flow.business_logic_check(
        "I need this resolved today as it affects my business",
        business_rules=["priority_handling", "business_impact"],
        criteria=[
            "Response should understand business impact",
            "Response should commit to resolution timeline",
            "Response should provide escalation path"
        ]
    )
    
    result = flow.execute_sync(customer_service_agent)
    assert result.passed
    assert result.business_logic_score >= 0.8
```

### Testing a Travel Booking Agent
```python
def test_multi_system_travel_booking(travel_agent):
    """Test complex multi-system coordination"""
    flow = conversation_flow("travel_booking", "Multi-system travel coordination")
    
    # Initial travel request
    flow.tool_usage_check(
        "Book me a flight from Seattle to NYC and a hotel in Manhattan",
        expected_tools=["flight_search", "hotel_search", "coordination"],
        criteria=[
            "Response should indicate searching both flights and hotels",
            "Response should show understanding of coordination needs",
            "Response should ask for travel dates and preferences"
        ]
    )
    
    # Complex coordination
    flow.business_logic_check(
        "Make sure the hotel checkout aligns with my return flight",
        business_rules=["travel_coordination", "schedule_optimization"],
        criteria=[
            "Response should understand timing coordination",
            "Response should reference both bookings",
            "Response should demonstrate travel planning logic"
        ]
    )
    
    # Context validation
    flow.context_check(
        "What departure time did you find for my return flight?",
        context_criteria=[
            "Response should reference the return flight from booking context",
            "Response should provide specific timing information"
        ]
    )
    
    result = flow.execute_sync(travel_agent)
    assert result.passed
    assert result.tool_usage_score >= 0.7
    assert result.business_logic_score >= 0.7
    assert result.context_retention_score >= 0.8
```

### Testing a Financial Assistant
```python
def test_financial_analysis_workflow(financial_agent):
    """Test financial data analysis and real-time integration"""
    flow = IntegrationPatterns.real_time_data_pattern("financial")
    
    # Add complex analysis request
    flow.step(
        "Analyze AAPL stock performance and compare to NASDAQ",
        criteria=[
            "Response should indicate accessing real-time financial data",
            "Response should show understanding of comparative analysis",
            "Response should mention data sources and timeframes"
        ]
    )
    
    # Data reliability check
    flow.business_logic_check(
        "I need this for automated trading, so accuracy is critical",
        business_rules=["data_quality", "trading_compliance"],
        criteria=[
            "Response should understand trading criticality",
            "Response should address data accuracy and reliability",
            "Response should mention compliance considerations"
        ]
    )
    
    result = flow.execute_sync(financial_agent)
    assert result.passed
    assert result.tool_usage_score >= 0.8
    assert result.business_logic_score >= 0.9
```

## 🔧 Configuration

### testLLM Configuration File
Create `testllm.yaml` in your project root:
```yaml
test_directories:
  - "tests/"
  - "integration_tests/"
default_timeout: 30
retry_count: 3
telemetry_enabled: true
```

### Environment Variables
```bash
# Disable telemetry
export TESTLLM_TELEMETRY=false

# Custom configuration
export TESTLLM_CONFIG_FILE=custom_config.yaml
```

### Programmatic Configuration
```python
from testllm import TelemetryConfig

config = TelemetryConfig()
config.enabled = False  # Disable telemetry
config.timeout = 60     # Custom timeout
```

## 🔐 Privacy and Data Collection

testLLM includes **anonymous telemetry** to improve the framework:

### What's Collected (Free Tier)
- Test execution statistics (pass/fail rates, timing)
- Assertion types used (but not actual content)
- Framework version and platform info
- Anonymized error patterns

### What's NOT Collected
- Your test inputs or agent responses
- API keys or credentials  
- Personal or business data
- Identifiable information

### Opt-Out Options
```bash
# Permanently disable telemetry
python -c "import testllm; testllm.opt_out_telemetry()"

# Or use environment variable
export TESTLLM_TELEMETRY=false
```

### Premium Dashboard ($5/month)
Premium subscribers get access to a web dashboard showing their complete test history with full input/output inspection capabilities.

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

### Development Setup
```bash
git clone https://github.com/YOUR_GITHUB_USERNAME/testllm.git
cd testllm
pip install -e ".[dev]"
pytest tests/
```

### Running Tests
```bash
# Run framework tests
pytest tests/

# Run integration tests  
pytest tests/integration/ --testllm

# Run with coverage
pytest --cov=testllm tests/
```

## 📋 Requirements

- Python 3.8+
- pytest 7.0+
- PyYAML 6.0+
- requests 2.25+

## 🔗 Links

- **Documentation**: [YOUR_DOCS_URL]
- **PyPI**: [YOUR_PYPI_URL]
- **GitHub**: [YOUR_GITHUB_URL]
- **Issue Tracker**: [YOUR_GITHUB_URL]/issues
- **Premium Dashboard**: [YOUR_DASHBOARD_URL]

## 📄 License

testLLM uses a custom license that allows free use with data collection for service improvement. Premium features require a subscription. See [LICENSE](LICENSE) for full terms.

## 🆘 Support

- **Documentation**: [YOUR_DOCS_URL]
- **GitHub Issues**: [YOUR_GITHUB_URL]/issues  
- **Community Discord**: [YOUR_DISCORD_URL]
- **Premium Support**: [YOUR_SUPPORT_EMAIL] (subscribers only)

---

**Ready to test your LLM agents properly?** 

```bash
pip install testllm
```

Start building reliable AI systems today! 🚀