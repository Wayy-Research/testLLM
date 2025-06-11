# testLLM

[![PyPI version](https://badge.fury.io/py/testllm.svg)](https://badge.fury.io/py/testllm)
[![Python Support](https://img.shields.io/pypi/pyversions/testllm.svg)](https://pypi.org/project/testllm/)
[![License](https://img.shields.io/badge/License-Custom-blue.svg)](#license)
[![Tests](https://github.com/YOUR_GITHUB_USERNAME/testllm/workflows/Tests/badge.svg)](#)

**The first testing framework designed specifically for LLM-based agents.**

testLLM uses Claude Sonnet 4 as an intelligent evaluator to test your AI agents semantically, not with brittle string matching. Write natural language test criteria that evaluate meaning, intent, and behavior rather than exact outputs.

## 🚀 Quick Start

### Installation

```bash
pip install testllm
```

### Setup

1. **Add your Anthropic API key** to `.env`:
```bash
# Environment variables for testLLM
ANTHROPIC_API_KEY=your_anthropic_api_key_here
```

### 30-Second Example

**Write a semantic test** (`test_my_agent.py`):
```python
import pytest
from testllm import LocalAgent, semantic_test

@pytest.fixture
def my_agent():
    # Your agent implementation here
    class SimpleAgent:
        def __call__(self, prompt):
            return "Hello! How can I help you today?"
    
    return LocalAgent(model=SimpleAgent())

def test_greeting_behavior(my_agent):
    """Test that agent greets users appropriately"""
    test = semantic_test("greeting_test", "Agent should greet users appropriately")
    
    test.add_case(
        "Hello there!",
        "Response should be a friendly greeting",
        "Response should offer to help the user",
        "Response should be polite and professional"
    )
    
    results = test.execute_sync(my_agent)
    assert all(r.passed for r in results), f"Test failed: {[r.errors for r in results if not r.passed]}"
```

**Run it**:
```bash
pytest test_my_agent.py -v
```

That's it! Claude Sonnet 4 evaluates your agent's responses against natural language criteria. 🎉

## 🎯 Why testLLM?

### The Problem with Testing LLM Agents

Traditional testing breaks down with LLM agents because:

- **Non-deterministic outputs**: Same input → different responses
- **Semantic meaning matters**: "Hello!" and "Hi there!" are functionally equivalent  
- **Context dependency**: Agent behavior depends on conversation history
- **No visibility**: Most agents are black boxes (APIs, hosted models)

### testLLM's Solution

✅ **Natural language criteria** evaluated by Claude Sonnet 4  
✅ **Semantic understanding** instead of exact matching  
✅ **Multiple iterations** for reliable stochastic testing  
✅ **Works with any agent** (API, local, custom)  
✅ **pytest integration** for familiar workflow  

## 🏗️ Core Concepts

### 1. Semantic Testing with Claude Sonnet 4

testLLM uses Claude Sonnet 4 as an intelligent evaluator. Write natural language criteria:

```python
from testllm import semantic_test

def test_customer_support(agent):
    """Test customer support responses"""
    test = semantic_test("support_test", "Customer support behavior")
    
    test.add_case(
        "I need help with my account",
        "Response should acknowledge the help request",
        "Response should be professional and helpful",
        "Response should ask for more details or offer specific assistance"
    )
    
    results = test.execute_sync(agent)
    assert all(r.passed for r in results)
```

### 2. Universal Agent Support

testLLM works with **any** agent by focusing on input/output testing:

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

### 3. Stochastic Reliability

Since LLM outputs are non-deterministic, testLLM runs multiple evaluations:

```python
# Default: 3 iterations with 67% consensus threshold
test = semantic_test("reliability_test")

# Custom reliability settings
test = semantic_test(
    "custom_test", 
    evaluator_models=["claude-sonnet-4"],  # Always Claude Sonnet 4
    consensus_threshold=0.8  # 80% consensus required
)
```

### 4. Multiple Test Cases

Test different scenarios in a single test:

```python
def test_agent_versatility(agent):
    """Test agent across different scenarios"""
    test = semantic_test("versatility_test", "Agent handles various requests")
    
    # Greeting scenario
    test.add_case(
        "Hello!",
        "Response should be a friendly greeting",
        "Response should offer assistance"
    )
    
    # Information request
    test.add_case(
        "What's the weather in Seattle?",
        "Response should acknowledge the weather question",
        "Response should ask for clarification or provide helpful guidance"
    )
    
    # Task request
    test.add_case(
        "Help me write an email",
        "Response should offer to help with email writing",
        "Response should ask for details about the email"
    )
    
    results = test.execute_sync(agent)
    assert all(r.passed for r in results)
```

## 📚 Complete Documentation

### Semantic Testing Patterns

Write natural language criteria that Claude Sonnet 4 can evaluate:

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
    
    test.add_case(
        "What's the weather in Seattle?",
        "Response should acknowledge the weather question", 
        "Response should mention Seattle or ask for clarification",
        "Response should be helpful and not dismissive"
    )
    
    results = test.execute_sync(agent)
    assert all(r.passed for r in results), f"Test failed"
```

#### Using the Pytest Decorator
```python
from testllm import pytest_semantic_test

@pytest_semantic_test("conversation_test", "Test conversation abilities")
def test_conversation_flow(agent):
    """Test conversation handling"""
    return [
        ("Hi there!", [
            "Response should be a friendly greeting",
            "Response should offer to help or ask how to assist"
        ]),
        ("Can you help me with coding?", [
            "Response should acknowledge the coding help request",
            "Response should show willingness to help with programming"
        ])
    ]
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

### Testing a Customer Service Bot
```python
def test_customer_service_flow(customer_service_agent):
    """Test customer service using semantic evaluation"""
    test = semantic_test("customer_service", "Customer service bot handles inquiries professionally")
    
    test.add_case(
        "Hi, I need help with my order",
        "Response should acknowledge the help request professionally",
        "Response should ask for order details or account information",
        "Response should be empathetic and helpful"
    )
    
    test.add_case(
        "My order number is 12345 and it's late",
        "Response should acknowledge the order number",
        "Response should address the delivery concern",
        "Response should offer solutions or next steps"
    )
    
    results = test.execute_sync(customer_service_agent)
    assert all(r.passed for r in results)
```

### Testing a Code Assistant
```python
def test_code_generation(code_assistant_agent):
    """Test code generation capabilities"""
    test = semantic_test("code_generation", "Code assistant generates valid Python")
    
    test.add_case(
        "Write a function to calculate fibonacci numbers",
        "Response should contain a Python function definition",
        "Response should implement fibonacci logic correctly",
        "Response should include proper function syntax",
        "Response should be well-formatted and readable"
    )
    
    results = test.execute_sync(code_assistant_agent)
    assert all(r.passed for r in results)
```

### Testing a Data Analysis Agent
```python
def test_data_analysis(data_agent):
    """Test data analysis capabilities"""
    test = semantic_test("data_analysis", "Agent provides structured data insights")
    
    test.add_case(
        "Analyze this data and return JSON: [1,2,3,4,5]",
        "Response should provide data analysis in JSON format",
        "Response should include statistical measures like mean or sum",
        "Response should be properly formatted as valid JSON"
    )
    
    results = test.execute_sync(data_agent)
    assert all(r.passed for r in results)
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