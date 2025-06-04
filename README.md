# testLLM

[![PyPI version](https://badge.fury.io/py/testllm.svg)](https://badge.fury.io/py/testllm)
[![Python Support](https://img.shields.io/pypi/pyversions/testllm.svg)](https://pypi.org/project/testllm/)
[![License](https://img.shields.io/badge/License-Custom-blue.svg)](#license)
[![Tests](https://github.com/YOUR_GITHUB_USERNAME/testllm/workflows/Tests/badge.svg)](#)

**The first testing framework designed specifically for LLM-based agents.**

testLLM solves the unique challenges of testing AI agents by focusing on semantic validation rather than exact string matching. Test any agent implementation using Claude/Anthropic as the live testing agent, with support for local models and custom implementations.

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

**Write a pytest test** (`test_my_agent.py`):
```python
import pytest
from testllm import LocalAgent, ConversationTest, AgentAssertion

@pytest.fixture
def my_agent():
    # Your agent implementation here
    class SimpleAgent:
        def __call__(self, prompt):
            return "Hello! How can I help you today?"
    
    return LocalAgent(model=SimpleAgent())

def test_greeting(my_agent):
    """Test basic greeting functionality"""
    test = ConversationTest("greeting_test", "Agent should greet users appropriately")
    
    test.add_turn(
        "Hello there!",
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive"),
        AgentAssertion.max_length(150)
    )
    
    result = test.execute(my_agent)
    assert result.passed, f"Test failed: {result.errors}"
```

**Run it**:
```bash
pytest test_my_agent.py -v
```

That's it! 🎉

## 🎯 Why testLLM?

### The Problem with Testing LLM Agents

Traditional testing breaks down with LLM agents because:

- **Non-deterministic outputs**: Same input → different responses
- **Semantic meaning matters**: "Hello!" and "Hi there!" are functionally equivalent  
- **Context dependency**: Agent behavior depends on conversation history
- **No visibility**: Most agents are black boxes (APIs, hosted models)

### testLLM's Solution

✅ **Semantic assertions** instead of exact matching  
✅ **Works with any agent** (API, local, custom)  
✅ **Conversation-aware** testing with proper context  
✅ **Property-based validation** focusing on response characteristics  
✅ **pytest integration** for familiar workflow  

## 🏗️ Core Concepts

### 1. Universal Agent Support

testLLM works with **any** agent by focusing on input/output testing:

```python
# Claude/Anthropic agent (primary live testing agent)
from testllm import LocalAgent

class ClaudeAgent:
    def __init__(self, api_key):
        self.api_key = api_key
    
    def __call__(self, content):
        # Claude API integration
        return claude_response

agent = LocalAgent(model=ClaudeAgent(api_key))

# Local model
from testllm import LocalAgent  
agent = LocalAgent(model=your_local_model)

# Custom implementation
class MyAgent(AgentUnderTest):
    def send_message(self, content):
        return your_custom_logic(content)
```

### 2. Semantic Assertions

Test what matters, not exact wording:

```python
# Instead of brittle exact matching:
assert response == "Hello! How can I help you today?"

# Use semantic assertions:
AgentAssertion.contains("hello")           # Flexible text matching
AgentAssertion.sentiment("positive")       # Emotional tone
AgentAssertion.max_length(100)            # Response constraints
AgentAssertion.is_valid_json()            # Format validation
AgentAssertion.any_of(                     # Multiple acceptable responses
    AgentAssertion.contains("hello"),
    AgentAssertion.contains("hi"),
    AgentAssertion.contains("greetings")
)
```

### 3. Conversation Testing

Test realistic multi-turn interactions:

```python
def test_booking_conversation(agent):
    """Test a hotel booking conversation flow"""
    test = ConversationTest("booking_flow", "Test hotel booking conversation")
    
    # Turn 1: Initial request
    test.add_turn(
        "I want to book a hotel",
        AgentAssertion.contains("hotel"),
        AgentAssertion.sentiment("positive")
    )
    
    # Turn 2: Provide details
    test.add_turn(
        "In New York for 2 nights", 
        AgentAssertion.contains("New York"),
        AgentAssertion.contains("2"),
        AgentAssertion.max_length(200)
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"
```

## 📚 Complete Documentation

### Assertion Types

| Assertion | Purpose | Example |
|-----------|---------|---------|
| `contains` | Text must include pattern | `contains: "weather"` |
| `excludes` | Text must not include pattern | `excludes: "error"` |
| `regex` | Text matches regex | `regex: "\\d{1,2}°[CF]"` |
| `sentiment` | Emotional tone | `sentiment: "positive"` |
| `max_length` | Response length limit | `max_length: 200` |
| `min_length` | Response length minimum | `min_length: 10` |
| `json_valid` | Valid JSON format | `json_valid: true` |
| `json_schema` | Matches JSON schema | `json_schema: {...}` |
| `all_of` | All assertions pass | `all_of: [...]` |
| `any_of` | At least one passes | `any_of: [...]` |

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

### Writing Tests

#### Basic Test Structure
```python
from testllm import ConversationTest, AgentAssertion

def test_weather_query(agent):
    """Test weather query handling"""
    test = ConversationTest("weather_query", "Handle weather requests appropriately")
    
    test.add_turn(
        "What's the weather in Seattle?",
        AgentAssertion.contains("weather"),
        AgentAssertion.excludes("I don't know"),
        AgentAssertion.max_length(300)
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"
```

#### Multi-Turn Conversations
```python
def test_conversation_flow(agent):
    """Test a complete conversation"""
    test = ConversationTest("conversation_flow", "Test conversation flow")
    
    # Turn 1: Greeting
    test.add_turn(
        "Hi there!",
        AgentAssertion.contains("hello"),
        AgentAssertion.sentiment("positive")
    )
    
    # Turn 2: Follow-up
    test.add_turn(
        "Can you help me?",
        AgentAssertion.contains("help"),
        AgentAssertion.max_length(200)
    )
    
    result = test.execute(agent)
    assert result.passed, f"Test failed: {result.errors}"
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
    """Test customer service conversation flow"""
    test = ConversationTest("customer_service", "Customer service bot handles inquiries professionally")
    
    # Turn 1: Initial help request
    test.add_turn(
        "Hi, I need help with my order",
        AgentAssertion.sentiment("positive"),
        AgentAssertion.contains("help"),
        AgentAssertion.excludes("sorry")
    )
    
    # Turn 2: Provide order number
    test.add_turn(
        "My order number is 12345",
        AgentAssertion.contains("12345"),
        AgentAssertion.max_length(500),
        AgentAssertion.sentiment("positive")
    )
    
    result = test.execute(customer_service_agent)
    assert result.passed, f"Test failed: {result.errors}"
```

### Testing a Code Assistant
```python
def test_code_generation(code_assistant_agent):
    """Test code generation capabilities"""
    test = ConversationTest("code_generation", "Code assistant generates valid Python")
    
    test.add_turn(
        "Write a function to calculate fibonacci numbers",
        AgentAssertion.contains("def"),
        AgentAssertion.contains("fibonacci"),
        AgentAssertion.regex(r"def\s+\w+\s*\("),
        AgentAssertion.excludes("error")
    )
    
    result = test.execute(code_assistant_agent)
    assert result.passed, f"Test failed: {result.errors}"
```

### Testing a Data Analysis Agent
```python
def test_data_analysis(data_agent):
    """Test data analysis with JSON output"""
    test = ConversationTest("data_analysis", "Agent provides structured data insights")
    
    test.add_turn(
        "Analyze this data and return JSON: [1,2,3,4,5]",
        AgentAssertion.is_valid_json(),
        AgentAssertion.matches_json_schema({
            "type": "object",
            "properties": {
                "mean": {"type": "number"},
                "median": {"type": "number"}
            }
        })
    )
    
    result = test.execute(data_agent)
    assert result.passed, f"Test failed: {result.errors}"
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