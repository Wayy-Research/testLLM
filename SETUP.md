# testLLM Setup Guide

## Quick Start

1. **Install the package**:
   ```bash
   pip install -e .
   ```

2. **Run the basic tests**:
   ```bash
   pytest tests/test_simple.py -v
   ```

## Setup for Real LLM Testing (Optional)

If you want to test with real LLM APIs:

1. **Create a `.env` file** (copy from `.env.example`):
   ```bash
   cp .env.example .env
   ```

2. **Add your API keys** to `.env`:
   ```
   OPENAI_API_KEY=your_openai_api_key_here
   ANTHROPIC_API_KEY=your_anthropic_api_key_here
   ```

3. **Run tests with real agents**:
   ```bash
   pytest tests/test_real_agents.py -v
   ```

## Test Structure

- `tests/test_simple.py` - Basic tests that work without external dependencies
- `tests/test_real_agents.py` - Tests that use real LLM APIs (requires API keys)
- `test_yaml/` - YAML test definitions
- `examples/` - Example YAML test files

## Basic Usage

```python
from testllm import ConversationTest, UserTurn, AgentAssertion, LocalAgent

# Create a simple agent
class MyAgent:
    def __call__(self, content):
        return "Hello! How can I help you?"

agent = LocalAgent(model=MyAgent())

# Create and run a test
test = ConversationTest("greeting", "Test greeting")
test.add_turn(
    UserTurn("Hello"),
    AgentAssertion.contains("Hello"),
    AgentAssertion.max_length(100)
)

result = test.execute(agent)
assert result.passed
```

## Expected Test Results

After setup, all core tests should pass:

### Core Tests (should all pass ✅)
```bash
pytest tests/test_simple.py tests/test_mock_agent.py tests/test_yaml_definitions.py -v
```

- ✅ `test_simple.py` - 3 tests: Basic conversation, YAML loading, example tests
- ✅ `test_mock_agent.py` - 9 tests: All assertion types and agent functionality  
- ✅ `test_yaml_definitions.py` - 3 tests: YAML test definitions

**Total: 15 passing tests**

### Real API Tests (optional, require API keys)
```bash
pytest tests/test_real_agents.py -v
```
These tests require real API keys and may fail without proper configuration.

## Troubleshooting

If tests fail:
1. Make sure the package is installed: `pip install -e .`
2. Check that you're in the virtual environment: `source .venv/bin/activate`
3. Verify the test files exist in the expected locations
4. Run individual tests to isolate issues: `pytest tests/test_simple.py::test_basic_conversation -v`