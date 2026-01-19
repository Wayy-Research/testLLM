"""
Tool Testing Module - Comprehensive tool testing capabilities for testLLM.

This module provides three complementary approaches for testing tool usage:

1. **Tool Expectation DSL** - Declarative API for specifying tool expectations
2. **Tool Interceptor** - Framework-agnostic tool call interception and mocking
3. **Response Simulation** - Test agent behavior under various tool scenarios

Example Usage:

    # Approach 1: Tool Expectation DSL
    from testllm.tool_testing import expect_tools

    expectations = (
        expect_tools()
        .expect_call("search_flights")
        .with_arguments_containing(destination="NYC")
        .returning({"flights": [{"id": "F1", "price": 299}]})
    )

    # Approach 2: Tool Interceptor
    from testllm.tool_testing import ToolInterceptor

    interceptor = ToolInterceptor()
    interceptor.register_mock("search_flights", {"flights": [...]})

    # Approach 3: Response Simulation
    from testllm.tool_testing import simulate_tool

    simulator = (
        simulate_tool("search_flights")
        .on_success({"flights": [...]})
        .on_failure("Service unavailable")
        .on_timeout(30000)
    )
"""

# Types
from .types import (
    ToolCall,
    ToolResponse,
    ToolSchema,
    ToolCallStatus,
    ArgumentMatchMode,
    CallCountMode,
    ValidationResult,
    ExpectationResult,
    ToolExpectationSummary,
    InterceptedCall,
)

# Expectation DSL
from .expectations import (
    expect_tools,
    expect_tool,
    ToolExpectations,
    ToolExpectationBuilder,
    SingleToolExpectation,
)

# Interceptor
from .interceptor import (
    BaseToolInterceptor,
    ToolInterceptor,
    RecordingInterceptor,
    CompositeInterceptor,
)

# Adapters
from .adapters import (
    BaseToolAdapter,
    GenericToolAdapter,
    AnthropicAdapter,
    LangChainAdapter,
    AutoAdapter,
    parse_tool_calls,
)

# Simulation
from .simulation import (
    simulate_tool,
    tool_response_suite,
    chaos_simulator,
    ToolSimulator,
    ToolResponseSuite,
    ChaosSimulator,
    ScenarioType,
    ResponseScenario,
    SimulationTestCase,
)

# Pre-built scenarios
from .scenarios import (
    SearchScenarios,
    CRUDScenarios,
    APIScenarios,
    DatabaseScenarios,
    FileScenarios,
    CalendarScenarios,
    get_all_scenarios,
)

__all__ = [
    # Types
    "ToolCall",
    "ToolResponse",
    "ToolSchema",
    "ToolCallStatus",
    "ArgumentMatchMode",
    "CallCountMode",
    "ValidationResult",
    "ExpectationResult",
    "ToolExpectationSummary",
    "InterceptedCall",

    # Expectation DSL
    "expect_tools",
    "expect_tool",
    "ToolExpectations",
    "ToolExpectationBuilder",
    "SingleToolExpectation",

    # Interceptor
    "BaseToolInterceptor",
    "ToolInterceptor",
    "RecordingInterceptor",
    "CompositeInterceptor",

    # Adapters
    "BaseToolAdapter",
    "GenericToolAdapter",
    "AnthropicAdapter",
    "LangChainAdapter",
    "AutoAdapter",
    "parse_tool_calls",

    # Simulation
    "simulate_tool",
    "tool_response_suite",
    "chaos_simulator",
    "ToolSimulator",
    "ToolResponseSuite",
    "ChaosSimulator",
    "ScenarioType",
    "ResponseScenario",
    "SimulationTestCase",

    # Pre-built scenarios
    "SearchScenarios",
    "CRUDScenarios",
    "APIScenarios",
    "DatabaseScenarios",
    "FileScenarios",
    "CalendarScenarios",
    "get_all_scenarios",
]
