"""
testLLM Framework
Testing Framework for LLM-Based Agents
"""

from .core import (
    AgentUnderTest,
    ApiAgent,
    LocalAgent,
    InterceptedAgent,
    # Legacy exports
    ConversationTest,
    UserTurn,
    AgentAssertion,
    agent_test,
    TestResult,
    load_test_file,
    run_test_from_yaml,
)

# Primary semantic testing interface
from .semantic import (
    SemanticTest,
    SemanticTestResult,
    SemanticTestCase,
    semantic_test,
    pytest_semantic_test,
)

# Production flow testing
from .flows import (
    ConversationFlow,
    FlowResult,
    FlowStep,
    conversation_flow,
)

# Behavioral pattern testing
from .behavioral import (
    ToolUsagePatterns,
    BusinessLogicPatterns,
    ContextPatterns,
    IntegrationPatterns,
    PerformancePatterns,
)

from .assertions import (
    AssertionResult,
    BaseAssertion,
    ContainsAssertion,
    ExcludesAssertion,
    MaxLengthAssertion,
    MinLengthAssertion,
    SentimentAssertion,
    JsonValidAssertion,
    RegexAssertion,
    ToolUsageAssertion,
    AllOfAssertion,
    AnyOfAssertion,
)


from .reporting import (
    TestSuiteReport,
    export_report,
)

# Tool testing framework
from .tool_testing import (
    # Types
    ToolCall,
    ToolResponse,
    ToolSchema,
    ToolCallStatus,
    ValidationResult,
    ExpectationResult,
    ToolExpectationSummary,
    # Expectation DSL
    expect_tools,
    expect_tool,
    ToolExpectations,
    # Interceptor
    ToolInterceptor,
    RecordingInterceptor,
    # Adapters
    GenericToolAdapter,
    AnthropicAdapter,
    AutoAdapter,
    parse_tool_calls,
    # Simulation
    simulate_tool,
    tool_response_suite,
    chaos_simulator,
    ToolSimulator,
    ScenarioType,
    # Pre-built scenarios
    SearchScenarios,
    CRUDScenarios,
    APIScenarios,
)

from .__version__ import __version__
__all__ = [
    # Core agent interfaces
    "AgentUnderTest",
    "ApiAgent",
    "LocalAgent",
    "InterceptedAgent",

    # Primary semantic testing interface
    "SemanticTest",
    "SemanticTestResult",
    "SemanticTestCase",
    "semantic_test",
    "pytest_semantic_test",

    # Production flow testing
    "ConversationFlow",
    "FlowResult",
    "FlowStep",
    "conversation_flow",

    # Behavioral pattern testing
    "ToolUsagePatterns",
    "BusinessLogicPatterns",
    "ContextPatterns",
    "IntegrationPatterns",
    "PerformancePatterns",

    # Tool testing framework
    "ToolCall",
    "ToolResponse",
    "ToolSchema",
    "ToolCallStatus",
    "ValidationResult",
    "ExpectationResult",
    "ToolExpectationSummary",
    "expect_tools",
    "expect_tool",
    "ToolExpectations",
    "ToolInterceptor",
    "RecordingInterceptor",
    "GenericToolAdapter",
    "AnthropicAdapter",
    "AutoAdapter",
    "parse_tool_calls",
    "simulate_tool",
    "tool_response_suite",
    "chaos_simulator",
    "ToolSimulator",
    "ScenarioType",
    "SearchScenarios",
    "CRUDScenarios",
    "APIScenarios",

    # Legacy assertion-based testing
    "ConversationTest",
    "UserTurn",
    "AgentAssertion",
    "agent_test",
    "TestResult",
    "load_test_file",
    "run_test_from_yaml",
    "AssertionResult",
    "BaseAssertion",
    "ContainsAssertion",
    "ExcludesAssertion",
    "MaxLengthAssertion",
    "MinLengthAssertion",
    "SentimentAssertion",
    "JsonValidAssertion",
    "RegexAssertion",
    "ToolUsageAssertion",
    "AllOfAssertion",
    "AnyOfAssertion",

    # Reporting
    "TestSuiteReport",
    "export_report",
]