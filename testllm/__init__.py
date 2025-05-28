"""
testLLM Framework
Testing Framework for LLM-Based Agents
"""

from .core import (
    AgentUnderTest,
    ApiAgent,
    LocalAgent,
    ConversationTest,
    UserTurn,
    AgentAssertion,
    agent_test,
    TestResult,
)

from .assertions import (
    AssertionResult,
    BaseAssertion,
    ContainsAssertion,
    ExcludesAssertion,
    MaxLengthAssertion,
    SentimentAssertion,
    JsonValidAssertion,
    ToolUsageAssertion,
)


from .reporting import (
    TestSuiteReport,
    export_report,
)

__version__ = "0.1.0"
__all__ = [
    "AgentUnderTest",
    "ApiAgent", 
    "LocalAgent",
    "ConversationTest",
    "UserTurn",
    "AgentAssertion",
    "agent_test",
    "TestResult",
    "AssertionResult",
    "BaseAssertion",
    "ContainsAssertion",
    "ExcludesAssertion",
    "MaxLengthAssertion",
    "SentimentAssertion",
    "JsonValidAssertion",
    "ToolUsageAssertion",
    "TestSuiteReport",
    "export_report",
]