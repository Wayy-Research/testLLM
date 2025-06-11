"""
testLLM Framework
Testing Framework for LLM-Based Agents
"""

from .core import (
    AgentUnderTest,
    ApiAgent,
    LocalAgent,
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

from .__version__ import __version__
__all__ = [
    # Core agent interfaces
    "AgentUnderTest",
    "ApiAgent", 
    "LocalAgent",
    
    # Primary semantic testing interface
    "SemanticTest",
    "SemanticTestResult",
    "SemanticTestCase", 
    "semantic_test",
    "pytest_semantic_test",
    
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