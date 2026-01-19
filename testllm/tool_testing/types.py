"""
Shared types for tool testing functionality.

This module provides common data structures used across the tool testing framework.
"""

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Union
from enum import Enum
from datetime import datetime


class ToolCallStatus(Enum):
    """Status of a tool call"""
    PENDING = "pending"
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


class ArgumentMatchMode(Enum):
    """Mode for matching tool arguments"""
    EXACT = "exact"
    CONTAINS = "contains"
    SCHEMA = "schema"
    CUSTOM = "custom"


class CallCountMode(Enum):
    """Mode for counting tool calls"""
    EXACTLY = "exactly"
    AT_LEAST = "at_least"
    AT_MOST = "at_most"
    BETWEEN = "between"


@dataclass
class ToolCall:
    """Represents a single tool call made by an agent"""
    tool_name: str
    arguments: Dict[str, Any] = field(default_factory=dict)
    timestamp: Optional[datetime] = None
    call_id: Optional[str] = None
    status: ToolCallStatus = ToolCallStatus.PENDING
    duration_ms: Optional[float] = None
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "tool_name": self.tool_name,
            "arguments": self.arguments,
            "timestamp": self.timestamp.isoformat() if self.timestamp else None,
            "call_id": self.call_id,
            "status": self.status.value,
            "duration_ms": self.duration_ms,
            "error": self.error,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolCall':
        """Create from dictionary representation"""
        return cls(
            tool_name=data.get("tool_name", data.get("name", "")),
            arguments=data.get("arguments", data.get("input", {})),
            timestamp=datetime.fromisoformat(data["timestamp"]) if data.get("timestamp") else None,
            call_id=data.get("call_id", data.get("id")),
            status=ToolCallStatus(data.get("status", "pending")),
            duration_ms=data.get("duration_ms"),
            error=data.get("error"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ToolResponse:
    """Represents a response from a tool"""
    tool_name: str
    result: Any = None
    success: bool = True
    error_message: Optional[str] = None
    execution_time_ms: Optional[float] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "tool_name": self.tool_name,
            "result": self.result,
            "success": self.success,
            "error_message": self.error_message,
            "execution_time_ms": self.execution_time_ms,
            "metadata": self.metadata
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ToolResponse':
        """Create from dictionary representation"""
        return cls(
            tool_name=data.get("tool_name", ""),
            result=data.get("result", data.get("output")),
            success=data.get("success", True),
            error_message=data.get("error_message", data.get("error")),
            execution_time_ms=data.get("execution_time_ms"),
            metadata=data.get("metadata", {})
        )


@dataclass
class ToolSchema:
    """JSON Schema definition for a tool's parameters"""
    name: str
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    required: List[str] = field(default_factory=list)

    def validate(self, arguments: Dict[str, Any]) -> 'ValidationResult':
        """Validate arguments against this schema"""
        errors = []

        # Check required parameters
        for param in self.required:
            if param not in arguments:
                errors.append(f"Missing required parameter: {param}")

        # Check parameter types if specified
        properties = self.parameters.get("properties", {})
        for param_name, param_value in arguments.items():
            if param_name in properties:
                expected_type = properties[param_name].get("type")
                if expected_type and not self._check_type(param_value, expected_type):
                    errors.append(
                        f"Parameter '{param_name}' has wrong type: "
                        f"expected {expected_type}, got {type(param_value).__name__}"
                    )

        return ValidationResult(
            valid=len(errors) == 0,
            errors=errors
        )

    def _check_type(self, value: Any, expected_type: str) -> bool:
        """Check if value matches expected JSON Schema type"""
        type_map = {
            "string": str,
            "number": (int, float),
            "integer": int,
            "boolean": bool,
            "array": list,
            "object": dict,
            "null": type(None)
        }
        expected_python_type = type_map.get(expected_type)
        if expected_python_type is None:
            return True  # Unknown type, allow it
        return isinstance(value, expected_python_type)


@dataclass
class ValidationResult:
    """Result of a validation operation"""
    valid: bool
    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.valid


@dataclass
class ExpectationResult:
    """Result of checking a tool expectation"""
    expectation_type: str
    passed: bool
    tool_name: str
    expected: Any = None
    actual: Any = None
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "expectation_type": self.expectation_type,
            "passed": self.passed,
            "tool_name": self.tool_name,
            "expected": self.expected,
            "actual": self.actual,
            "message": self.message,
            "details": self.details
        }


@dataclass
class ToolExpectationSummary:
    """Summary of all tool expectations for a step"""
    all_passed: bool
    total_expectations: int
    passed_count: int
    failed_count: int
    results: List[ExpectationResult] = field(default_factory=list)
    tool_calls_made: List[ToolCall] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "all_passed": self.all_passed,
            "total_expectations": self.total_expectations,
            "passed_count": self.passed_count,
            "failed_count": self.failed_count,
            "results": [r.to_dict() for r in self.results],
            "tool_calls_made": [tc.to_dict() for tc in self.tool_calls_made]
        }


@dataclass
class InterceptedCall:
    """A tool call that was intercepted by the interceptor"""
    original_call: ToolCall
    intercepted_at: datetime = field(default_factory=datetime.now)
    mock_response: Optional[ToolResponse] = None
    was_mocked: bool = False
    validation_result: Optional[ValidationResult] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary representation"""
        return {
            "original_call": self.original_call.to_dict(),
            "intercepted_at": self.intercepted_at.isoformat(),
            "mock_response": self.mock_response.to_dict() if self.mock_response else None,
            "was_mocked": self.was_mocked,
            "validation_result": {
                "valid": self.validation_result.valid,
                "errors": self.validation_result.errors
            } if self.validation_result else None
        }


# Type aliases for convenience
ToolCallSequence = List[ToolCall]
ArgumentMatcher = Union[Dict[str, Any], callable]
ResponseGenerator = Union[Dict[str, Any], callable]
