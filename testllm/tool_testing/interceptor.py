"""
Tool Interceptor - Framework-agnostic tool call interception and mocking.

This module provides an abstract base class for intercepting tool calls at various
levels (agent, API, etc.) and a concrete implementation for testing.

Example usage:
    from testllm.tool_testing import ToolInterceptor

    interceptor = ToolInterceptor()
    interceptor.register_mock("search_flights", {"flights": [...]})

    # Use with InterceptedAgent wrapper
    agent = InterceptedAgent(my_agent, interceptor)
"""

from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
import threading
import uuid

from .types import (
    ToolCall,
    ToolResponse,
    ToolSchema,
    ValidationResult,
    InterceptedCall,
    ToolCallStatus,
)


class BaseToolInterceptor(ABC):
    """
    Abstract base class for tool interceptors.

    Implementations can intercept tool calls at different levels:
    - Agent level (wrapping agent.send_message)
    - API level (HTTP request interception)
    - Custom protocol level
    """

    @abstractmethod
    def intercept(self, call: ToolCall) -> Optional[ToolResponse]:
        """
        Intercept a tool call and optionally return a mock response.

        Args:
            call: The tool call to intercept

        Returns:
            ToolResponse if mocked, None to let the call through
        """
        pass

    @abstractmethod
    def record(self, call: ToolCall, response: Optional[ToolResponse] = None) -> None:
        """
        Record a tool call for later verification.

        Args:
            call: The tool call that was made
            response: The response received (if any)
        """
        pass

    @abstractmethod
    def get_calls(self, tool_name: Optional[str] = None) -> List[ToolCall]:
        """
        Get recorded tool calls.

        Args:
            tool_name: Optional filter by tool name

        Returns:
            List of recorded tool calls
        """
        pass

    @abstractmethod
    def clear(self) -> None:
        """Clear all recorded calls and mocks"""
        pass


class ToolInterceptor(BaseToolInterceptor):
    """
    Concrete implementation of a tool interceptor for testing.

    Features:
    - Register mock responses for specific tools
    - Record all tool calls with timestamps
    - Validate tool calls against schemas
    - Support for conditional mocking
    - Thread-safe for concurrent testing
    """

    def __init__(self):
        self._mocks: Dict[str, Union[Any, Callable[[ToolCall], Any]]] = {}
        self._schemas: Dict[str, ToolSchema] = {}
        self._recorded_calls: List[InterceptedCall] = []
        self._validators: Dict[str, Callable[[ToolCall], ValidationResult]] = {}
        self._conditional_mocks: List[tuple[Callable[[ToolCall], bool], Any]] = []
        self._lock = threading.Lock()
        self._call_hooks: List[Callable[[ToolCall, Optional[ToolResponse]], None]] = []

    def register_mock(
        self,
        tool_name: str,
        response: Union[Any, Callable[[ToolCall], Any]]
    ) -> 'ToolInterceptor':
        """
        Register a mock response for a tool.

        Args:
            tool_name: Name of the tool to mock
            response: Static response or callable that generates response

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._mocks[tool_name] = response
        return self

    def register_schema(
        self,
        tool_name: str,
        schema: Union[Dict[str, Any], ToolSchema]
    ) -> 'ToolInterceptor':
        """
        Register a schema for validating tool call arguments.

        Args:
            tool_name: Name of the tool
            schema: JSON Schema dict or ToolSchema instance

        Returns:
            Self for method chaining
        """
        with self._lock:
            if isinstance(schema, dict):
                self._schemas[tool_name] = ToolSchema(
                    name=tool_name,
                    parameters=schema,
                    required=schema.get("required", [])
                )
            else:
                self._schemas[tool_name] = schema
        return self

    def register_validator(
        self,
        tool_name: str,
        validator: Callable[[ToolCall], ValidationResult]
    ) -> 'ToolInterceptor':
        """
        Register a custom validator for a tool.

        Args:
            tool_name: Name of the tool
            validator: Function that validates tool calls

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._validators[tool_name] = validator
        return self

    def register_conditional_mock(
        self,
        condition: Callable[[ToolCall], bool],
        response: Union[Any, Callable[[ToolCall], Any]]
    ) -> 'ToolInterceptor':
        """
        Register a mock that only activates when condition is met.

        Args:
            condition: Function that returns True if mock should activate
            response: Static response or callable that generates response

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._conditional_mocks.append((condition, response))
        return self

    def add_call_hook(
        self,
        hook: Callable[[ToolCall, Optional[ToolResponse]], None]
    ) -> 'ToolInterceptor':
        """
        Add a hook that gets called on every tool call.

        Useful for logging, debugging, or custom validation.

        Args:
            hook: Function called with (call, response) on every interception

        Returns:
            Self for method chaining
        """
        with self._lock:
            self._call_hooks.append(hook)
        return self

    def intercept(self, call: ToolCall) -> Optional[ToolResponse]:
        """
        Intercept a tool call and optionally return a mock response.

        Args:
            call: The tool call to intercept

        Returns:
            ToolResponse if mocked, None to let the call through
        """
        with self._lock:
            # Validate against schema if registered
            validation_result = None
            if call.tool_name in self._schemas:
                validation_result = self._schemas[call.tool_name].validate(call.arguments)

            # Run custom validator if registered
            if call.tool_name in self._validators:
                custom_validation = self._validators[call.tool_name](call)
                if validation_result:
                    validation_result.errors.extend(custom_validation.errors)
                    validation_result.valid = validation_result.valid and custom_validation.valid
                else:
                    validation_result = custom_validation

            # Check conditional mocks first
            for condition, response in self._conditional_mocks:
                if condition(call):
                    mock_response = self._create_response(call, response)
                    self._record_call(call, mock_response, True, validation_result)
                    return mock_response

            # Check direct mocks
            if call.tool_name in self._mocks:
                mock_response = self._create_response(call, self._mocks[call.tool_name])
                self._record_call(call, mock_response, True, validation_result)
                return mock_response

            # No mock, just record the call
            self._record_call(call, None, False, validation_result)
            return None

    def _create_response(self, call: ToolCall, response: Union[Any, Callable]) -> ToolResponse:
        """Create a ToolResponse from a mock value or callable"""
        if callable(response):
            result = response(call)
        else:
            result = response

        # Handle ToolResponse already being returned
        if isinstance(result, ToolResponse):
            return result

        return ToolResponse(
            tool_name=call.tool_name,
            result=result,
            success=True
        )

    def _record_call(
        self,
        call: ToolCall,
        response: Optional[ToolResponse],
        was_mocked: bool,
        validation_result: Optional[ValidationResult]
    ) -> None:
        """Record a tool call internally"""
        intercepted = InterceptedCall(
            original_call=call,
            intercepted_at=datetime.now(),
            mock_response=response,
            was_mocked=was_mocked,
            validation_result=validation_result
        )
        self._recorded_calls.append(intercepted)

        # Run hooks
        for hook in self._call_hooks:
            try:
                hook(call, response)
            except Exception:
                pass  # Ignore hook errors

    def record(self, call: ToolCall, response: Optional[ToolResponse] = None) -> None:
        """
        Manually record a tool call.

        Args:
            call: The tool call that was made
            response: The response received (if any)
        """
        with self._lock:
            self._record_call(call, response, False, None)

    def get_calls(self, tool_name: Optional[str] = None) -> List[ToolCall]:
        """
        Get recorded tool calls.

        Args:
            tool_name: Optional filter by tool name

        Returns:
            List of recorded tool calls
        """
        with self._lock:
            calls = [ic.original_call for ic in self._recorded_calls]
            if tool_name:
                calls = [c for c in calls if c.tool_name == tool_name]
            return calls

    def get_intercepted_calls(self, tool_name: Optional[str] = None) -> List[InterceptedCall]:
        """
        Get detailed intercepted call records.

        Args:
            tool_name: Optional filter by tool name

        Returns:
            List of InterceptedCall records
        """
        with self._lock:
            calls = self._recorded_calls.copy()
            if tool_name:
                calls = [c for c in calls if c.original_call.tool_name == tool_name]
            return calls

    def get_call_sequence(self) -> List[str]:
        """
        Get the sequence of tool names that were called.

        Returns:
            List of tool names in order of calling
        """
        with self._lock:
            return [ic.original_call.tool_name for ic in self._recorded_calls]

    def get_validation_errors(self, tool_name: Optional[str] = None) -> List[str]:
        """
        Get all validation errors that occurred.

        Args:
            tool_name: Optional filter by tool name

        Returns:
            List of error messages
        """
        with self._lock:
            errors = []
            for ic in self._recorded_calls:
                if tool_name and ic.original_call.tool_name != tool_name:
                    continue
                # Use 'is not None' because ValidationResult.__bool__ returns self.valid
                if ic.validation_result is not None and not ic.validation_result.valid:
                    errors.extend(ic.validation_result.errors)
            return errors

    def was_called(self, tool_name: str) -> bool:
        """Check if a tool was called at least once"""
        return len(self.get_calls(tool_name)) > 0

    def call_count(self, tool_name: str) -> int:
        """Get the number of times a tool was called"""
        return len(self.get_calls(tool_name))

    def clear(self) -> None:
        """Clear all recorded calls (but keep mocks)"""
        with self._lock:
            self._recorded_calls.clear()

    def reset(self) -> None:
        """Reset interceptor completely (clear calls and mocks)"""
        with self._lock:
            self._mocks.clear()
            self._schemas.clear()
            self._validators.clear()
            self._conditional_mocks.clear()
            self._recorded_calls.clear()
            self._call_hooks.clear()


class RecordingInterceptor(BaseToolInterceptor):
    """
    Simple interceptor that only records calls without mocking.

    Useful for observing tool call behavior without modifying it.
    """

    def __init__(self):
        self._recorded_calls: List[ToolCall] = []
        self._lock = threading.Lock()

    def intercept(self, call: ToolCall) -> Optional[ToolResponse]:
        """Record the call but don't mock it"""
        with self._lock:
            self._recorded_calls.append(call)
        return None

    def record(self, call: ToolCall, response: Optional[ToolResponse] = None) -> None:
        """Record a tool call"""
        with self._lock:
            self._recorded_calls.append(call)

    def get_calls(self, tool_name: Optional[str] = None) -> List[ToolCall]:
        """Get recorded tool calls"""
        with self._lock:
            calls = self._recorded_calls.copy()
            if tool_name:
                calls = [c for c in calls if c.tool_name == tool_name]
            return calls

    def clear(self) -> None:
        """Clear recorded calls"""
        with self._lock:
            self._recorded_calls.clear()


class CompositeInterceptor(BaseToolInterceptor):
    """
    Interceptor that chains multiple interceptors together.

    The first interceptor to return a mock response wins.
    All interceptors record the call.
    """

    def __init__(self, *interceptors: BaseToolInterceptor):
        self._interceptors = list(interceptors)

    def add_interceptor(self, interceptor: BaseToolInterceptor) -> 'CompositeInterceptor':
        """Add an interceptor to the chain"""
        self._interceptors.append(interceptor)
        return self

    def intercept(self, call: ToolCall) -> Optional[ToolResponse]:
        """Try each interceptor in order, return first mock response"""
        for interceptor in self._interceptors:
            response = interceptor.intercept(call)
            if response is not None:
                # Record in other interceptors too
                for other in self._interceptors:
                    if other != interceptor:
                        other.record(call, response)
                return response
        return None

    def record(self, call: ToolCall, response: Optional[ToolResponse] = None) -> None:
        """Record in all interceptors"""
        for interceptor in self._interceptors:
            interceptor.record(call, response)

    def get_calls(self, tool_name: Optional[str] = None) -> List[ToolCall]:
        """Get calls from the first interceptor"""
        if self._interceptors:
            return self._interceptors[0].get_calls(tool_name)
        return []

    def clear(self) -> None:
        """Clear all interceptors"""
        for interceptor in self._interceptors:
            interceptor.clear()
