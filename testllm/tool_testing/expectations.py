"""
Tool Expectation DSL - A declarative API for specifying tool expectations.

This module provides a fluent builder for tool expectations that integrates
naturally with ConversationFlow and other testing components.

Example usage:
    from testllm.tool_testing import expect_tools

    expectations = (
        expect_tools()
        .expect_call("search_flights")
        .with_arguments_containing(destination="NYC")
        .returning({"flights": [{"id": "F1", "price": 299}]})
        .times(1)
    )
"""

import re
import json
from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime

from .types import (
    ToolCall,
    ToolResponse,
    ToolSchema,
    ValidationResult,
    ExpectationResult,
    ToolExpectationSummary,
    ArgumentMatchMode,
    CallCountMode,
)


@dataclass
class SingleToolExpectation:
    """Expectation for a single tool call"""
    tool_name: str
    argument_matcher: Optional[Union[Dict[str, Any], Callable]] = None
    argument_match_mode: ArgumentMatchMode = ArgumentMatchMode.CONTAINS
    mock_response: Optional[Any] = None
    mock_response_generator: Optional[Callable[[ToolCall], Any]] = None
    call_count: int = 1
    call_count_mode: CallCountMode = CallCountMode.AT_LEAST
    call_count_max: Optional[int] = None
    schema: Optional[ToolSchema] = None
    custom_validator: Optional[Callable[[ToolCall], ValidationResult]] = None
    description: str = ""

    def matches_call(self, call: ToolCall) -> bool:
        """Check if a tool call matches this expectation"""
        if call.tool_name != self.tool_name:
            return False

        if self.argument_matcher is None:
            return True

        return self._match_arguments(call.arguments)

    def _match_arguments(self, actual_args: Dict[str, Any]) -> bool:
        """Match arguments based on the configured mode"""
        if self.argument_matcher is None:
            return True

        if callable(self.argument_matcher):
            return self.argument_matcher(actual_args)

        if self.argument_match_mode == ArgumentMatchMode.EXACT:
            return actual_args == self.argument_matcher

        elif self.argument_match_mode == ArgumentMatchMode.CONTAINS:
            return self._arguments_contain(actual_args, self.argument_matcher)

        elif self.argument_match_mode == ArgumentMatchMode.SCHEMA:
            if self.schema:
                result = self.schema.validate(actual_args)
                return result.valid
            return True

        elif self.argument_match_mode == ArgumentMatchMode.CUSTOM:
            if callable(self.argument_matcher):
                return self.argument_matcher(actual_args)
            return True

        return True

    def _arguments_contain(self, actual: Dict[str, Any], expected: Dict[str, Any]) -> bool:
        """Check if actual arguments contain all expected key-value pairs"""
        for key, expected_value in expected.items():
            if key not in actual:
                return False
            actual_value = actual[key]

            # Handle nested dicts
            if isinstance(expected_value, dict) and isinstance(actual_value, dict):
                if not self._arguments_contain(actual_value, expected_value):
                    return False
            # Handle regex patterns
            elif isinstance(expected_value, str) and expected_value.startswith("regex:"):
                pattern = expected_value[6:]
                if not re.match(pattern, str(actual_value)):
                    return False
            # Handle type checks
            elif isinstance(expected_value, type):
                if not isinstance(actual_value, expected_value):
                    return False
            # Direct comparison
            elif actual_value != expected_value:
                return False

        return True

    def validate_call_count(self, actual_count: int) -> ExpectationResult:
        """Validate the number of calls matches the expectation"""
        if self.call_count_mode == CallCountMode.EXACTLY:
            passed = actual_count == self.call_count
            message = f"Expected exactly {self.call_count} calls, got {actual_count}"

        elif self.call_count_mode == CallCountMode.AT_LEAST:
            passed = actual_count >= self.call_count
            message = f"Expected at least {self.call_count} calls, got {actual_count}"

        elif self.call_count_mode == CallCountMode.AT_MOST:
            passed = actual_count <= self.call_count
            message = f"Expected at most {self.call_count} calls, got {actual_count}"

        elif self.call_count_mode == CallCountMode.BETWEEN:
            min_count = self.call_count
            max_count = self.call_count_max or self.call_count
            passed = min_count <= actual_count <= max_count
            message = f"Expected between {min_count} and {max_count} calls, got {actual_count}"

        else:
            passed = True
            message = ""

        return ExpectationResult(
            expectation_type="call_count",
            passed=passed,
            tool_name=self.tool_name,
            expected=self.call_count,
            actual=actual_count,
            message="" if passed else message
        )

    def get_mock_response(self, call: ToolCall) -> Optional[ToolResponse]:
        """Get the mock response for this tool call"""
        if self.mock_response_generator:
            result = self.mock_response_generator(call)
            return ToolResponse(
                tool_name=self.tool_name,
                result=result,
                success=True
            )

        if self.mock_response is not None:
            return ToolResponse(
                tool_name=self.tool_name,
                result=self.mock_response,
                success=True
            )

        return None


class ToolExpectationBuilder:
    """
    Fluent builder for creating tool expectations.

    Example:
        expectation = (
            ToolExpectationBuilder("search_flights")
            .with_arguments_containing(destination="NYC")
            .returning({"flights": [...]})
            .times(1)
            .build()
        )
    """

    def __init__(self, tool_name: str):
        self._tool_name = tool_name
        self._argument_matcher: Optional[Union[Dict[str, Any], Callable]] = None
        self._argument_match_mode = ArgumentMatchMode.CONTAINS
        self._mock_response: Optional[Any] = None
        self._mock_response_generator: Optional[Callable[[ToolCall], Any]] = None
        self._call_count = 1
        self._call_count_mode = CallCountMode.AT_LEAST
        self._call_count_max: Optional[int] = None
        self._schema: Optional[ToolSchema] = None
        self._custom_validator: Optional[Callable[[ToolCall], ValidationResult]] = None
        self._description = ""

    def with_arguments(self, **kwargs) -> 'ToolExpectationBuilder':
        """Expect exact argument match"""
        self._argument_matcher = kwargs
        self._argument_match_mode = ArgumentMatchMode.EXACT
        return self

    def with_arguments_containing(self, **kwargs) -> 'ToolExpectationBuilder':
        """Expect arguments to contain these key-value pairs"""
        self._argument_matcher = kwargs
        self._argument_match_mode = ArgumentMatchMode.CONTAINS
        return self

    def with_arguments_matching(self, matcher: Callable[[Dict[str, Any]], bool]) -> 'ToolExpectationBuilder':
        """Expect arguments to pass a custom matcher function"""
        self._argument_matcher = matcher
        self._argument_match_mode = ArgumentMatchMode.CUSTOM
        return self

    def with_schema(self, schema: Union[Dict[str, Any], ToolSchema]) -> 'ToolExpectationBuilder':
        """Validate arguments against a JSON schema"""
        if isinstance(schema, dict):
            self._schema = ToolSchema(
                name=self._tool_name,
                parameters=schema,
                required=schema.get("required", [])
            )
        else:
            self._schema = schema
        self._argument_match_mode = ArgumentMatchMode.SCHEMA
        return self

    def returning(self, response: Any) -> 'ToolExpectationBuilder':
        """Configure a mock response for this tool"""
        self._mock_response = response
        return self

    def returning_dynamic(self, generator: Callable[[ToolCall], Any]) -> 'ToolExpectationBuilder':
        """Configure a dynamic mock response generator"""
        self._mock_response_generator = generator
        return self

    def times(self, count: int) -> 'ToolExpectationBuilder':
        """Expect exactly this many calls"""
        self._call_count = count
        self._call_count_mode = CallCountMode.EXACTLY
        return self

    def at_least(self, count: int) -> 'ToolExpectationBuilder':
        """Expect at least this many calls"""
        self._call_count = count
        self._call_count_mode = CallCountMode.AT_LEAST
        return self

    def at_most(self, count: int) -> 'ToolExpectationBuilder':
        """Expect at most this many calls"""
        self._call_count = count
        self._call_count_mode = CallCountMode.AT_MOST
        return self

    def between(self, min_count: int, max_count: int) -> 'ToolExpectationBuilder':
        """Expect between min and max calls (inclusive)"""
        self._call_count = min_count
        self._call_count_max = max_count
        self._call_count_mode = CallCountMode.BETWEEN
        return self

    def never(self) -> 'ToolExpectationBuilder':
        """Expect this tool to never be called"""
        self._call_count = 0
        self._call_count_mode = CallCountMode.EXACTLY
        return self

    def with_validator(self, validator: Callable[[ToolCall], ValidationResult]) -> 'ToolExpectationBuilder':
        """Add a custom validator function"""
        self._custom_validator = validator
        return self

    def described_as(self, description: str) -> 'ToolExpectationBuilder':
        """Add a description for this expectation"""
        self._description = description
        return self

    def build(self) -> SingleToolExpectation:
        """Build the final tool expectation"""
        return SingleToolExpectation(
            tool_name=self._tool_name,
            argument_matcher=self._argument_matcher,
            argument_match_mode=self._argument_match_mode,
            mock_response=self._mock_response,
            mock_response_generator=self._mock_response_generator,
            call_count=self._call_count,
            call_count_mode=self._call_count_mode,
            call_count_max=self._call_count_max,
            schema=self._schema,
            custom_validator=self._custom_validator,
            description=self._description
        )


class ToolExpectations:
    """
    Collection of tool expectations for a test step.

    This is the main entry point for the tool expectation DSL.

    Example:
        expectations = (
            expect_tools()
            .expect_call("search_flights")
            .with_arguments_containing(destination="NYC")
            .returning({"flights": [{"id": "F1", "price": 299}]})
            .expect_call("book_flight")
            .with_arguments_containing(flight_id="F1")
        )
    """

    def __init__(self):
        self._expectations: List[SingleToolExpectation] = []
        self._current_builder: Optional[ToolExpectationBuilder] = None
        self._sequence_expectations: List[List[str]] = []
        self._strict_sequence: bool = False
        self._no_unexpected_calls: bool = False

    def _finalize_current(self) -> None:
        """Finalize the current builder if any"""
        if self._current_builder:
            self._expectations.append(self._current_builder.build())
            self._current_builder = None

    def expect_call(self, tool_name: str) -> 'ToolExpectations':
        """Start expecting a call to the specified tool"""
        self._finalize_current()
        self._current_builder = ToolExpectationBuilder(tool_name)
        return self

    def with_arguments(self, **kwargs) -> 'ToolExpectations':
        """Expect exact argument match"""
        if self._current_builder:
            self._current_builder.with_arguments(**kwargs)
        return self

    def with_arguments_containing(self, **kwargs) -> 'ToolExpectations':
        """Expect arguments to contain these key-value pairs"""
        if self._current_builder:
            self._current_builder.with_arguments_containing(**kwargs)
        return self

    def with_arguments_matching(self, matcher: Callable[[Dict[str, Any]], bool]) -> 'ToolExpectations':
        """Expect arguments to pass a custom matcher function"""
        if self._current_builder:
            self._current_builder.with_arguments_matching(matcher)
        return self

    def with_schema(self, schema: Union[Dict[str, Any], ToolSchema]) -> 'ToolExpectations':
        """Validate arguments against a JSON schema"""
        if self._current_builder:
            self._current_builder.with_schema(schema)
        return self

    def returning(self, response: Any) -> 'ToolExpectations':
        """Configure a mock response for this tool"""
        if self._current_builder:
            self._current_builder.returning(response)
        return self

    def returning_dynamic(self, generator: Callable[[ToolCall], Any]) -> 'ToolExpectations':
        """Configure a dynamic mock response generator"""
        if self._current_builder:
            self._current_builder.returning_dynamic(generator)
        return self

    def times(self, count: int) -> 'ToolExpectations':
        """Expect exactly this many calls"""
        if self._current_builder:
            self._current_builder.times(count)
        return self

    def at_least(self, count: int) -> 'ToolExpectations':
        """Expect at least this many calls"""
        if self._current_builder:
            self._current_builder.at_least(count)
        return self

    def at_most(self, count: int) -> 'ToolExpectations':
        """Expect at most this many calls"""
        if self._current_builder:
            self._current_builder.at_most(count)
        return self

    def between(self, min_count: int, max_count: int) -> 'ToolExpectations':
        """Expect between min and max calls (inclusive)"""
        if self._current_builder:
            self._current_builder.between(min_count, max_count)
        return self

    def never(self) -> 'ToolExpectations':
        """Expect this tool to never be called"""
        if self._current_builder:
            self._current_builder.never()
        return self

    def described_as(self, description: str) -> 'ToolExpectations':
        """Add a description for this expectation"""
        if self._current_builder:
            self._current_builder.described_as(description)
        return self

    def expect_sequence(self, *tool_names: str) -> 'ToolExpectations':
        """Expect tools to be called in the specified sequence"""
        self._finalize_current()
        self._sequence_expectations.append(list(tool_names))
        return self

    def strict_sequence(self) -> 'ToolExpectations':
        """Enforce strict sequencing (no other calls between sequence elements)"""
        self._strict_sequence = True
        return self

    def no_unexpected_calls(self) -> 'ToolExpectations':
        """Fail if any unexpected tool calls are made"""
        self._finalize_current()
        self._no_unexpected_calls = True
        return self

    def verify(self, tool_calls: List[ToolCall]) -> ToolExpectationSummary:
        """
        Verify that the tool calls match all expectations.

        Args:
            tool_calls: List of actual tool calls made

        Returns:
            ToolExpectationSummary with verification results
        """
        self._finalize_current()
        results: List[ExpectationResult] = []

        # Group calls by tool name
        calls_by_tool: Dict[str, List[ToolCall]] = {}
        for call in tool_calls:
            if call.tool_name not in calls_by_tool:
                calls_by_tool[call.tool_name] = []
            calls_by_tool[call.tool_name].append(call)

        # Verify each expectation
        for expectation in self._expectations:
            matching_calls = [
                call for call in calls_by_tool.get(expectation.tool_name, [])
                if expectation.matches_call(call)
            ]

            # Check call count
            count_result = expectation.validate_call_count(len(matching_calls))
            results.append(count_result)

            # Check schema validation if configured
            if expectation.schema:
                for call in matching_calls:
                    validation = expectation.schema.validate(call.arguments)
                    results.append(ExpectationResult(
                        expectation_type="schema_validation",
                        passed=validation.valid,
                        tool_name=expectation.tool_name,
                        expected="valid schema",
                        actual=call.arguments,
                        message="; ".join(validation.errors) if not validation.valid else ""
                    ))

            # Run custom validator if configured
            if expectation.custom_validator:
                for call in matching_calls:
                    validation = expectation.custom_validator(call)
                    results.append(ExpectationResult(
                        expectation_type="custom_validation",
                        passed=validation.valid,
                        tool_name=expectation.tool_name,
                        expected="custom validation pass",
                        actual=call.arguments,
                        message="; ".join(validation.errors) if not validation.valid else ""
                    ))

        # Verify sequences
        for sequence in self._sequence_expectations:
            sequence_result = self._verify_sequence(tool_calls, sequence)
            results.append(sequence_result)

        # Check for unexpected calls
        if self._no_unexpected_calls:
            expected_tools = {exp.tool_name for exp in self._expectations}
            for call in tool_calls:
                if call.tool_name not in expected_tools:
                    results.append(ExpectationResult(
                        expectation_type="no_unexpected",
                        passed=False,
                        tool_name=call.tool_name,
                        expected="no call",
                        actual="called",
                        message=f"Unexpected tool call: {call.tool_name}"
                    ))

        # Calculate summary
        passed_count = sum(1 for r in results if r.passed)
        failed_count = len(results) - passed_count

        return ToolExpectationSummary(
            all_passed=failed_count == 0,
            total_expectations=len(results),
            passed_count=passed_count,
            failed_count=failed_count,
            results=results,
            tool_calls_made=tool_calls
        )

    def _verify_sequence(self, tool_calls: List[ToolCall], expected_sequence: List[str]) -> ExpectationResult:
        """Verify that tools were called in the expected sequence"""
        actual_sequence = [call.tool_name for call in tool_calls]

        if self._strict_sequence:
            # Strict: sequence must appear exactly as specified
            expected_str = "->".join(expected_sequence)
            # Find subsequence in actual
            for i in range(len(actual_sequence) - len(expected_sequence) + 1):
                if actual_sequence[i:i + len(expected_sequence)] == expected_sequence:
                    return ExpectationResult(
                        expectation_type="sequence",
                        passed=True,
                        tool_name="sequence",
                        expected=expected_str,
                        actual="->".join(actual_sequence)
                    )
            return ExpectationResult(
                expectation_type="sequence",
                passed=False,
                tool_name="sequence",
                expected=expected_str,
                actual="->".join(actual_sequence),
                message=f"Expected strict sequence {expected_str} not found"
            )
        else:
            # Non-strict: tools must appear in order but other calls can be between
            seq_idx = 0
            for call in tool_calls:
                if seq_idx < len(expected_sequence) and call.tool_name == expected_sequence[seq_idx]:
                    seq_idx += 1

            passed = seq_idx == len(expected_sequence)
            return ExpectationResult(
                expectation_type="sequence",
                passed=passed,
                tool_name="sequence",
                expected="->".join(expected_sequence),
                actual="->".join(actual_sequence),
                message="" if passed else f"Expected sequence not found, matched {seq_idx}/{len(expected_sequence)}"
            )

    def get_mock_responses(self) -> Dict[str, SingleToolExpectation]:
        """Get all expectations that have mock responses configured"""
        self._finalize_current()
        return {
            exp.tool_name: exp
            for exp in self._expectations
            if exp.mock_response is not None or exp.mock_response_generator is not None
        }

    def get_expectations(self) -> List[SingleToolExpectation]:
        """Get all expectations"""
        self._finalize_current()
        return self._expectations.copy()


def expect_tools() -> ToolExpectations:
    """
    Factory function to create a new ToolExpectations builder.

    This is the main entry point for the tool expectation DSL.

    Example:
        expectations = (
            expect_tools()
            .expect_call("search_flights")
            .with_arguments_containing(destination="NYC")
            .returning({"flights": [{"id": "F1", "price": 299}]})
        )
    """
    return ToolExpectations()


def expect_tool(tool_name: str) -> ToolExpectationBuilder:
    """
    Factory function to create a single tool expectation builder.

    Example:
        expectation = (
            expect_tool("search_flights")
            .with_arguments_containing(destination="NYC")
            .build()
        )
    """
    return ToolExpectationBuilder(tool_name)
