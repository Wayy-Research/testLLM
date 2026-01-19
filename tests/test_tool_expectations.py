"""
Tests for Tool Expectation DSL
"""

import pytest
from testllm.tool_testing import (
    expect_tools,
    expect_tool,
    ToolExpectations,
    ToolExpectationBuilder,
    SingleToolExpectation,
    ToolCall,
    ToolSchema,
    ArgumentMatchMode,
    CallCountMode,
)


class TestToolExpectationBuilder:
    """Tests for the ToolExpectationBuilder class"""

    def test_basic_builder(self):
        """Test basic builder creation"""
        builder = ToolExpectationBuilder("search_flights")
        expectation = builder.build()

        assert expectation.tool_name == "search_flights"
        assert expectation.call_count == 1
        assert expectation.call_count_mode == CallCountMode.AT_LEAST

    def test_with_arguments(self):
        """Test exact argument matching"""
        expectation = (
            expect_tool("search_flights")
            .with_arguments(destination="NYC", date="2024-01-15")
            .build()
        )

        assert expectation.argument_match_mode == ArgumentMatchMode.EXACT
        assert expectation.argument_matcher == {"destination": "NYC", "date": "2024-01-15"}

    def test_with_arguments_containing(self):
        """Test partial argument matching"""
        expectation = (
            expect_tool("search_flights")
            .with_arguments_containing(destination="NYC")
            .build()
        )

        assert expectation.argument_match_mode == ArgumentMatchMode.CONTAINS
        assert expectation.argument_matcher == {"destination": "NYC"}

    def test_with_arguments_matching(self):
        """Test custom argument matcher"""
        def custom_matcher(args):
            return args.get("price", 0) < 500

        expectation = (
            expect_tool("search_flights")
            .with_arguments_matching(custom_matcher)
            .build()
        )

        assert expectation.argument_match_mode == ArgumentMatchMode.CUSTOM
        assert callable(expectation.argument_matcher)

    def test_returning(self):
        """Test mock response configuration"""
        mock_data = {"flights": [{"id": "F1", "price": 299}]}
        expectation = (
            expect_tool("search_flights")
            .returning(mock_data)
            .build()
        )

        assert expectation.mock_response == mock_data

    def test_call_count_times(self):
        """Test exact call count"""
        expectation = (
            expect_tool("search_flights")
            .times(2)
            .build()
        )

        assert expectation.call_count == 2
        assert expectation.call_count_mode == CallCountMode.EXACTLY

    def test_call_count_at_least(self):
        """Test at least call count"""
        expectation = (
            expect_tool("search_flights")
            .at_least(1)
            .build()
        )

        assert expectation.call_count == 1
        assert expectation.call_count_mode == CallCountMode.AT_LEAST

    def test_call_count_at_most(self):
        """Test at most call count"""
        expectation = (
            expect_tool("search_flights")
            .at_most(3)
            .build()
        )

        assert expectation.call_count == 3
        assert expectation.call_count_mode == CallCountMode.AT_MOST

    def test_call_count_between(self):
        """Test between call count"""
        expectation = (
            expect_tool("search_flights")
            .between(1, 5)
            .build()
        )

        assert expectation.call_count == 1
        assert expectation.call_count_max == 5
        assert expectation.call_count_mode == CallCountMode.BETWEEN

    def test_never(self):
        """Test never called expectation"""
        expectation = (
            expect_tool("dangerous_operation")
            .never()
            .build()
        )

        assert expectation.call_count == 0
        assert expectation.call_count_mode == CallCountMode.EXACTLY


class TestSingleToolExpectation:
    """Tests for SingleToolExpectation matching logic"""

    def test_matches_call_name(self):
        """Test matching by tool name"""
        expectation = SingleToolExpectation(tool_name="search_flights")
        call = ToolCall(tool_name="search_flights")

        assert expectation.matches_call(call) is True

    def test_no_match_different_name(self):
        """Test no match for different tool name"""
        expectation = SingleToolExpectation(tool_name="search_flights")
        call = ToolCall(tool_name="book_flight")

        assert expectation.matches_call(call) is False

    def test_matches_exact_arguments(self):
        """Test exact argument matching"""
        expectation = SingleToolExpectation(
            tool_name="search_flights",
            argument_matcher={"destination": "NYC"},
            argument_match_mode=ArgumentMatchMode.EXACT
        )

        matching_call = ToolCall(tool_name="search_flights", arguments={"destination": "NYC"})
        non_matching_call = ToolCall(tool_name="search_flights", arguments={"destination": "NYC", "extra": "param"})

        assert expectation.matches_call(matching_call) is True
        assert expectation.matches_call(non_matching_call) is False

    def test_matches_contains_arguments(self):
        """Test partial argument matching"""
        expectation = SingleToolExpectation(
            tool_name="search_flights",
            argument_matcher={"destination": "NYC"},
            argument_match_mode=ArgumentMatchMode.CONTAINS
        )

        call_with_extra = ToolCall(
            tool_name="search_flights",
            arguments={"destination": "NYC", "date": "2024-01-15"}
        )

        assert expectation.matches_call(call_with_extra) is True

    def test_validate_call_count_exactly(self):
        """Test exact call count validation"""
        expectation = SingleToolExpectation(
            tool_name="search_flights",
            call_count=2,
            call_count_mode=CallCountMode.EXACTLY
        )

        result_pass = expectation.validate_call_count(2)
        result_fail = expectation.validate_call_count(1)

        assert result_pass.passed is True
        assert result_fail.passed is False

    def test_validate_call_count_at_least(self):
        """Test at least call count validation"""
        expectation = SingleToolExpectation(
            tool_name="search_flights",
            call_count=1,
            call_count_mode=CallCountMode.AT_LEAST
        )

        assert expectation.validate_call_count(1).passed is True
        assert expectation.validate_call_count(5).passed is True
        assert expectation.validate_call_count(0).passed is False

    def test_validate_call_count_between(self):
        """Test between call count validation"""
        expectation = SingleToolExpectation(
            tool_name="search_flights",
            call_count=1,
            call_count_max=3,
            call_count_mode=CallCountMode.BETWEEN
        )

        assert expectation.validate_call_count(1).passed is True
        assert expectation.validate_call_count(2).passed is True
        assert expectation.validate_call_count(3).passed is True
        assert expectation.validate_call_count(0).passed is False
        assert expectation.validate_call_count(4).passed is False


class TestToolExpectations:
    """Tests for the main ToolExpectations class"""

    def test_expect_tools_factory(self):
        """Test factory function"""
        expectations = expect_tools()
        assert isinstance(expectations, ToolExpectations)

    def test_fluent_chain(self):
        """Test fluent method chaining"""
        expectations = (
            expect_tools()
            .expect_call("search_flights")
            .with_arguments_containing(destination="NYC")
            .returning({"flights": []})
            .times(1)
            .expect_call("book_flight")
            .with_arguments_containing(flight_id="F1")
        )

        built_expectations = expectations.get_expectations()
        assert len(built_expectations) == 2
        assert built_expectations[0].tool_name == "search_flights"
        assert built_expectations[1].tool_name == "book_flight"

    def test_verify_success(self):
        """Test successful verification"""
        expectations = (
            expect_tools()
            .expect_call("search_flights")
            .at_least(1)
        )

        calls = [
            ToolCall(tool_name="search_flights", arguments={"destination": "NYC"})
        ]

        result = expectations.verify(calls)
        assert result.all_passed is True
        assert result.passed_count == 1

    def test_verify_failure_missing_call(self):
        """Test verification failure when call is missing"""
        expectations = (
            expect_tools()
            .expect_call("search_flights")
            .times(1)
        )

        calls = []  # No calls made

        result = expectations.verify(calls)
        assert result.all_passed is False
        assert result.failed_count > 0

    def test_verify_sequence(self):
        """Test sequence verification"""
        expectations = (
            expect_tools()
            .expect_sequence("search_flights", "select_flight", "book_flight")
        )

        # Correct sequence
        calls = [
            ToolCall(tool_name="search_flights"),
            ToolCall(tool_name="select_flight"),
            ToolCall(tool_name="book_flight"),
        ]

        result = expectations.verify(calls)
        assert result.all_passed is True

    def test_verify_sequence_with_extra_calls(self):
        """Test sequence verification allows extra calls in between"""
        expectations = (
            expect_tools()
            .expect_sequence("search", "book")
        )

        # Sequence with extra call in between
        calls = [
            ToolCall(tool_name="search"),
            ToolCall(tool_name="validate"),  # Extra call
            ToolCall(tool_name="book"),
        ]

        result = expectations.verify(calls)
        assert result.all_passed is True

    def test_verify_strict_sequence(self):
        """Test strict sequence verification"""
        expectations = (
            expect_tools()
            .expect_sequence("search", "book")
            .strict_sequence()
        )

        # Strict sequence - no extra calls allowed
        correct_calls = [
            ToolCall(tool_name="search"),
            ToolCall(tool_name="book"),
        ]

        wrong_calls = [
            ToolCall(tool_name="search"),
            ToolCall(tool_name="validate"),
            ToolCall(tool_name="book"),
        ]

        assert expectations.verify(correct_calls).all_passed is True
        # Strict sequence with extra call in between should fail
        result = expectations.verify(wrong_calls)
        # Check that the sequence wasn't found in strict mode
        sequence_results = [r for r in result.results if r.expectation_type == "sequence"]
        assert any(not r.passed for r in sequence_results)

    def test_no_unexpected_calls(self):
        """Test no unexpected calls validation"""
        expectations = (
            expect_tools()
            .expect_call("search_flights")
            .at_least(1)
            .no_unexpected_calls()
        )

        # Only expected call
        valid_calls = [ToolCall(tool_name="search_flights")]
        result = expectations.verify(valid_calls)
        assert result.all_passed is True

        # Unexpected call
        invalid_calls = [
            ToolCall(tool_name="search_flights"),
            ToolCall(tool_name="unexpected_tool"),
        ]
        result = expectations.verify(invalid_calls)
        assert result.all_passed is False

    def test_get_mock_responses(self):
        """Test getting mock responses"""
        expectations = (
            expect_tools()
            .expect_call("search_flights")
            .returning({"flights": []})
            .expect_call("book_flight")
            .returning({"booking_id": "B1"})
            .expect_call("no_mock")
            # No returning() for this one
        )

        mocks = expectations.get_mock_responses()
        assert "search_flights" in mocks
        assert "book_flight" in mocks
        assert "no_mock" not in mocks


class TestToolSchema:
    """Tests for ToolSchema validation"""

    def test_basic_validation(self):
        """Test basic schema validation"""
        schema = ToolSchema(
            name="search_flights",
            parameters={
                "properties": {
                    "destination": {"type": "string"},
                    "max_price": {"type": "number"}
                }
            },
            required=["destination"]
        )

        # Valid arguments
        valid_result = schema.validate({"destination": "NYC", "max_price": 500})
        assert valid_result.valid is True

        # Missing required parameter
        invalid_result = schema.validate({"max_price": 500})
        assert invalid_result.valid is False
        assert "Missing required parameter: destination" in invalid_result.errors

    def test_type_validation(self):
        """Test type checking in schema validation"""
        schema = ToolSchema(
            name="search_flights",
            parameters={
                "properties": {
                    "destination": {"type": "string"},
                    "max_price": {"type": "number"}
                }
            }
        )

        # Correct types
        valid_result = schema.validate({"destination": "NYC", "max_price": 500})
        assert valid_result.valid is True

        # Wrong type
        invalid_result = schema.validate({"destination": 123, "max_price": "not a number"})
        assert invalid_result.valid is False
        assert len(invalid_result.errors) == 2


class TestArgumentMatching:
    """Tests for argument matching edge cases"""

    def test_nested_dict_matching(self):
        """Test matching nested dictionaries"""
        expectation = SingleToolExpectation(
            tool_name="complex_tool",
            argument_matcher={"config": {"nested": "value"}},
            argument_match_mode=ArgumentMatchMode.CONTAINS
        )

        call = ToolCall(
            tool_name="complex_tool",
            arguments={"config": {"nested": "value", "extra": "data"}, "other": "param"}
        )

        assert expectation.matches_call(call) is True

    def test_regex_pattern_matching(self):
        """Test regex pattern matching in arguments"""
        expectation = SingleToolExpectation(
            tool_name="search",
            argument_matcher={"query": "regex:^flight.*"},
            argument_match_mode=ArgumentMatchMode.CONTAINS
        )

        matching_call = ToolCall(tool_name="search", arguments={"query": "flight to NYC"})
        non_matching_call = ToolCall(tool_name="search", arguments={"query": "hotel in NYC"})

        assert expectation.matches_call(matching_call) is True
        assert expectation.matches_call(non_matching_call) is False

    def test_type_matching_in_arguments(self):
        """Test type matching in arguments"""
        expectation = SingleToolExpectation(
            tool_name="process",
            argument_matcher={"count": int, "name": str},
            argument_match_mode=ArgumentMatchMode.CONTAINS
        )

        valid_call = ToolCall(tool_name="process", arguments={"count": 5, "name": "test"})
        invalid_call = ToolCall(tool_name="process", arguments={"count": "five", "name": "test"})

        assert expectation.matches_call(valid_call) is True
        assert expectation.matches_call(invalid_call) is False
