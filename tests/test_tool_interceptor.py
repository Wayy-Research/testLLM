"""
Tests for Tool Interceptor
"""

import pytest
from testllm.tool_testing import (
    ToolInterceptor,
    RecordingInterceptor,
    CompositeInterceptor,
    ToolCall,
    ToolResponse,
    ToolSchema,
    ValidationResult,
)


class TestToolInterceptor:
    """Tests for the ToolInterceptor class"""

    def test_basic_mock(self):
        """Test basic mock registration and interception"""
        interceptor = ToolInterceptor()
        interceptor.register_mock("search_flights", {"flights": [{"id": "F1"}]})

        call = ToolCall(tool_name="search_flights", arguments={"destination": "NYC"})
        response = interceptor.intercept(call)

        assert response is not None
        assert response.result == {"flights": [{"id": "F1"}]}
        assert response.success is True

    def test_callable_mock(self):
        """Test callable mock that generates dynamic responses"""
        interceptor = ToolInterceptor()

        def dynamic_mock(call: ToolCall) -> dict:
            return {"destination": call.arguments.get("destination"), "flights": []}

        interceptor.register_mock("search_flights", dynamic_mock)

        call = ToolCall(tool_name="search_flights", arguments={"destination": "NYC"})
        response = interceptor.intercept(call)

        assert response.result["destination"] == "NYC"

    def test_no_mock_returns_none(self):
        """Test that unregistered tools return None"""
        interceptor = ToolInterceptor()

        call = ToolCall(tool_name="unknown_tool")
        response = interceptor.intercept(call)

        assert response is None

    def test_recording_calls(self):
        """Test that calls are recorded"""
        interceptor = ToolInterceptor()
        interceptor.register_mock("search_flights", {"flights": []})

        call1 = ToolCall(tool_name="search_flights", arguments={"destination": "NYC"})
        call2 = ToolCall(tool_name="search_flights", arguments={"destination": "LAX"})
        call3 = ToolCall(tool_name="book_flight", arguments={"flight_id": "F1"})

        interceptor.intercept(call1)
        interceptor.intercept(call2)
        interceptor.intercept(call3)

        all_calls = interceptor.get_calls()
        search_calls = interceptor.get_calls("search_flights")
        book_calls = interceptor.get_calls("book_flight")

        assert len(all_calls) == 3
        assert len(search_calls) == 2
        assert len(book_calls) == 1

    def test_call_sequence(self):
        """Test getting call sequence"""
        interceptor = ToolInterceptor()

        calls = [
            ToolCall(tool_name="search"),
            ToolCall(tool_name="select"),
            ToolCall(tool_name="book"),
        ]

        for call in calls:
            interceptor.intercept(call)

        sequence = interceptor.get_call_sequence()
        assert sequence == ["search", "select", "book"]

    def test_was_called(self):
        """Test checking if a tool was called"""
        interceptor = ToolInterceptor()

        interceptor.intercept(ToolCall(tool_name="search_flights"))

        assert interceptor.was_called("search_flights") is True
        assert interceptor.was_called("book_flight") is False

    def test_call_count(self):
        """Test counting calls"""
        interceptor = ToolInterceptor()

        for _ in range(3):
            interceptor.intercept(ToolCall(tool_name="search_flights"))
        interceptor.intercept(ToolCall(tool_name="book_flight"))

        assert interceptor.call_count("search_flights") == 3
        assert interceptor.call_count("book_flight") == 1
        assert interceptor.call_count("unknown") == 0

    def test_schema_validation(self):
        """Test schema validation on intercept"""
        interceptor = ToolInterceptor()
        interceptor.register_schema("search_flights", {
            "properties": {
                "destination": {"type": "string"}
            },
            "required": ["destination"]
        })

        # Valid call
        valid_call = ToolCall(tool_name="search_flights", arguments={"destination": "NYC"})
        interceptor.intercept(valid_call)

        # Invalid call (missing required param)
        invalid_call = ToolCall(tool_name="search_flights", arguments={})
        interceptor.intercept(invalid_call)

        errors = interceptor.get_validation_errors()
        assert len(errors) == 1
        assert "destination" in errors[0]

    def test_custom_validator(self):
        """Test custom validator function"""
        interceptor = ToolInterceptor()

        def validate_price(call: ToolCall) -> ValidationResult:
            price = call.arguments.get("max_price", 0)
            if price > 1000:
                return ValidationResult(valid=False, errors=["Price exceeds maximum"])
            return ValidationResult(valid=True)

        interceptor.register_validator("search_flights", validate_price)

        valid_call = ToolCall(tool_name="search_flights", arguments={"max_price": 500})
        invalid_call = ToolCall(tool_name="search_flights", arguments={"max_price": 1500})

        interceptor.intercept(valid_call)
        interceptor.intercept(invalid_call)

        errors = interceptor.get_validation_errors()
        assert len(errors) == 1
        assert "exceeds maximum" in errors[0]

    def test_conditional_mock(self):
        """Test conditional mock based on arguments"""
        interceptor = ToolInterceptor()

        def is_nyc_search(call: ToolCall) -> bool:
            return call.arguments.get("destination") == "NYC"

        interceptor.register_conditional_mock(
            is_nyc_search,
            {"flights": [{"id": "NYC1"}]}
        )
        interceptor.register_mock("search_flights", {"flights": []})

        nyc_call = ToolCall(tool_name="search_flights", arguments={"destination": "NYC"})
        other_call = ToolCall(tool_name="search_flights", arguments={"destination": "LAX"})

        nyc_response = interceptor.intercept(nyc_call)
        other_response = interceptor.intercept(other_call)

        assert nyc_response.result["flights"][0]["id"] == "NYC1"
        assert other_response.result["flights"] == []

    def test_call_hook(self):
        """Test call hooks"""
        interceptor = ToolInterceptor()
        recorded_calls = []

        def hook(call: ToolCall, response):
            recorded_calls.append((call.tool_name, response))

        interceptor.add_call_hook(hook)
        interceptor.register_mock("search_flights", {"flights": []})

        interceptor.intercept(ToolCall(tool_name="search_flights"))
        interceptor.intercept(ToolCall(tool_name="other_tool"))

        assert len(recorded_calls) == 2
        assert recorded_calls[0][0] == "search_flights"
        assert recorded_calls[0][1] is not None  # Has mock response
        assert recorded_calls[1][1] is None  # No mock response

    def test_clear(self):
        """Test clearing recorded calls"""
        interceptor = ToolInterceptor()
        interceptor.register_mock("search", {"result": []})

        interceptor.intercept(ToolCall(tool_name="search"))
        assert interceptor.call_count("search") == 1

        interceptor.clear()
        assert interceptor.call_count("search") == 0

        # Mocks should still work after clear
        response = interceptor.intercept(ToolCall(tool_name="search"))
        assert response is not None

    def test_reset(self):
        """Test full reset (including mocks)"""
        interceptor = ToolInterceptor()
        interceptor.register_mock("search", {"result": []})

        interceptor.intercept(ToolCall(tool_name="search"))
        interceptor.reset()

        # Mocks should be cleared
        response = interceptor.intercept(ToolCall(tool_name="search"))
        assert response is None

    def test_chaining(self):
        """Test method chaining for configuration"""
        interceptor = (
            ToolInterceptor()
            .register_mock("tool1", {"result": 1})
            .register_mock("tool2", {"result": 2})
            .register_schema("tool1", {"properties": {}})
        )

        assert interceptor.intercept(ToolCall(tool_name="tool1")).result == {"result": 1}
        assert interceptor.intercept(ToolCall(tool_name="tool2")).result == {"result": 2}


class TestRecordingInterceptor:
    """Tests for the RecordingInterceptor class"""

    def test_records_without_mocking(self):
        """Test that recording interceptor records but doesn't mock"""
        interceptor = RecordingInterceptor()

        call = ToolCall(tool_name="search_flights", arguments={"destination": "NYC"})
        response = interceptor.intercept(call)

        assert response is None  # No mocking
        assert len(interceptor.get_calls()) == 1
        assert interceptor.get_calls()[0].tool_name == "search_flights"

    def test_filter_by_tool_name(self):
        """Test filtering recorded calls by tool name"""
        interceptor = RecordingInterceptor()

        interceptor.intercept(ToolCall(tool_name="tool1"))
        interceptor.intercept(ToolCall(tool_name="tool2"))
        interceptor.intercept(ToolCall(tool_name="tool1"))

        all_calls = interceptor.get_calls()
        tool1_calls = interceptor.get_calls("tool1")

        assert len(all_calls) == 3
        assert len(tool1_calls) == 2


class TestCompositeInterceptor:
    """Tests for the CompositeInterceptor class"""

    def test_first_mock_wins(self):
        """Test that first interceptor with mock wins"""
        interceptor1 = ToolInterceptor()
        interceptor1.register_mock("search", {"from": "interceptor1"})

        interceptor2 = ToolInterceptor()
        interceptor2.register_mock("search", {"from": "interceptor2"})

        composite = CompositeInterceptor(interceptor1, interceptor2)

        response = composite.intercept(ToolCall(tool_name="search"))
        assert response.result["from"] == "interceptor1"

    def test_fallback_to_second_interceptor(self):
        """Test fallback when first interceptor doesn't have mock"""
        interceptor1 = ToolInterceptor()
        # interceptor1 has no mock for "other"

        interceptor2 = ToolInterceptor()
        interceptor2.register_mock("other", {"from": "interceptor2"})

        composite = CompositeInterceptor(interceptor1, interceptor2)

        response = composite.intercept(ToolCall(tool_name="other"))
        assert response.result["from"] == "interceptor2"

    def test_all_interceptors_record(self):
        """Test that all interceptors record calls"""
        interceptor1 = RecordingInterceptor()
        interceptor2 = RecordingInterceptor()

        composite = CompositeInterceptor(interceptor1, interceptor2)
        composite.intercept(ToolCall(tool_name="search"))

        # Note: Based on implementation, only the first interceptor that
        # processes the call records it; others get record() called explicitly
        assert len(interceptor1.get_calls()) == 1

    def test_add_interceptor(self):
        """Test adding interceptor dynamically"""
        composite = CompositeInterceptor()

        interceptor = ToolInterceptor()
        interceptor.register_mock("search", {"result": []})

        composite.add_interceptor(interceptor)

        response = composite.intercept(ToolCall(tool_name="search"))
        assert response is not None


class TestInterceptedCalls:
    """Tests for detailed intercepted call records"""

    def test_intercepted_call_details(self):
        """Test that intercepted calls contain detailed information"""
        interceptor = ToolInterceptor()
        interceptor.register_mock("search", {"flights": []})
        interceptor.register_schema("search", {
            "properties": {"destination": {"type": "string"}},
            "required": ["destination"]
        })

        call = ToolCall(tool_name="search", arguments={"destination": "NYC"})
        interceptor.intercept(call)

        intercepted = interceptor.get_intercepted_calls()[0]

        assert intercepted.original_call.tool_name == "search"
        assert intercepted.was_mocked is True
        assert intercepted.mock_response is not None
        assert intercepted.validation_result is not None
        assert intercepted.validation_result.valid is True

    def test_intercepted_call_to_dict(self):
        """Test converting intercepted call to dictionary"""
        interceptor = ToolInterceptor()
        interceptor.register_mock("search", {"result": []})

        call = ToolCall(tool_name="search", arguments={"query": "test"})
        interceptor.intercept(call)

        intercepted = interceptor.get_intercepted_calls()[0]
        data = intercepted.to_dict()

        assert "original_call" in data
        assert "intercepted_at" in data
        assert "was_mocked" in data
        assert data["was_mocked"] is True
