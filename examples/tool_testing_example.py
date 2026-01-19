"""
Tool Testing Examples for testLLM

This file demonstrates the three complementary approaches for testing tool usage
in LLM-based agents:

1. Tool Expectation DSL - Declarative API for specifying tool expectations
2. Tool Interceptor - Framework-agnostic tool call interception and mocking
3. Response Simulation - Test agent behavior under various tool scenarios
"""

# =============================================================================
# APPROACH 1: Tool Expectation DSL
# =============================================================================

from testllm import (
    conversation_flow,
    expect_tools,
    expect_tool,
    ToolExpectations,
)


def example_tool_expectation_dsl():
    """
    Demonstrates the Tool Expectation DSL for declaring expected tool behavior.
    This integrates naturally with ConversationFlow.
    """
    print("=" * 60)
    print("APPROACH 1: Tool Expectation DSL")
    print("=" * 60)

    # Example 1: Basic tool expectation
    basic_expectation = (
        expect_tools()
        .expect_call("search_flights")
        .with_arguments_containing(destination="NYC")
        .returning({"flights": [{"id": "F1", "price": 299}]})
    )
    print("\n1. Basic expectation created:")
    print(f"   Expects call to: search_flights")
    print(f"   With args containing: destination=NYC")

    # Example 2: Call count validation
    count_expectation = (
        expect_tools()
        .expect_call("search_flights")
        .times(1)  # Exactly once
        .expect_call("validate_booking")
        .at_least(1)  # At least once
        .expect_call("send_notification")
        .at_most(3)  # At most 3 times
    )
    print("\n2. Call count expectations:")
    print(f"   search_flights: exactly 1 time")
    print(f"   validate_booking: at least 1 time")
    print(f"   send_notification: at most 3 times")

    # Example 3: Sequence validation
    sequence_expectation = (
        expect_tools()
        .expect_sequence("search_flights", "select_flight", "book_flight")
    )
    print("\n3. Sequence expectation:")
    print("   search_flights -> select_flight -> book_flight")

    # Example 4: Using with ConversationFlow
    flow = (
        conversation_flow("flight_booking", "Test flight booking with tool expectations")
        .tool_step(
            "Find me flights from SFO to NYC next Monday",
            criteria=[
                "Should present flight options with prices",
                "Should include departure times"
            ],
            tool_expectations=(
                expect_tools()
                .expect_call("search_flights")
                .with_arguments_containing(origin="SFO", destination="NYC")
                .returning({
                    "flights": [
                        {"id": "F1", "price": 299, "departure": "10:00"},
                        {"id": "F2", "price": 349, "departure": "14:00"}
                    ]
                })
            )
        )
        .step(
            "Book flight F1",
            criteria=["Should confirm booking with confirmation number"]
        )
        .with_tool_expectations(
            expect_tools()
            .expect_call("book_flight")
            .with_arguments_containing(flight_id="F1")
            .times(1)
        )
    )
    print("\n4. ConversationFlow with tool expectations:")
    print(f"   Flow: {flow.flow_id}")
    print(f"   Steps: {len(flow.steps)}")


# =============================================================================
# APPROACH 2: Tool Interceptor
# =============================================================================

from testllm import (
    ToolInterceptor,
    RecordingInterceptor,
    InterceptedAgent,
    LocalAgent,
)
from testllm.tool_testing import ToolCall, ToolSchema


def example_tool_interceptor():
    """
    Demonstrates the Tool Interceptor for capturing and mocking tool calls.
    """
    print("\n" + "=" * 60)
    print("APPROACH 2: Tool Interceptor")
    print("=" * 60)

    # Example 1: Basic interception with mocking
    interceptor = ToolInterceptor()
    interceptor.register_mock("search_flights", {
        "flights": [
            {"id": "F1", "airline": "United", "price": 299},
            {"id": "F2", "airline": "Delta", "price": 349}
        ]
    })
    print("\n1. Basic mock registered for search_flights")

    # Simulate a tool call
    call = ToolCall(tool_name="search_flights", arguments={"destination": "NYC"})
    response = interceptor.intercept(call)
    print(f"   Mock response: {response.result}")

    # Example 2: Dynamic mock based on arguments
    def dynamic_search(call: ToolCall):
        dest = call.arguments.get("destination", "unknown")
        return {"flights": [], "searched_for": dest}

    interceptor.register_mock("dynamic_search", dynamic_search)
    print("\n2. Dynamic mock registered")

    call = ToolCall(tool_name="dynamic_search", arguments={"destination": "LAX"})
    response = interceptor.intercept(call)
    print(f"   Dynamic response: {response.result}")

    # Example 3: Schema validation
    interceptor.register_schema("book_flight", {
        "properties": {
            "flight_id": {"type": "string"},
            "passenger_name": {"type": "string"}
        },
        "required": ["flight_id", "passenger_name"]
    })
    print("\n3. Schema validation registered for book_flight")

    # Valid call
    valid_call = ToolCall(
        tool_name="book_flight",
        arguments={"flight_id": "F1", "passenger_name": "John Doe"}
    )
    interceptor.intercept(valid_call)
    print(f"   Valid call - errors: {interceptor.get_validation_errors()}")

    # Invalid call (missing required field)
    invalid_call = ToolCall(
        tool_name="book_flight",
        arguments={"flight_id": "F1"}  # Missing passenger_name
    )
    interceptor.intercept(invalid_call)
    print(f"   Invalid call - errors: {interceptor.get_validation_errors()}")

    # Example 4: Recording and analyzing calls
    print("\n4. Call recording and analysis:")
    print(f"   Total calls: {len(interceptor.get_calls())}")
    print(f"   Call sequence: {interceptor.get_call_sequence()}")
    print(f"   search_flights called: {interceptor.was_called('search_flights')}")
    print(f"   search_flights count: {interceptor.call_count('search_flights')}")


# =============================================================================
# APPROACH 3: Response Simulation Framework
# =============================================================================

from testllm import (
    simulate_tool,
    tool_response_suite,
    chaos_simulator,
    ScenarioType,
    SearchScenarios,
    APIScenarios,
)


def example_response_simulation():
    """
    Demonstrates the Response Simulation Framework for testing edge cases.
    """
    print("\n" + "=" * 60)
    print("APPROACH 3: Response Simulation Framework")
    print("=" * 60)

    # Example 1: Configure multiple scenarios for a tool
    flight_search = (
        simulate_tool("search_flights")
        .on_success({
            "flights": [
                {"id": "F1", "price": 299},
                {"id": "F2", "price": 349}
            ]
        })
        .on_failure("Flight search service unavailable")
        .on_timeout(30000)
        .on_empty({"flights": [], "message": "No flights found"})
        .on_rate_limited()
    )
    print("\n1. Tool simulator configured with 5 scenarios:")
    for scenario in flight_search.get_all_scenarios():
        print(f"   - {scenario.scenario_type.value}: {scenario.description or 'configured'}")

    # Example 2: Generate responses for different scenarios
    call = ToolCall(tool_name="search_flights", arguments={"destination": "NYC"})

    success_response = flight_search.get_scenario(ScenarioType.SUCCESS).generate_response(call)
    failure_response = flight_search.get_scenario(ScenarioType.FAILURE).generate_response(call)

    print("\n2. Generated responses:")
    print(f"   Success: {success_response.result}")
    print(f"   Failure: {failure_response.error_message}")

    # Example 3: Create a test suite for resilience testing
    suite = (
        tool_response_suite("flight_booking_resilience", "Test agent handles all scenarios")
        .add_simulator(flight_search)
        .test_all_scenarios(
            "search_flights",
            user_input="Find flights to NYC",
            criteria=[
                "Agent should handle the response gracefully",
                "Agent should not expose internal errors to user"
            ]
        )
    )
    print("\n3. Test suite created:")
    print(f"   Suite: {suite.suite_id}")
    print(f"   Test cases: {len(suite.get_test_cases())}")
    for tc in suite.get_test_cases():
        print(f"   - {tc.name}: {tc.scenario_type.value}")

    # Example 4: Pre-built scenarios
    print("\n4. Pre-built scenarios available:")
    weather_sim = APIScenarios.weather_api()
    print(f"   Weather API: {len(weather_sim.get_all_scenarios())} scenarios")

    hotel_sim = SearchScenarios.hotel_search()
    print(f"   Hotel Search: {len(hotel_sim.get_all_scenarios())} scenarios")

    # Example 5: Chaos testing
    chaos = (
        chaos_simulator(failure_rate=0.2)  # 20% chance of failure
        .add_failure_mode("search_flights", ScenarioType.TIMEOUT)
        .add_failure_mode("search_flights", ScenarioType.RATE_LIMITED)
        .set_success_response("search_flights", {"flights": [{"id": "F1"}]})
    )
    print("\n5. Chaos simulator configured:")
    print(f"   Failure rate: 20%")
    print(f"   Failure modes: TIMEOUT, RATE_LIMITED")


# =============================================================================
# ADAPTERS: Parsing tool calls from different frameworks
# =============================================================================

from testllm import (
    AnthropicAdapter,
    GenericToolAdapter,
    AutoAdapter,
    parse_tool_calls,
)


def example_adapters():
    """
    Demonstrates framework adapters for parsing tool calls from different formats.
    """
    print("\n" + "=" * 60)
    print("ADAPTERS: Framework-specific parsing")
    print("=" * 60)

    # Generic format (dict with tool_name/name)
    generic_call = {
        "tool_name": "search_flights",
        "arguments": {"destination": "NYC"},
        "id": "call_abc123"
    }
    parsed_generic = GenericToolAdapter.parse_call(generic_call)
    print(f"\n1. Generic format parsed:")
    print(f"   Tool: {parsed_generic.tool_name}")
    print(f"   Args: {parsed_generic.arguments}")

    # Anthropic format
    anthropic_call = {
        "type": "tool_use",
        "id": "toolu_123",
        "name": "search_flights",
        "input": {"destination": "NYC"}
    }
    parsed_anthropic = AnthropicAdapter.parse_call(anthropic_call)
    print(f"\n2. Anthropic format parsed:")
    print(f"   Tool: {parsed_anthropic.tool_name}")
    print(f"   Args: {parsed_anthropic.arguments}")

    # Auto-detection
    detected = AutoAdapter.detect_format(anthropic_call)
    print(f"\n3. Auto-detected format: {detected}")

    # Parse multiple calls
    calls = parse_tool_calls([generic_call])
    print(f"\n4. Batch parsing: {len(calls)} calls parsed")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    example_tool_expectation_dsl()
    example_tool_interceptor()
    example_response_simulation()
    example_adapters()

    print("\n" + "=" * 60)
    print("All examples completed!")
    print("=" * 60)
