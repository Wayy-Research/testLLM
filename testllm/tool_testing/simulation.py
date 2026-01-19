"""
Response Simulation Framework - Test agent behavior under various tool scenarios.

This module provides tools for simulating different tool response scenarios
(success, failure, timeout, empty, etc.) to test agent resilience and error handling.

Example usage:
    from testllm.tool_testing import simulate_tool, tool_response_suite

    flight_search = (
        simulate_tool("search_flights")
        .on_success({"flights": [...]})
        .on_failure("Service unavailable")
        .on_timeout(30000)
        .on_empty()
    )

    suite = (
        tool_response_suite("resilience_tests")
        .add_simulator(flight_search)
        .test_all_scenarios("search_flights", "Find flights", ["Should handle gracefully"])
    )
"""

from typing import Any, Callable, Dict, List, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import random
import time

from .types import ToolCall, ToolResponse
from .interceptor import ToolInterceptor


class ScenarioType(Enum):
    """Types of response scenarios"""
    SUCCESS = "success"
    FAILURE = "failure"
    TIMEOUT = "timeout"
    EMPTY = "empty"
    PARTIAL = "partial"
    RATE_LIMITED = "rate_limited"
    AUTH_ERROR = "auth_error"
    INVALID_INPUT = "invalid_input"
    NOT_FOUND = "not_found"
    CUSTOM = "custom"


@dataclass
class ResponseScenario:
    """A single response scenario configuration"""
    scenario_type: ScenarioType
    response: Optional[Any] = None
    error_message: Optional[str] = None
    delay_ms: int = 0
    probability: float = 1.0  # For chaos testing
    condition: Optional[Callable[[ToolCall], bool]] = None
    name: str = ""
    description: str = ""

    def should_trigger(self, call: ToolCall) -> bool:
        """Check if this scenario should trigger for the given call"""
        if self.condition and not self.condition(call):
            return False
        if self.probability < 1.0:
            return random.random() < self.probability
        return True

    def generate_response(self, call: ToolCall) -> ToolResponse:
        """Generate the response for this scenario"""
        # Simulate delay
        if self.delay_ms > 0:
            time.sleep(self.delay_ms / 1000.0)

        if self.scenario_type == ScenarioType.SUCCESS:
            return ToolResponse(
                tool_name=call.tool_name,
                result=self.response,
                success=True
            )

        elif self.scenario_type == ScenarioType.FAILURE:
            return ToolResponse(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error_message=self.error_message or "Tool execution failed"
            )

        elif self.scenario_type == ScenarioType.TIMEOUT:
            # Simulate timeout by sleeping and then returning error
            return ToolResponse(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error_message="Request timed out",
                execution_time_ms=self.delay_ms
            )

        elif self.scenario_type == ScenarioType.EMPTY:
            return ToolResponse(
                tool_name=call.tool_name,
                result=self.response if self.response is not None else [],
                success=True
            )

        elif self.scenario_type == ScenarioType.PARTIAL:
            return ToolResponse(
                tool_name=call.tool_name,
                result=self.response,
                success=True,
                metadata={"partial": True}
            )

        elif self.scenario_type == ScenarioType.RATE_LIMITED:
            return ToolResponse(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error_message="Rate limit exceeded. Please retry after 60 seconds.",
                metadata={"retry_after": 60}
            )

        elif self.scenario_type == ScenarioType.AUTH_ERROR:
            return ToolResponse(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error_message="Authentication failed. Invalid or expired credentials."
            )

        elif self.scenario_type == ScenarioType.INVALID_INPUT:
            return ToolResponse(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error_message=self.error_message or "Invalid input parameters"
            )

        elif self.scenario_type == ScenarioType.NOT_FOUND:
            return ToolResponse(
                tool_name=call.tool_name,
                result=None,
                success=False,
                error_message="Resource not found"
            )

        elif self.scenario_type == ScenarioType.CUSTOM:
            if callable(self.response):
                result = self.response(call)
                if isinstance(result, ToolResponse):
                    return result
                return ToolResponse(
                    tool_name=call.tool_name,
                    result=result,
                    success=True
                )
            return ToolResponse(
                tool_name=call.tool_name,
                result=self.response,
                success=True
            )

        # Default fallback
        return ToolResponse(
            tool_name=call.tool_name,
            result=None,
            success=False,
            error_message="Unknown scenario type"
        )


class ToolSimulator:
    """
    Fluent builder for configuring tool response scenarios.

    Example:
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": [{"id": "F1", "price": 299}]})
            .on_failure("Service unavailable")
            .on_empty()
        )
    """

    def __init__(self, tool_name: str):
        self.tool_name = tool_name
        self.scenarios: Dict[ScenarioType, ResponseScenario] = {}
        self._default_scenario: Optional[ScenarioType] = None
        self._current_scenario: Optional[ScenarioType] = None

    def on_success(self, response: Any) -> 'ToolSimulator':
        """Configure success scenario response"""
        self.scenarios[ScenarioType.SUCCESS] = ResponseScenario(
            scenario_type=ScenarioType.SUCCESS,
            response=response,
            name=f"{self.tool_name}_success",
            description="Successful tool execution"
        )
        self._current_scenario = ScenarioType.SUCCESS
        if self._default_scenario is None:
            self._default_scenario = ScenarioType.SUCCESS
        return self

    def on_failure(self, error_message: str = "Tool execution failed") -> 'ToolSimulator':
        """Configure failure scenario"""
        self.scenarios[ScenarioType.FAILURE] = ResponseScenario(
            scenario_type=ScenarioType.FAILURE,
            error_message=error_message,
            name=f"{self.tool_name}_failure",
            description="Tool execution failure"
        )
        self._current_scenario = ScenarioType.FAILURE
        return self

    def on_timeout(self, delay_ms: int = 30000) -> 'ToolSimulator':
        """Configure timeout scenario"""
        self.scenarios[ScenarioType.TIMEOUT] = ResponseScenario(
            scenario_type=ScenarioType.TIMEOUT,
            delay_ms=delay_ms,
            name=f"{self.tool_name}_timeout",
            description=f"Tool timeout after {delay_ms}ms"
        )
        self._current_scenario = ScenarioType.TIMEOUT
        return self

    def on_empty(self, response: Any = None) -> 'ToolSimulator':
        """Configure empty result scenario"""
        self.scenarios[ScenarioType.EMPTY] = ResponseScenario(
            scenario_type=ScenarioType.EMPTY,
            response=response if response is not None else [],
            name=f"{self.tool_name}_empty",
            description="Tool returns empty result"
        )
        self._current_scenario = ScenarioType.EMPTY
        return self

    def on_partial(self, response: Any) -> 'ToolSimulator':
        """Configure partial result scenario"""
        self.scenarios[ScenarioType.PARTIAL] = ResponseScenario(
            scenario_type=ScenarioType.PARTIAL,
            response=response,
            name=f"{self.tool_name}_partial",
            description="Tool returns partial results"
        )
        self._current_scenario = ScenarioType.PARTIAL
        return self

    def on_rate_limited(self) -> 'ToolSimulator':
        """Configure rate limiting scenario"""
        self.scenarios[ScenarioType.RATE_LIMITED] = ResponseScenario(
            scenario_type=ScenarioType.RATE_LIMITED,
            name=f"{self.tool_name}_rate_limited",
            description="Tool rate limited"
        )
        self._current_scenario = ScenarioType.RATE_LIMITED
        return self

    def on_auth_error(self) -> 'ToolSimulator':
        """Configure authentication error scenario"""
        self.scenarios[ScenarioType.AUTH_ERROR] = ResponseScenario(
            scenario_type=ScenarioType.AUTH_ERROR,
            name=f"{self.tool_name}_auth_error",
            description="Tool authentication failure"
        )
        self._current_scenario = ScenarioType.AUTH_ERROR
        return self

    def on_invalid_input(self, error_message: str = "Invalid input parameters") -> 'ToolSimulator':
        """Configure invalid input scenario"""
        self.scenarios[ScenarioType.INVALID_INPUT] = ResponseScenario(
            scenario_type=ScenarioType.INVALID_INPUT,
            error_message=error_message,
            name=f"{self.tool_name}_invalid_input",
            description="Tool invalid input"
        )
        self._current_scenario = ScenarioType.INVALID_INPUT
        return self

    def on_not_found(self) -> 'ToolSimulator':
        """Configure not found scenario"""
        self.scenarios[ScenarioType.NOT_FOUND] = ResponseScenario(
            scenario_type=ScenarioType.NOT_FOUND,
            name=f"{self.tool_name}_not_found",
            description="Resource not found"
        )
        self._current_scenario = ScenarioType.NOT_FOUND
        return self

    def on_custom(
        self,
        name: str,
        response: Union[Any, Callable[[ToolCall], Any]],
        description: str = ""
    ) -> 'ToolSimulator':
        """Configure a custom scenario"""
        self.scenarios[ScenarioType.CUSTOM] = ResponseScenario(
            scenario_type=ScenarioType.CUSTOM,
            response=response,
            name=name,
            description=description or f"Custom scenario: {name}"
        )
        self._current_scenario = ScenarioType.CUSTOM
        return self

    def with_delay(self, delay_ms: int) -> 'ToolSimulator':
        """Add delay to the current scenario"""
        if self._current_scenario and self._current_scenario in self.scenarios:
            self.scenarios[self._current_scenario].delay_ms = delay_ms
        return self

    def with_probability(self, probability: float) -> 'ToolSimulator':
        """Set probability for the current scenario (for chaos testing)"""
        if self._current_scenario and self._current_scenario in self.scenarios:
            self.scenarios[self._current_scenario].probability = probability
        return self

    def with_condition(self, condition: Callable[[ToolCall], bool]) -> 'ToolSimulator':
        """Add condition for when the current scenario triggers"""
        if self._current_scenario and self._current_scenario in self.scenarios:
            self.scenarios[self._current_scenario].condition = condition
        return self

    def as_default(self) -> 'ToolSimulator':
        """Set the current scenario as the default"""
        if self._current_scenario:
            self._default_scenario = self._current_scenario
        return self

    def get_scenario(self, scenario_type: ScenarioType) -> Optional[ResponseScenario]:
        """Get a specific scenario"""
        return self.scenarios.get(scenario_type)

    def get_default_scenario(self) -> Optional[ResponseScenario]:
        """Get the default scenario"""
        if self._default_scenario:
            return self.scenarios.get(self._default_scenario)
        return None

    def get_all_scenarios(self) -> List[ResponseScenario]:
        """Get all configured scenarios"""
        return list(self.scenarios.values())

    def create_interceptor_for_scenario(self, scenario_type: ScenarioType) -> ToolInterceptor:
        """Create an interceptor configured for a specific scenario"""
        interceptor = ToolInterceptor()
        scenario = self.scenarios.get(scenario_type)

        if scenario:
            def mock_generator(call: ToolCall) -> ToolResponse:
                return scenario.generate_response(call)
            interceptor.register_mock(self.tool_name, mock_generator)

        return interceptor


@dataclass
class SimulationTestCase:
    """A test case generated from a simulation scenario"""
    name: str
    description: str
    scenario_type: ScenarioType
    user_input: str
    criteria: List[str]
    simulator: ToolSimulator
    expected_behavior: str = ""


class ToolResponseSuite:
    """
    Suite for testing agent behavior across multiple tool response scenarios.

    Example:
        suite = (
            tool_response_suite("flight_booking_resilience")
            .add_simulator(flight_search_simulator)
            .test_all_scenarios(
                "search_flights",
                "Find flights to NYC",
                ["Should handle response gracefully"]
            )
        )
    """

    def __init__(self, suite_id: str, description: str = ""):
        self.suite_id = suite_id
        self.description = description
        self.simulators: Dict[str, ToolSimulator] = {}
        self.test_cases: List[SimulationTestCase] = []

    def add_simulator(self, simulator: ToolSimulator) -> 'ToolResponseSuite':
        """Add a tool simulator to the suite"""
        self.simulators[simulator.tool_name] = simulator
        return self

    def test_scenario(
        self,
        tool_name: str,
        scenario_type: ScenarioType,
        user_input: str,
        criteria: List[str],
        name: Optional[str] = None,
        expected_behavior: str = ""
    ) -> 'ToolResponseSuite':
        """Add a test for a specific scenario"""
        simulator = self.simulators.get(tool_name)
        if not simulator:
            raise ValueError(f"No simulator registered for tool: {tool_name}")

        scenario = simulator.get_scenario(scenario_type)
        if not scenario:
            raise ValueError(f"Scenario {scenario_type} not configured for tool: {tool_name}")

        test_name = name or f"{tool_name}_{scenario_type.value}"
        self.test_cases.append(SimulationTestCase(
            name=test_name,
            description=scenario.description,
            scenario_type=scenario_type,
            user_input=user_input,
            criteria=criteria,
            simulator=simulator,
            expected_behavior=expected_behavior
        ))
        return self

    def test_all_scenarios(
        self,
        tool_name: str,
        user_input: str,
        criteria: List[str],
        exclude: Optional[List[ScenarioType]] = None
    ) -> 'ToolResponseSuite':
        """Add tests for all configured scenarios of a tool"""
        simulator = self.simulators.get(tool_name)
        if not simulator:
            raise ValueError(f"No simulator registered for tool: {tool_name}")

        exclude = exclude or []
        for scenario_type, scenario in simulator.scenarios.items():
            if scenario_type not in exclude:
                self.test_cases.append(SimulationTestCase(
                    name=f"{tool_name}_{scenario_type.value}",
                    description=scenario.description,
                    scenario_type=scenario_type,
                    user_input=user_input,
                    criteria=criteria,
                    simulator=simulator
                ))
        return self

    def get_test_cases(self) -> List[SimulationTestCase]:
        """Get all test cases"""
        return self.test_cases.copy()

    def get_interceptor_for_test(self, test_case: SimulationTestCase) -> ToolInterceptor:
        """Get an interceptor configured for a specific test case"""
        return test_case.simulator.create_interceptor_for_scenario(test_case.scenario_type)


class ChaosSimulator:
    """
    Chaos testing simulator that randomly injects failures.

    Example:
        chaos = ChaosSimulator(failure_rate=0.2)
        chaos.add_failure_mode("search_flights", ScenarioType.TIMEOUT)
        chaos.add_failure_mode("search_flights", ScenarioType.RATE_LIMITED)

        interceptor = chaos.create_interceptor()
    """

    def __init__(self, failure_rate: float = 0.1):
        """
        Initialize chaos simulator.

        Args:
            failure_rate: Probability of injecting a failure (0.0 to 1.0)
        """
        self.failure_rate = failure_rate
        self.failure_modes: Dict[str, List[ResponseScenario]] = {}
        self.success_responses: Dict[str, Any] = {}

    def add_failure_mode(
        self,
        tool_name: str,
        scenario_type: ScenarioType,
        weight: float = 1.0,
        **kwargs
    ) -> 'ChaosSimulator':
        """Add a possible failure mode for a tool"""
        if tool_name not in self.failure_modes:
            self.failure_modes[tool_name] = []

        scenario = ResponseScenario(
            scenario_type=scenario_type,
            probability=weight,
            **kwargs
        )
        self.failure_modes[tool_name].append(scenario)
        return self

    def set_success_response(self, tool_name: str, response: Any) -> 'ChaosSimulator':
        """Set the success response for a tool"""
        self.success_responses[tool_name] = response
        return self

    def create_interceptor(self) -> ToolInterceptor:
        """Create an interceptor with chaos injection"""
        interceptor = ToolInterceptor()

        for tool_name, failures in self.failure_modes.items():
            def create_mock(tn: str, f: List[ResponseScenario]):
                def mock_generator(call: ToolCall) -> ToolResponse:
                    # Decide if we should inject failure
                    if random.random() < self.failure_rate and f:
                        # Pick a failure mode weighted by probability
                        total_weight = sum(s.probability for s in f)
                        r = random.random() * total_weight
                        cumulative = 0.0
                        for scenario in f:
                            cumulative += scenario.probability
                            if r <= cumulative:
                                return scenario.generate_response(call)

                    # Return success response
                    success_data = self.success_responses.get(tn, {})
                    return ToolResponse(
                        tool_name=tn,
                        result=success_data,
                        success=True
                    )
                return mock_generator

            interceptor.register_mock(tool_name, create_mock(tool_name, failures))

        return interceptor


def simulate_tool(tool_name: str) -> ToolSimulator:
    """
    Factory function to create a tool simulator.

    Example:
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": [...]})
            .on_failure("Service unavailable")
        )
    """
    return ToolSimulator(tool_name)


def tool_response_suite(suite_id: str, description: str = "") -> ToolResponseSuite:
    """
    Factory function to create a tool response test suite.

    Example:
        suite = (
            tool_response_suite("resilience_tests")
            .add_simulator(flight_simulator)
            .test_all_scenarios("search_flights", "Find flights", [...])
        )
    """
    return ToolResponseSuite(suite_id, description)


def chaos_simulator(failure_rate: float = 0.1) -> ChaosSimulator:
    """
    Factory function to create a chaos simulator.

    Example:
        chaos = chaos_simulator(failure_rate=0.2)
        chaos.add_failure_mode("search_flights", ScenarioType.TIMEOUT)
    """
    return ChaosSimulator(failure_rate)
