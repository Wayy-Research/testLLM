"""
Tests for Tool Response Simulation
"""

import pytest
from testllm.tool_testing import (
    simulate_tool,
    tool_response_suite,
    chaos_simulator,
    ToolSimulator,
    ToolResponseSuite,
    ChaosSimulator,
    ScenarioType,
    ResponseScenario,
    ToolCall,
    ToolInterceptor,
)
from testllm.tool_testing.scenarios import (
    SearchScenarios,
    CRUDScenarios,
    APIScenarios,
    get_all_scenarios,
)


class TestToolSimulator:
    """Tests for the ToolSimulator class"""

    def test_basic_simulator(self):
        """Test creating a basic simulator"""
        simulator = simulate_tool("search_flights")
        assert simulator.tool_name == "search_flights"

    def test_success_scenario(self):
        """Test success scenario configuration"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": [{"id": "F1", "price": 299}]})
        )

        scenario = simulator.get_scenario(ScenarioType.SUCCESS)
        assert scenario is not None
        assert scenario.response == {"flights": [{"id": "F1", "price": 299}]}

    def test_failure_scenario(self):
        """Test failure scenario configuration"""
        simulator = (
            simulate_tool("search_flights")
            .on_failure("Service unavailable")
        )

        scenario = simulator.get_scenario(ScenarioType.FAILURE)
        assert scenario is not None
        assert scenario.error_message == "Service unavailable"

    def test_timeout_scenario(self):
        """Test timeout scenario configuration"""
        simulator = (
            simulate_tool("search_flights")
            .on_timeout(5000)  # 5 seconds
        )

        scenario = simulator.get_scenario(ScenarioType.TIMEOUT)
        assert scenario is not None
        assert scenario.delay_ms == 5000

    def test_empty_scenario(self):
        """Test empty result scenario"""
        simulator = (
            simulate_tool("search_flights")
            .on_empty({"flights": [], "message": "No results"})
        )

        scenario = simulator.get_scenario(ScenarioType.EMPTY)
        assert scenario is not None
        assert scenario.response == {"flights": [], "message": "No results"}

    def test_partial_scenario(self):
        """Test partial result scenario"""
        simulator = (
            simulate_tool("search_flights")
            .on_partial({"flights": [{"id": "F1"}], "incomplete": True})
        )

        scenario = simulator.get_scenario(ScenarioType.PARTIAL)
        assert scenario is not None

    def test_rate_limited_scenario(self):
        """Test rate limited scenario"""
        simulator = simulate_tool("search_flights").on_rate_limited()

        scenario = simulator.get_scenario(ScenarioType.RATE_LIMITED)
        assert scenario is not None

    def test_auth_error_scenario(self):
        """Test authentication error scenario"""
        simulator = simulate_tool("search_flights").on_auth_error()

        scenario = simulator.get_scenario(ScenarioType.AUTH_ERROR)
        assert scenario is not None

    def test_custom_scenario(self):
        """Test custom scenario"""
        simulator = (
            simulate_tool("search_flights")
            .on_custom("maintenance", {"status": "under maintenance"}, "System maintenance")
        )

        scenario = simulator.get_scenario(ScenarioType.CUSTOM)
        assert scenario is not None
        assert scenario.name == "maintenance"

    def test_fluent_chaining(self):
        """Test fluent method chaining"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": []})
            .on_failure("Error")
            .on_timeout(30000)
            .on_empty()
        )

        assert simulator.get_scenario(ScenarioType.SUCCESS) is not None
        assert simulator.get_scenario(ScenarioType.FAILURE) is not None
        assert simulator.get_scenario(ScenarioType.TIMEOUT) is not None
        assert simulator.get_scenario(ScenarioType.EMPTY) is not None

    def test_with_delay(self):
        """Test adding delay to scenario"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": []})
            .with_delay(1000)
        )

        scenario = simulator.get_scenario(ScenarioType.SUCCESS)
        assert scenario.delay_ms == 1000

    def test_as_default(self):
        """Test setting default scenario"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": []})
            .as_default()
            .on_failure("Error")
        )

        default = simulator.get_default_scenario()
        assert default.scenario_type == ScenarioType.SUCCESS

    def test_get_all_scenarios(self):
        """Test getting all configured scenarios"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": []})
            .on_failure("Error")
            .on_timeout(30000)
        )

        all_scenarios = simulator.get_all_scenarios()
        assert len(all_scenarios) == 3


class TestResponseScenario:
    """Tests for ResponseScenario response generation"""

    def test_generate_success_response(self):
        """Test generating success response"""
        scenario = ResponseScenario(
            scenario_type=ScenarioType.SUCCESS,
            response={"data": "test"}
        )

        call = ToolCall(tool_name="test")
        response = scenario.generate_response(call)

        assert response.success is True
        assert response.result == {"data": "test"}

    def test_generate_failure_response(self):
        """Test generating failure response"""
        scenario = ResponseScenario(
            scenario_type=ScenarioType.FAILURE,
            error_message="Something went wrong"
        )

        call = ToolCall(tool_name="test")
        response = scenario.generate_response(call)

        assert response.success is False
        assert response.error_message == "Something went wrong"

    def test_generate_rate_limited_response(self):
        """Test generating rate limited response"""
        scenario = ResponseScenario(scenario_type=ScenarioType.RATE_LIMITED)

        call = ToolCall(tool_name="test")
        response = scenario.generate_response(call)

        assert response.success is False
        assert "Rate limit" in response.error_message

    def test_generate_auth_error_response(self):
        """Test generating auth error response"""
        scenario = ResponseScenario(scenario_type=ScenarioType.AUTH_ERROR)

        call = ToolCall(tool_name="test")
        response = scenario.generate_response(call)

        assert response.success is False
        assert "Authentication" in response.error_message

    def test_custom_callable_response(self):
        """Test custom callable response generator"""
        def custom_generator(call: ToolCall):
            return {"echo": call.arguments.get("message")}

        scenario = ResponseScenario(
            scenario_type=ScenarioType.CUSTOM,
            response=custom_generator
        )

        call = ToolCall(tool_name="test", arguments={"message": "hello"})
        response = scenario.generate_response(call)

        assert response.result["echo"] == "hello"

    def test_conditional_trigger(self):
        """Test conditional scenario trigger"""
        scenario = ResponseScenario(
            scenario_type=ScenarioType.SUCCESS,
            condition=lambda call: call.arguments.get("premium") is True
        )

        premium_call = ToolCall(tool_name="test", arguments={"premium": True})
        regular_call = ToolCall(tool_name="test", arguments={"premium": False})

        assert scenario.should_trigger(premium_call) is True
        assert scenario.should_trigger(regular_call) is False


class TestToolResponseSuite:
    """Tests for ToolResponseSuite"""

    def test_create_suite(self):
        """Test creating a response suite"""
        suite = tool_response_suite("test_suite", "Test description")
        assert suite.suite_id == "test_suite"
        assert suite.description == "Test description"

    def test_add_simulator(self):
        """Test adding simulator to suite"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": []})
            .on_failure("Error")
        )

        suite = (
            tool_response_suite("test_suite")
            .add_simulator(simulator)
        )

        assert "search_flights" in suite.simulators

    def test_test_scenario(self):
        """Test adding individual scenario test"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": []})
            .on_failure("Error")
        )

        suite = (
            tool_response_suite("test_suite")
            .add_simulator(simulator)
            .test_scenario(
                "search_flights",
                ScenarioType.SUCCESS,
                "Find flights to NYC",
                ["Should display flight options"]
            )
        )

        test_cases = suite.get_test_cases()
        assert len(test_cases) == 1
        assert test_cases[0].scenario_type == ScenarioType.SUCCESS

    def test_test_all_scenarios(self):
        """Test adding tests for all scenarios"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": []})
            .on_failure("Error")
            .on_timeout(30000)
        )

        suite = (
            tool_response_suite("test_suite")
            .add_simulator(simulator)
            .test_all_scenarios(
                "search_flights",
                "Find flights",
                ["Should handle gracefully"]
            )
        )

        test_cases = suite.get_test_cases()
        assert len(test_cases) == 3

    def test_test_all_scenarios_with_exclusions(self):
        """Test excluding certain scenarios"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": []})
            .on_failure("Error")
            .on_timeout(30000)
        )

        suite = (
            tool_response_suite("test_suite")
            .add_simulator(simulator)
            .test_all_scenarios(
                "search_flights",
                "Find flights",
                ["Should handle gracefully"],
                exclude=[ScenarioType.TIMEOUT]
            )
        )

        test_cases = suite.get_test_cases()
        assert len(test_cases) == 2
        assert not any(tc.scenario_type == ScenarioType.TIMEOUT for tc in test_cases)

    def test_get_interceptor_for_test(self):
        """Test getting interceptor configured for a specific test case"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": [{"id": "F1"}]})
        )

        suite = (
            tool_response_suite("test_suite")
            .add_simulator(simulator)
            .test_scenario(
                "search_flights",
                ScenarioType.SUCCESS,
                "Find flights",
                ["Should show results"]
            )
        )

        test_case = suite.get_test_cases()[0]
        interceptor = suite.get_interceptor_for_test(test_case)

        # Interceptor should mock the tool with the scenario response
        call = ToolCall(tool_name="search_flights")
        response = interceptor.intercept(call)

        assert response is not None
        assert response.result["flights"][0]["id"] == "F1"


class TestChaosSimulator:
    """Tests for ChaosSimulator"""

    def test_create_chaos_simulator(self):
        """Test creating a chaos simulator"""
        chaos = chaos_simulator(failure_rate=0.5)
        assert chaos.failure_rate == 0.5

    def test_add_failure_mode(self):
        """Test adding failure modes"""
        chaos = (
            chaos_simulator(failure_rate=0.2)
            .add_failure_mode("search_flights", ScenarioType.TIMEOUT)
            .add_failure_mode("search_flights", ScenarioType.FAILURE)
        )

        assert "search_flights" in chaos.failure_modes
        assert len(chaos.failure_modes["search_flights"]) == 2

    def test_set_success_response(self):
        """Test setting success response"""
        chaos = (
            chaos_simulator()
            .set_success_response("search_flights", {"flights": []})
        )

        assert chaos.success_responses["search_flights"] == {"flights": []}

    def test_create_interceptor(self):
        """Test creating interceptor from chaos simulator"""
        chaos = (
            chaos_simulator(failure_rate=0.0)  # 0% failure for deterministic test
            .add_failure_mode("search_flights", ScenarioType.TIMEOUT)
            .set_success_response("search_flights", {"flights": [{"id": "F1"}]})
        )

        interceptor = chaos.create_interceptor()
        call = ToolCall(tool_name="search_flights")
        response = interceptor.intercept(call)

        # With 0% failure rate, should always get success response
        assert response.success is True
        assert response.result["flights"][0]["id"] == "F1"


class TestPrebuiltScenarios:
    """Tests for pre-built scenarios"""

    def test_flight_search_scenarios(self):
        """Test pre-built flight search scenarios"""
        simulator = SearchScenarios.flight_search()

        assert simulator.tool_name == "search_flights"
        assert simulator.get_scenario(ScenarioType.SUCCESS) is not None
        assert simulator.get_scenario(ScenarioType.EMPTY) is not None
        assert simulator.get_scenario(ScenarioType.FAILURE) is not None
        assert simulator.get_scenario(ScenarioType.TIMEOUT) is not None
        assert simulator.get_scenario(ScenarioType.RATE_LIMITED) is not None

    def test_hotel_search_scenarios(self):
        """Test pre-built hotel search scenarios"""
        simulator = SearchScenarios.hotel_search()

        assert simulator.tool_name == "search_hotels"
        assert simulator.get_scenario(ScenarioType.SUCCESS) is not None

    def test_user_crud_scenarios(self):
        """Test pre-built user CRUD scenarios"""
        simulators = CRUDScenarios.user_management()

        assert "create_user" in simulators
        assert "get_user" in simulators
        assert "update_user" in simulators
        assert "delete_user" in simulators

        # Check create_user has expected scenarios
        create_sim = simulators["create_user"]
        assert create_sim.get_scenario(ScenarioType.SUCCESS) is not None

    def test_weather_api_scenarios(self):
        """Test pre-built weather API scenarios"""
        simulator = APIScenarios.weather_api()

        assert simulator.tool_name == "get_weather"
        assert simulator.get_scenario(ScenarioType.SUCCESS) is not None
        assert simulator.get_scenario(ScenarioType.NOT_FOUND) is not None

    def test_payment_api_scenarios(self):
        """Test pre-built payment API scenarios"""
        simulator = APIScenarios.payment_api()

        assert simulator.tool_name == "process_payment"
        assert simulator.get_scenario(ScenarioType.SUCCESS) is not None
        assert simulator.get_scenario(ScenarioType.FAILURE) is not None

    def test_get_all_scenarios(self):
        """Test getting all pre-built scenarios"""
        all_scenarios = get_all_scenarios()

        assert "search" in all_scenarios
        assert "crud_user" in all_scenarios
        assert "api" in all_scenarios
        assert "database" in all_scenarios
        assert "files" in all_scenarios
        assert "calendar" in all_scenarios


class TestSimulatorInterceptorIntegration:
    """Tests for simulator and interceptor integration"""

    def test_create_interceptor_for_scenario(self):
        """Test creating interceptor from simulator scenario"""
        simulator = (
            simulate_tool("search_flights")
            .on_success({"flights": [{"id": "F1", "price": 299}]})
            .on_failure("Service down")
        )

        # Get interceptor for success scenario
        success_interceptor = simulator.create_interceptor_for_scenario(ScenarioType.SUCCESS)
        call = ToolCall(tool_name="search_flights")
        response = success_interceptor.intercept(call)

        assert response.success is True
        assert response.result["flights"][0]["id"] == "F1"

        # Get interceptor for failure scenario
        failure_interceptor = simulator.create_interceptor_for_scenario(ScenarioType.FAILURE)
        response = failure_interceptor.intercept(call)

        assert response.success is False
