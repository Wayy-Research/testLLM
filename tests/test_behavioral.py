"""
Test file for behavioral pattern testing functionality
Run with: pytest tests/test_behavioral.py -v
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from testllm.behavioral import (
    ToolUsagePatterns, BusinessLogicPatterns, ContextPatterns,
    IntegrationPatterns, PerformancePatterns
)
from testllm.flows import ConversationFlow, FlowResult
from testllm.core import AgentUnderTest
from testllm.evaluation_loop import ConsensusResult


class MockBehavioralAgent(AgentUnderTest):
    """Mock agent for testing behavioral patterns"""
    
    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.responses = {
            # Search patterns
            "search for files": "I'm performing a file search across your system. Please wait while I scan the directories.",
            "find weather data": "I'm searching our weather database for current conditions. This may take a moment.",
            
            # API integration patterns
            "get current weather": "I'm checking real-time weather data from our meteorological service.",
            "stock price of AAPL": "I'm retrieving the current stock price from the financial data provider.",
            "process payment": "I'm securely processing your payment through our payment gateway.",
            
            # Data processing patterns
            "analyze sales data": "I'm performing analysis on your sales dataset. Given the size, this will take approximately 5 minutes.",
            "calculate statistics": "I'm computing statistical measures for your data including mean, median, and standard deviation.",
            
            # Authentication patterns
            "I'm a new user": "Welcome! I see you're a new user. Let me guide you through our onboarding process.",
            "I'm a returning user": "Welcome back! I've identified you as a returning user with an existing account.",
            "I'm a premium user": "Hello! I see you have premium status. I'll prioritize your request with our enhanced service level.",
            
            # Purchase workflow
            "buy premium plan": "I see you're interested in our premium plan. It includes advanced features and costs $29/month.",
            "complete purchase": "I'll guide you through the purchase process. First, I need to verify your payment information.",
            "payment completed": "Thank you! Your payment has been processed successfully. Your premium features will be activated within 24 hours.",
            
            # Error handling
            "invalid location INVALID": "I'm sorry, I couldn't find that location. Could you please provide a valid city name?",
            "I meant Seattle": "Perfect! I understand you meant Seattle. Let me look up the weather for Seattle now.",
            
            # Memory and context
            "Hi, my name is John": "Nice to meet you, John! I've noted your name and I'm here to help.",
            "I need a laptop": "I understand you're looking for a laptop. I'll help you find the right one for your needs.",
            "for software development": "Got it! A laptop for software development. I'll focus on models with strong performance and development tools.",
            "What was my name?": "Your name is John, as you mentioned earlier in our conversation.",
            "What am I shopping for?": "You're shopping for a laptop, specifically for software development work.",
            
            # Preference tracking
            "I prefer detailed explanations": "Understood! I'll make sure to provide detailed, comprehensive explanations in my responses.",
            "explain machine learning": "Machine learning is a comprehensive field of artificial intelligence that involves algorithms learning patterns from data. It encompasses supervised learning (with labeled data), unsupervised learning (finding patterns in unlabeled data), and reinforcement learning (learning through interaction and feedback). The process typically involves data collection, preprocessing, model selection, training, validation, and deployment phases.",
            
            # Real-time data
            "current stock information": "I'm accessing real-time financial data. The current information was last updated 2 minutes ago.",
            "most recent data": "Yes, this data was retrieved just now from our live feeds and represents the most current information available.",
            
            # Multi-system integration
            "book flight and hotel": "I'll coordinate both your flight and hotel reservations. Let me check availability and ensure the timing works well together.",
            "hotel near airport": "I'll search for hotels with good airport proximity and factor that into your booking coordination.",
            
            # Performance patterns
            "analyze 5 years stock data": "This is a complex analysis requiring processing of extensive historical data. I estimate this will take 15-20 minutes to complete properly.",
            "quick summary instead": "Certainly! I can provide a high-level summary based on key metrics rather than the full detailed analysis.",
            "process 50GB in 30 seconds": "I understand the urgency, but processing 50GB of data in 30 seconds isn't feasible with current systems. I can offer a preview analysis of a sample, or we can discuss prioritizing specific portions of the dataset."
        }
    
    def send_message(self, message: str) -> str:
        """Mock send_message with behavioral pattern responses"""
        # Find the best matching response
        message_lower = message.lower()
        for key, response in self.responses.items():
            if key.lower() in message_lower:
                self.conversation_history.append({"user": message, "agent": response})
                return response
        
        # Default response
        default_response = f"I understand your request about '{message}'. Let me help you with that."
        self.conversation_history.append({"user": message, "agent": default_response})
        return default_response
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_history = []


@pytest.fixture
def mock_behavioral_agent():
    """Fixture providing a mock behavioral agent"""
    return MockBehavioralAgent()


@pytest.fixture
def mock_consensus_result():
    """Fixture providing a successful consensus result"""
    return ConsensusResult(
        criterion="Test criterion",
        consensus_score=1.0,
        passed=True,
        individual_results=[]
    )


def mock_flow_execution(flow, agent):
    """Helper function to mock flow execution with successful results"""
    with patch('testllm.flows.EvaluationLoop') as mock_eval_class:
        mock_evaluator = AsyncMock()
        
        def mock_evaluate_response(user_input, agent_response, criteria):
            return [
                ConsensusResult(criterion.criterion, 1.0, True, [])
                for criterion in criteria
            ]
        
        mock_evaluator.evaluate_response.side_effect = mock_evaluate_response
        mock_eval_class.return_value = mock_evaluator
        
        return flow.execute_sync(agent)


class TestToolUsagePatterns:
    """Test the ToolUsagePatterns class"""
    
    def test_search_pattern_creation(self):
        """Test search pattern flow creation"""
        flow = ToolUsagePatterns.search_pattern("find my files", "file")
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "search_pattern_file"
        assert flow.description == "Test file search behavioral pattern"
        assert len(flow.steps) == 1
        
        step = flow.steps[0]
        assert step.user_input == "find my files"
        assert step.expect_tool_usage_indicators == ["file_search"]
        assert any("file search" in criterion for criterion in step.criteria)
    
    def test_search_pattern_execution(self, mock_behavioral_agent):
        """Test search pattern execution"""
        flow = ToolUsagePatterns.search_pattern("search for files", "file")
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 1
        assert len(result.step_results) == 1
    
    def test_api_integration_pattern_creation(self):
        """Test API integration pattern flow creation"""
        flow = ToolUsagePatterns.api_integration_pattern("get weather data", "weather")
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "api_integration_weather"
        assert len(flow.steps) == 1
        
        step = flow.steps[0]
        assert step.user_input == "get weather data"
        assert step.expect_tool_usage_indicators == ["weather_api"]
        assert any("weather information" in criterion for criterion in step.criteria)
    
    def test_api_integration_pattern_execution(self, mock_behavioral_agent):
        """Test API integration pattern execution"""
        flow = ToolUsagePatterns.api_integration_pattern("get current weather", "weather")
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 1
    
    def test_data_processing_pattern_creation(self):
        """Test data processing pattern flow creation"""
        flow = ToolUsagePatterns.data_processing_pattern("analyze my data", "analysis")
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "data_processing_analysis"
        assert len(flow.steps) == 1
        
        step = flow.steps[0]
        assert step.user_input == "analyze my data"
        assert step.expect_tool_usage_indicators == ["data_analysis"]
        assert any("analysis" in criterion for criterion in step.criteria)
    
    def test_data_processing_pattern_execution(self, mock_behavioral_agent):
        """Test data processing pattern execution"""
        flow = ToolUsagePatterns.data_processing_pattern("analyze sales data", "analysis")
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 1


class TestBusinessLogicPatterns:
    """Test the BusinessLogicPatterns class"""
    
    def test_user_authentication_flow_creation(self):
        """Test user authentication flow creation"""
        flow = BusinessLogicPatterns.user_authentication_flow("new")
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "auth_flow_new"
        assert flow.description == "Test authentication flow for new user"
        assert len(flow.steps) == 2  # Initial contact + verification
        
        # Check first step
        first_step = flow.steps[0]
        assert "new user" in first_step.user_input
        assert first_step.expect_business_logic == ["user_identification", "access_level_determination"]
        
        # Check second step
        second_step = flow.steps[1]
        assert "access my account" in second_step.user_input
        assert second_step.expect_business_logic == ["identity_verification", "security_protocols"]
    
    def test_user_authentication_flow_execution(self, mock_behavioral_agent):
        """Test user authentication flow execution"""
        flow = BusinessLogicPatterns.user_authentication_flow("premium")
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 2
        assert len(result.step_results) == 2
    
    def test_purchase_workflow_creation(self):
        """Test purchase workflow creation"""
        flow = BusinessLogicPatterns.purchase_workflow()
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "purchase_workflow"
        assert flow.description == "Test purchase business logic"
        assert len(flow.steps) == 3  # Product inquiry + process + confirmation
        
        # Check product inquiry step
        product_step = flow.steps[0]
        assert "premium plan" in product_step.user_input
        assert product_step.expect_business_logic == ["product_availability", "pricing_rules"]
        
        # Check purchase process step
        process_step = flow.steps[1]
        assert "complete the purchase" in process_step.user_input
        assert process_step.expect_business_logic == ["payment_processing", "order_management"]
        
        # Check confirmation step
        confirm_step = flow.steps[2]
        assert "completed payment" in confirm_step.user_input
        assert confirm_step.expect_business_logic == ["order_confirmation", "service_activation"]
    
    def test_purchase_workflow_execution(self, mock_behavioral_agent):
        """Test purchase workflow execution"""
        flow = BusinessLogicPatterns.purchase_workflow()
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 3
        assert len(result.step_results) == 3
    
    def test_error_handling_workflow_creation(self):
        """Test error handling workflow creation"""
        flow = BusinessLogicPatterns.error_handling_workflow()
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "error_handling"
        assert flow.description == "Test error handling business logic"
        assert len(flow.steps) == 2  # Invalid request + recovery
        
        # Check error handling step
        error_step = flow.steps[0]
        assert "INVALID_LOCATION_12345" in error_step.user_input
        assert error_step.expect_business_logic == ["input_validation", "error_recovery"]
        
        # Check recovery step
        recovery_step = flow.steps[1]
        assert "Seattle" in recovery_step.user_input
        assert recovery_step.expect_business_logic == ["error_recovery", "context_correction"]
    
    def test_error_handling_workflow_execution(self, mock_behavioral_agent):
        """Test error handling workflow execution"""
        flow = BusinessLogicPatterns.error_handling_workflow()
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 2


class TestContextPatterns:
    """Test the ContextPatterns class"""
    
    def test_multi_turn_memory_creation(self):
        """Test multi-turn memory flow creation"""
        flow = ContextPatterns.multi_turn_memory()
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "memory_test"
        assert flow.description == "Test conversation memory"
        assert len(flow.steps) == 4  # Establish + add details + memory check + context application
        
        # Check context establishment
        establish_step = flow.steps[0]
        assert "John" in establish_step.user_input
        assert "laptop" in establish_step.user_input
        
        # Check memory check step
        memory_step = flow.steps[2]
        assert memory_step.expect_context_retention == True
        assert "What was my name" in memory_step.user_input
        
        # Check context application step
        context_step = flow.steps[3]
        assert context_step.expect_context_retention == True
        assert "What was I shopping for" in context_step.user_input
    
    def test_multi_turn_memory_execution(self, mock_behavioral_agent):
        """Test multi-turn memory flow execution"""
        flow = ContextPatterns.multi_turn_memory()
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 4
        assert len(result.step_results) == 4
    
    def test_preference_tracking_creation(self):
        """Test preference tracking flow creation"""
        flow = ContextPatterns.preference_tracking()
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "preference_tracking"
        assert flow.description == "Test preference learning"
        assert len(flow.steps) == 3  # Express + apply + consistency
        
        # Check preference expression
        express_step = flow.steps[0]
        assert "detailed explanations" in express_step.user_input
        
        # Check preference application
        apply_step = flow.steps[1]
        assert apply_step.expect_context_retention == True
        assert "machine learning" in apply_step.user_input
        
        # Check preference consistency
        consistency_step = flow.steps[2]
        assert consistency_step.expect_context_retention == True
        assert "quantum computing" in consistency_step.user_input
    
    def test_preference_tracking_execution(self, mock_behavioral_agent):
        """Test preference tracking flow execution"""
        flow = ContextPatterns.preference_tracking()
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 3


class TestIntegrationPatterns:
    """Test the IntegrationPatterns class"""
    
    def test_real_time_data_pattern_creation(self):
        """Test real-time data pattern creation"""
        flow = IntegrationPatterns.real_time_data_pattern("stock")
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "realtime_stock"
        assert flow.description == "Test real-time stock data integration"
        assert len(flow.steps) == 2  # Data request + freshness check
        
        # Check data request step
        data_step = flow.steps[0]
        assert "stock information" in data_step.user_input
        assert data_step.expect_tool_usage_indicators == ["stock_api", "realtime_data"]
        
        # Check freshness check step
        freshness_step = flow.steps[1]
        assert freshness_step.expect_context_retention == True
        assert "most recent data" in freshness_step.user_input
    
    def test_real_time_data_pattern_execution(self, mock_behavioral_agent):
        """Test real-time data pattern execution"""
        flow = IntegrationPatterns.real_time_data_pattern("stock")
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 2
    
    def test_multi_system_integration_creation(self):
        """Test multi-system integration creation"""
        flow = IntegrationPatterns.multi_system_integration()
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "multi_system_integration"
        assert flow.description == "Test coordination across multiple systems"
        assert len(flow.steps) == 2  # Multi-booking + coordination
        
        # Check multi-booking step
        booking_step = flow.steps[0]
        assert "flight and reserve a hotel" in booking_step.user_input
        assert booking_step.expect_tool_usage_indicators == ["flight_booking", "hotel_reservation", "calendar_check"]
        
        # Check coordination step
        coord_step = flow.steps[1]
        assert "hotel is near the airport" in coord_step.user_input
        assert coord_step.expect_business_logic == ["location_coordination", "travel_optimization"]
    
    def test_multi_system_integration_execution(self, mock_behavioral_agent):
        """Test multi-system integration execution"""
        flow = IntegrationPatterns.multi_system_integration()
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 2


class TestPerformancePatterns:
    """Test the PerformancePatterns class"""
    
    def test_complex_request_handling_creation(self):
        """Test complex request handling pattern creation"""
        flow = PerformancePatterns.complex_request_handling()
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "complex_request_handling"
        assert flow.description == "Test complex request performance behavior"
        assert len(flow.steps) == 1
        
        # Check complex request step
        complex_step = flow.steps[0]
        assert "5 years of stock data" in complex_step.user_input
        assert "top 100 companies" in complex_step.user_input
        assert any("complexity" in criterion for criterion in complex_step.criteria)
        assert any("timing" in criterion for criterion in complex_step.criteria)
    
    def test_complex_request_handling_execution(self, mock_behavioral_agent):
        """Test complex request handling execution"""
        flow = PerformancePatterns.complex_request_handling()
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 1
    
    def test_resource_limitation_handling_creation(self):
        """Test resource limitation handling pattern creation"""
        flow = PerformancePatterns.resource_limitation_handling()
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "resource_limitation_handling"
        assert flow.description == "Test resource limitation behavioral patterns"
        assert len(flow.steps) == 1
        
        # Check resource limitation step
        limit_step = flow.steps[0]
        assert "50GB dataset" in limit_step.user_input
        assert "30 seconds" in limit_step.user_input
        assert any("unrealistic" in criterion for criterion in limit_step.criteria)
        assert any("limitations" in criterion for criterion in limit_step.criteria)
    
    def test_resource_limitation_handling_execution(self, mock_behavioral_agent):
        """Test resource limitation handling execution"""
        flow = PerformancePatterns.resource_limitation_handling()
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 1


class TestBehavioralPatternIntegration:
    """Integration tests for behavioral patterns"""
    
    def test_multiple_patterns_execution(self, mock_behavioral_agent):
        """Test executing multiple behavioral patterns"""
        # Create different pattern flows
        search_flow = ToolUsagePatterns.search_pattern("search files", "file")
        auth_flow = BusinessLogicPatterns.user_authentication_flow("new")
        memory_flow = ContextPatterns.multi_turn_memory()
        
        # Execute all flows
        search_result = mock_flow_execution(search_flow, mock_behavioral_agent)
        auth_result = mock_flow_execution(auth_flow, mock_behavioral_agent)
        memory_result = mock_flow_execution(memory_flow, mock_behavioral_agent)
        
        # All should pass
        assert search_result.passed
        assert auth_result.passed
        assert memory_result.passed
        
        # Check execution metrics
        assert search_result.steps_executed >= 1
        assert auth_result.steps_executed >= 2
        assert memory_result.steps_executed >= 4
    
    def test_pattern_customization(self, mock_behavioral_agent):
        """Test that patterns can be customized and extended"""
        # Create base pattern
        flow = ToolUsagePatterns.search_pattern("search data", "database")
        
        # Add custom steps
        flow.step(
            "Filter results by date",
            criteria=["Should apply date filtering"],
            expect_tool_usage=["date_filter"]
        )
        
        # Execute customized flow
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        assert result.passed
        assert result.steps_executed == 2  # Original + custom step
    
    def test_behavioral_score_tracking(self, mock_behavioral_agent):
        """Test behavioral score tracking across patterns"""
        # Create flow with specific behavioral expectations
        flow = BusinessLogicPatterns.purchase_workflow()
        result = mock_flow_execution(flow, mock_behavioral_agent)
        
        # Should have business logic scoring
        assert result.business_logic_score >= 0.0
        assert result.business_logic_score <= 1.0
        
        # For context patterns
        context_flow = ContextPatterns.multi_turn_memory()
        context_result = mock_flow_execution(context_flow, mock_behavioral_agent)
        
        assert context_result.context_retention_score >= 0.0
        assert context_result.context_retention_score <= 1.0


if __name__ == "__main__":
    # Run tests directly
    import sys
    import subprocess
    
    print("Running behavioral pattern tests...")
    print("=" * 50)
    
    # Run with pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v",
        "--tb=short"
    ], capture_output=False)
    
    sys.exit(result.returncode)