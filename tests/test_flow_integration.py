"""
Integration tests for the complete flow testing system
Tests end-to-end functionality with realistic scenarios
Run with: pytest tests/test_flow_integration.py -v
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from testllm import (
    LocalAgent, conversation_flow, ConversationFlow,
    ToolUsagePatterns, BusinessLogicPatterns, ContextPatterns,
    IntegrationPatterns, PerformancePatterns
)
from testllm.core import AgentUnderTest
from testllm.flows import FlowResult
from testllm.evaluation_loop import ConsensusResult


class ProductionMockAgent(AgentUnderTest):
    """
    Realistic production agent mock for integration testing.
    Simulates a complex agent with multiple capabilities.
    """
    
    def __init__(self):
        super().__init__()
        self.conversation_memory = {}
        self.user_preferences = {}
        self.session_context = []
        self.system_state = {"authenticated": False, "user_type": None}
        
        # Comprehensive response patterns
        self.response_patterns = {
            # Customer onboarding
            "new customer": self._handle_new_customer,
            "my name is": self._handle_name_introduction,
            "returning customer": self._handle_returning_customer,
            "premium customer": self._handle_premium_customer,
            
            # Tool usage simulation
            "book flight": self._handle_flight_booking,
            "search": self._handle_search_request,
            "weather": self._handle_weather_request,
            "analyze data": self._handle_data_analysis,
            "stock price": self._handle_stock_request,
            
            # Business operations
            "purchase": self._handle_purchase_request,
            "upgrade": self._handle_upgrade_request,
            "payment": self._handle_payment_processing,
            "refund": self._handle_refund_request,
            
            # Error scenarios
            "invalid": self._handle_invalid_input,
            "error": self._handle_error_scenario,
            
            # Context and memory tests
            "what was my name": self._recall_name,
            "what am I looking for": self._recall_context,
            "remember": self._handle_memory_request,
            
            # Preference handling
            "I prefer": self._handle_preference_setting,
            "detailed explanation": self._provide_detailed_response,
            
            # Performance scenarios
            "complex analysis": self._handle_complex_request,
            "large dataset": self._handle_large_data_request,
        }
    
    def send_message(self, message: str) -> str:
        """Process message with realistic agent behavior"""
        message_lower = message.lower()
        
        # Add to session context
        self.session_context.append({"type": "user", "content": message})
        
        # Find matching handler
        for pattern, handler in self.response_patterns.items():
            if pattern in message_lower:
                response = handler(message)
                self.session_context.append({"type": "agent", "content": response})
                return response
        
        # Default intelligent response
        response = self._generate_default_response(message)
        self.session_context.append({"type": "agent", "content": response})
        return response
    
    def reset_conversation(self):
        """Reset all conversation state"""
        self.conversation_memory = {}
        self.user_preferences = {}
        self.session_context = []
        self.system_state = {"authenticated": False, "user_type": None}
    
    # Handler methods for different scenarios
    def _handle_new_customer(self, message):
        self.system_state["user_type"] = "new"
        return "Welcome! I see you're a new customer. I'll guide you through our onboarding process step by step. First, I'll need to collect some basic information to set up your account properly."
    
    def _handle_name_introduction(self, message):
        # Extract name from message
        name_part = message.lower().split("name is")[-1].strip()
        name = name_part.split()[0] if name_part else "Customer"
        self.conversation_memory["user_name"] = name.title()
        return f"Nice to meet you, {name.title()}! I've recorded your name in our system. How can I assist you today?"
    
    def _handle_returning_customer(self, message):
        self.system_state["user_type"] = "returning"
        return "Welcome back! I've identified you as a returning customer. I can see your account history and am ready to help you with any questions or requests."
    
    def _handle_premium_customer(self, message):
        self.system_state["user_type"] = "premium"
        return "Hello! I see you have premium status with us. I'll prioritize your request and provide our enhanced level of service. What can I help you with today?"
    
    def _handle_flight_booking(self, message):
        return "I'm accessing our flight booking system to search for available options. I'm checking real-time availability across multiple airlines and will factor in your preferences for timing and pricing."
    
    def _handle_search_request(self, message):
        if "file" in message.lower():
            return "I'm performing a comprehensive file search across your accessible directories. This includes scanning metadata and content where permissions allow."
        return "I'm initiating a search query and will provide relevant results ranked by relevance and recency."
    
    def _handle_weather_request(self, message):
        return "I'm retrieving current weather data from our meteorological service. This includes real-time conditions, forecasts, and any relevant weather alerts for your location."
    
    def _handle_data_analysis(self, message):
        return "I'm beginning the data analysis process. Depending on the dataset size and complexity, this may take several minutes. I'll provide progress updates and can offer preliminary insights while the full analysis completes."
    
    def _handle_stock_request(self, message):
        return "I'm accessing real-time financial market data. The information I provide will include the most current prices, trading volume, and relevant market indicators. Data is typically delayed by 15 minutes unless you have real-time access."
    
    def _handle_purchase_request(self, message):
        return "I understand you're interested in making a purchase. I'll guide you through our secure purchasing process, including product selection, pricing confirmation, and payment processing."
    
    def _handle_upgrade_request(self, message):
        user_type = self.system_state.get("user_type", "standard")
        if user_type == "premium":
            return "As a premium customer, you already have access to our highest tier. I can show you additional services or enterprise options if you're interested."
        return "I see you're interested in upgrading your account. Our premium tier includes advanced features, priority support, and enhanced capabilities. The current pricing is $29/month."
    
    def _handle_payment_processing(self, message):
        return "I'm processing your payment through our secure payment gateway. All transactions are encrypted and PCI-compliant. You'll receive a confirmation email once processing is complete."
    
    def _handle_refund_request(self, message):
        return "I understand you're requesting a refund. I'll review your account and recent transactions to process this according to our refund policy. Most refunds are processed within 3-5 business days."
    
    def _handle_invalid_input(self, message):
        return "I'm sorry, but I couldn't process that request as it appears to contain invalid information. Could you please clarify or provide the information in a different format?"
    
    def _handle_error_scenario(self, message):
        return "I encountered an issue processing your request. Let me try an alternative approach. If this continues, I can escalate to our technical support team."
    
    def _recall_name(self, message):
        name = self.conversation_memory.get("user_name")
        if name:
            return f"Your name is {name}, as you mentioned earlier in our conversation."
        return "I don't have your name recorded in our current conversation. Could you please remind me?"
    
    def _recall_context(self, message):
        # Look for recent context about what user was looking for
        for entry in reversed(self.session_context[-10:]):  # Last 10 entries
            if entry["type"] == "user" and any(word in entry["content"].lower() for word in ["need", "looking", "want", "buy"]):
                return f"Based on our conversation, you were asking about: {entry['content']}"
        return "I'd be happy to help you remember what you were looking for. Could you provide a bit more context?"
    
    def _handle_memory_request(self, message):
        return "I maintain context throughout our conversation and can reference previous topics, preferences, and requests you've made during this session."
    
    def _handle_preference_setting(self, message):
        if "detailed" in message.lower():
            self.user_preferences["explanation_style"] = "detailed"
            return "I've noted that you prefer detailed explanations. I'll make sure to provide comprehensive information in my responses going forward."
        return "I've recorded your preference and will adjust my responses accordingly."
    
    def _provide_detailed_response(self, message):
        style = self.user_preferences.get("explanation_style", "standard")
        if style == "detailed":
            return "As requested, I'll provide a detailed explanation. [This would be a comprehensive, in-depth response covering multiple aspects of the topic, including background context, step-by-step processes, and relevant examples.]"
        return "I'll provide the information you requested with appropriate detail level."
    
    def _handle_complex_request(self, message):
        return "This is a complex analysis request that will require significant processing time. I estimate this will take approximately 15-20 minutes to complete thoroughly. I can provide a preliminary overview now and send detailed results when the full analysis is finished."
    
    def _handle_large_data_request(self, message):
        return "Processing large datasets requires careful resource management. For a 50GB dataset, realistic processing time would be 2-4 hours depending on the analysis type. I can offer to process a representative sample for quicker insights, or we can discuss prioritizing specific portions of the data."
    
    def _generate_default_response(self, message):
        user_name = self.conversation_memory.get("user_name", "")
        name_part = f"{user_name}, " if user_name else ""
        return f"I understand your request{', ' + name_part if name_part else ''}about '{message}'. Let me help you with that. Could you provide a bit more detail about what specifically you're looking for?"


@pytest.fixture
def production_agent():
    """Fixture providing a production-like agent"""
    return LocalAgent(model=ProductionMockAgent())


def mock_successful_evaluation():
    """Helper to mock successful flow evaluation"""
    def mock_flow_execution_context():
        return patch('testllm.flows.EvaluationLoop', return_value=create_mock_evaluator())
    
    def create_mock_evaluator():
        mock_evaluator = AsyncMock()
        
        def mock_evaluate_response(user_input, agent_response, criteria):
            # Return a successful consensus result for each criterion
            return [
                ConsensusResult(criterion.criterion, 1.0, True, [])
                for criterion in criteria
            ]
        
        mock_evaluator.evaluate_response.side_effect = mock_evaluate_response
        return mock_evaluator
    
    return mock_flow_execution_context()


class TestEndToEndFlowExecution:
    """Test complete end-to-end flow execution scenarios"""
    
    def test_customer_onboarding_complete_flow(self, production_agent):
        """Test complete customer onboarding from start to finish"""
        with mock_successful_evaluation():
            flow = conversation_flow("e2e_onboarding", "End-to-end customer onboarding")
            
            # Step 1: Initial contact
            flow.step(
                "Hello, I'm a new customer interested in your services",
                criteria=[
                    "Response should acknowledge new customer status",
                    "Response should begin onboarding process",
                    "Response should be professional and welcoming"
                ]
            )
            
            # Step 2: Information gathering
            flow.step(
                "My name is Sarah Johnson and I need a business account",
                criteria=[
                    "Response should acknowledge the name Sarah Johnson",
                    "Response should understand business account requirement",
                    "Response should ask for relevant business information"
                ],
                expect_context_retention=True
            )
            
            # Step 3: Memory validation
            flow.context_check(
                "What type of account was I requesting?",
                context_criteria=[
                    "Response should remember business account request",
                    "Response should demonstrate conversation awareness"
                ]
            )
            
            # Step 4: Process completion
            flow.business_logic_check(
                "I'm ready to complete the setup process",
                business_rules=["account_creation", "verification"],
                criteria=[
                    "Response should outline completion steps",
                    "Response should mention verification requirements"
                ]
            )
            
            result = flow.execute_sync(production_agent)
            
            # Comprehensive validation
            assert result.passed, f"Onboarding flow failed: {result.flow_errors}"
            assert result.steps_executed == 4
            assert result.overall_score >= 0.7
            assert result.context_retention_score >= 0.7
            assert result.business_logic_score >= 0.7
    
    def test_complex_travel_booking_integration(self, production_agent):
        """Test complex multi-system travel booking scenario"""
        with mock_successful_evaluation():
            flow = conversation_flow("travel_integration", "Complex travel booking")
            
            # Step 1: Initial travel request
            flow.tool_usage_check(
                "I need to book a round-trip flight from Seattle to New York next week",
                expected_tools=["flight_search", "availability_check"],
                criteria=[
                    "Response should indicate flight search process",
                    "Response should acknowledge specific route",
                    "Response should ask for travel dates"
                ]
            )
            
            # Step 2: Add accommodation
            flow.tool_usage_check(
                "I also need a hotel in Manhattan, preferably under $200 per night",
                expected_tools=["hotel_search", "location_filter"],
                criteria=[
                    "Response should coordinate with flight booking",
                    "Response should acknowledge price constraint",
                    "Response should indicate Manhattan search"
                ]
            )
            
            # Step 3: Complex coordination
            flow.business_logic_check(
                "Make sure the hotel checkout aligns with my return flight",
                business_rules=["travel_coordination", "schedule_optimization"],
                criteria=[
                    "Response should understand coordination need",
                    "Response should reference previous booking context",
                    "Response should demonstrate travel planning logic"
                ]
            )
            
            result = flow.execute_sync(production_agent)
            
            assert result.passed
            assert result.tool_usage_score >= 0.7
            assert result.business_logic_score >= 0.7
            assert len(result.step_results) == 3
    
    def test_error_recovery_resilience_flow(self, production_agent):
        """Test comprehensive error handling and recovery"""
        with mock_successful_evaluation():
            flow = BusinessLogicPatterns.error_handling_workflow()
            
            # Add additional error scenarios
            flow.step(
                "Can you process this obviously malformed request: ;;;invalid;;;",
                criteria=[
                    "Response should handle malformed input gracefully",
                    "Response should not crash or return unhelpful errors",
                    "Response should ask for clarification"
                ]
            )
            
            result = flow.execute_sync(production_agent)
            
            assert result.passed
            assert result.business_logic_score >= 0.7
            assert result.steps_executed >= 2  # Original + additional error scenario
    
    def test_advanced_context_retention_flow(self, production_agent):
        """Test sophisticated context management across complex conversation"""
        with mock_successful_evaluation():
            flow = ContextPatterns.multi_turn_memory()
            
            # Add complex context scenarios
            flow.step(
                "I also mentioned I work in machine learning and need GPU support",
                criteria=[
                    "Response should connect to previous laptop discussion",
                    "Response should understand ML specialization",
                    "Response should maintain all context elements"
                ],
                expect_context_retention=True
            )
            
            flow.context_check(
                "Summarize everything you know about my requirements",
                context_criteria=[
                    "Response should mention name (John)",
                    "Response should mention laptop for software development",
                    "Response should mention machine learning and GPU needs",
                    "Response should demonstrate comprehensive context integration"
                ]
            )
            
            result = flow.execute_sync(production_agent)
            
            assert result.passed
            assert result.context_retention_score >= 0.8
            assert result.steps_executed >= 6  # Original + additional context steps


class TestBehavioralPatternIntegration:
    """Test integration between different behavioral patterns"""
    
    def test_combined_tool_and_business_patterns(self, production_agent):
        """Test combination of tool usage and business logic patterns"""
        with mock_successful_evaluation():
            # Tool usage pattern
            search_flow = ToolUsagePatterns.search_pattern("search customer database", "database")
            
            # Business logic pattern  
            auth_flow = BusinessLogicPatterns.user_authentication_flow("premium")
            
            # Execute both
            search_result = search_flow.execute_sync(production_agent)
            auth_result = auth_flow.execute_sync(production_agent)
            
            assert search_result.passed
            assert auth_result.passed
            assert search_result.tool_usage_score >= 0.6
            assert auth_result.business_logic_score >= 0.7
    
    def test_context_across_different_patterns(self, production_agent):
        """Test context retention across different behavioral patterns"""
        with mock_successful_evaluation():
            # Start with context establishment
            context_flow = ContextPatterns.preference_tracking()
            context_result = context_flow.execute_sync(production_agent)
            
            # Continue with tool usage that should respect context
            tool_flow = conversation_flow("context_aware_tools", "Tools with context")
            tool_flow.step(
                "Search for detailed information about quantum computing",
                criteria=[
                    "Response should provide detailed information",
                    "Response should respect previously stated preference for detail"
                ],
                expect_context_retention=True
            )
            
            tool_result = tool_flow.execute_sync(production_agent)
            
            assert context_result.passed
            assert tool_result.passed
            assert context_result.context_retention_score >= 0.7
    
    def test_performance_pattern_integration(self, production_agent):
        """Test performance patterns with other behavioral patterns"""
        with mock_successful_evaluation():
            # Performance pattern
            perf_flow = PerformancePatterns.complex_request_handling()
            
            # Add follow-up that tests adaptability
            perf_flow.step(
                "Actually, can you give me just the key highlights instead?",
                criteria=[
                    "Response should understand request for simpler alternative",
                    "Response should adapt to changed requirements",
                    "Response should maintain helpfulness"
                ],
                expect_context_retention=True
            )
            
            result = perf_flow.execute_sync(production_agent)
            
            assert result.passed
            assert result.steps_executed == 2
            assert result.context_retention_score >= 0.6


class TestFlowCustomizationAndExtension:
    """Test flow customization and extension capabilities"""
    
    def test_custom_business_flow_creation(self, production_agent):
        """Test creating custom business flows for specific use cases"""
        with mock_successful_evaluation():
            # Create custom VIP customer flow
            flow = conversation_flow("vip_customer_flow", "VIP customer service flow")
            
            flow.business_logic_check(
                "I'm a VIP customer with an urgent production issue",
                business_rules=["vip_escalation", "priority_handling"],
                criteria=[
                    "Response should acknowledge VIP status",
                    "Response should prioritize urgent request",
                    "Response should offer expedited handling"
                ]
            )
            
            flow.business_logic_check(
                "This is affecting my live production system right now",
                business_rules=["emergency_response", "escalation"],
                criteria=[
                    "Response should understand production impact",
                    "Response should escalate appropriately",
                    "Response should provide immediate action steps"
                ]
            )
            
            result = flow.execute_sync(production_agent)
            
            assert result.passed
            assert result.business_logic_score >= 0.8
    
    def test_flow_extension_capabilities(self, production_agent):
        """Test extending existing flows with additional steps"""
        with mock_successful_evaluation():
            # Start with existing pattern
            base_flow = ToolUsagePatterns.api_integration_pattern("get weather data", "weather")
            
            # Extend with additional verification steps
            base_flow.step(
                "Is this data from the last hour?",
                criteria=[
                    "Response should indicate data freshness",
                    "Response should provide timestamp information"
                ],
                expect_context_retention=True
            )
            
            base_flow.business_logic_check(
                "I need this for a critical weather alert system",
                business_rules=["data_quality", "critical_system_support"],
                criteria=[
                    "Response should understand criticality",
                    "Response should ensure data quality standards"
                ]
            )
            
            result = base_flow.execute_sync(production_agent)
            
            assert result.passed
            assert result.steps_executed >= 3  # Original + 2 additional
            assert result.tool_usage_score >= 0.6
            assert result.business_logic_score >= 0.6
    
    def test_parallel_flow_concepts(self, production_agent):
        """Test concepts for parallel flow execution"""
        with mock_successful_evaluation():
            # Create multiple independent flows
            flows = [
                ToolUsagePatterns.search_pattern("search files", "file"),
                BusinessLogicPatterns.user_authentication_flow("new"),
                ContextPatterns.preference_tracking()
            ]
            
            # Execute all flows
            results = []
            for flow in flows:
                result = flow.execute_sync(production_agent)
                results.append(result)
            
            # All should pass independently
            assert all(r.passed for r in results)
            assert len(results) == 3
            
            # Each should have appropriate scores
            tool_result, auth_result, context_result = results
            assert tool_result.tool_usage_score >= 0.6
            assert auth_result.business_logic_score >= 0.7
            assert context_result.context_retention_score >= 0.7


class TestRealWorldScenarios:
    """Test realistic production scenarios"""
    
    def test_e_commerce_complete_purchase_flow(self, production_agent):
        """Test complete e-commerce purchase workflow"""
        with mock_successful_evaluation():
            flow = conversation_flow("ecommerce_purchase", "E-commerce purchase flow")
            
            # Product discovery
            flow.tool_usage_check(
                "I'm looking for a high-performance laptop for machine learning",
                expected_tools=["product_search", "specification_filter"],
                criteria=[
                    "Response should search product catalog",
                    "Response should understand ML performance requirements"
                ]
            )
            
            # Product selection and business logic
            flow.business_logic_check(
                "I want the Dell XPS with 32GB RAM, what's the price and availability?",
                business_rules=["inventory_check", "pricing_calculation"],
                criteria=[
                    "Response should check specific product availability",
                    "Response should provide current pricing",
                    "Response should offer purchasing options"
                ]
            )
            
            # Purchase process
            flow.business_logic_check(
                "I'd like to purchase this with expedited shipping",
                business_rules=["payment_processing", "shipping_options"],
                criteria=[
                    "Response should initiate purchase process",
                    "Response should confirm expedited shipping option",
                    "Response should outline next steps"
                ]
            )
            
            # Confirmation and follow-up
            flow.context_check(
                "When will this arrive?",
                context_criteria=[
                    "Response should reference the expedited shipping",
                    "Response should provide delivery timeline",
                    "Response should maintain purchase context"
                ]
            )
            
            result = flow.execute_sync(production_agent)
            
            assert result.passed
            assert result.steps_executed == 4
            assert result.business_logic_score >= 0.8
            assert result.context_retention_score >= 0.7
    
    def test_customer_support_escalation_flow(self, production_agent):
        """Test customer support escalation scenario"""
        with mock_successful_evaluation():
            flow = conversation_flow("support_escalation", "Customer support escalation")
            
            # Initial support request
            flow.step(
                "I've been having trouble with my account for three days and no one has helped me",
                criteria=[
                    "Response should acknowledge the frustration",
                    "Response should show empathy for the situation",
                    "Response should offer immediate assistance"
                ]
            )
            
            # Escalation trigger
            flow.business_logic_check(
                "This is the fourth time I'm contacting support about the same issue",
                business_rules=["escalation_trigger", "case_history_review"],
                criteria=[
                    "Response should recognize escalation need",
                    "Response should reference case history",
                    "Response should offer higher-level support"
                ]
            )
            
            # Resolution approach
            flow.business_logic_check(
                "I need this resolved today as it's affecting my business operations",
                business_rules=["priority_handling", "business_impact_assessment"],
                criteria=[
                    "Response should understand business impact",
                    "Response should commit to resolution timeline",
                    "Response should provide escalation path"
                ]
            )
            
            result = flow.execute_sync(production_agent)
            
            assert result.passed
            assert result.business_logic_score >= 0.8
    
    def test_api_integration_reliability_flow(self, production_agent):
        """Test API integration with reliability considerations"""
        with mock_successful_evaluation():
            flow = IntegrationPatterns.real_time_data_pattern("financial")
            
            # Add reliability testing
            flow.step(
                "What happens if the financial data service is temporarily unavailable?",
                criteria=[
                    "Response should address service reliability",
                    "Response should explain fallback mechanisms",
                    "Response should discuss data staleness handling"
                ]
            )
            
            flow.business_logic_check(
                "I need this data for automated trading, so reliability is critical",
                business_rules=["high_availability", "data_integrity"],
                criteria=[
                    "Response should understand criticality",
                    "Response should address reliability requirements",
                    "Response should mention SLA considerations"
                ]
            )
            
            result = flow.execute_sync(production_agent)
            
            assert result.passed
            assert result.tool_usage_score >= 0.7
            assert result.business_logic_score >= 0.8


if __name__ == "__main__":
    # Run integration tests directly
    import sys
    import subprocess
    
    print("Running flow integration tests...")
    print("=" * 50)
    
    # Run with pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v",
        "--tb=short"
    ], capture_output=False)
    
    sys.exit(result.returncode)