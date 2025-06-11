"""
Production-Ready Flow Testing Examples
Demonstrates real-world agentic system testing scenarios
"""

import pytest
from testllm import LocalAgent
from testllm.flows import conversation_flow, ConversationFlow
from testllm.behavioral import (
    ToolUsagePatterns, BusinessLogicPatterns, 
    ContextPatterns, IntegrationPatterns, PerformancePatterns
)


@pytest.fixture
def production_agent():
    """
    Mock production agent with realistic behaviors.
    Replace this with your actual agent for real testing.
    """
    def production_agent_logic(prompt):
        prompt_lower = prompt.lower()
        
        # Customer service responses
        if "new customer" in prompt_lower:
            return "Welcome! I see you're a new customer. Let me help you get started with our onboarding process. I'll need to collect some basic information to set up your account."
        
        # Context retention
        if "name is" in prompt_lower:
            name = prompt.split("name is")[-1].strip()
            return f"Nice to meet you, {name}! I've noted your name in our system. How can I assist you today?"
        
        if "what was my name" in prompt_lower:
            return "I don't see your name in our current conversation. Could you please remind me?"
        
        # Tool usage simulation
        if "book" in prompt_lower and "flight" in prompt_lower:
            return "I'm searching our flight database for available options. This may take a moment as I check real-time availability and pricing across multiple airlines."
        
        if "weather" in prompt_lower:
            return "Let me check the current weather conditions for you. I'm pulling real-time data from our weather service."
        
        # Business logic
        if "premium" in prompt_lower and ("plan" in prompt_lower or "upgrade" in prompt_lower):
            return "I see you're interested in our premium plan. It includes advanced features like priority support, expanded storage, and premium integrations. The current price is $29/month. Would you like me to start the upgrade process?"
        
        # Error handling
        if "invalid_location" in prompt_lower or "12345" in prompt_lower:
            return "I'm sorry, but I couldn't find that location in our database. Could you please provide a valid city name or ZIP code?"
        
        if "seattle" in prompt_lower and "weather" in prompt_lower:
            return "Perfect! I found Seattle. The current weather in Seattle is 58°F with light rain and overcast skies. This information was last updated 15 minutes ago."
        
        # Complex requests
        if "analyze" in prompt_lower and ("years" in prompt_lower or "data" in prompt_lower):
            return "This is a complex analysis request that will require processing large amounts of historical data. I estimate this will take approximately 15-20 minutes to complete. I'll break this down into phases: data collection, analysis, and report generation. Should I proceed?"
        
        # Default helpful response
        return "I understand your request. Let me help you with that. Could you provide a bit more detail about what specifically you're looking for?"
    
    class ProductionAgentClass:
        def __call__(self, content):
            return production_agent_logic(content)
        
        def send_message(self, content):
            return production_agent_logic(content)
        
        def reset_conversation(self):
            pass
    
    return LocalAgent(model=ProductionAgentClass())


# Example 1: Customer Onboarding Flow
def test_customer_onboarding_flow(production_agent):
    """Test complete customer onboarding process"""
    flow = conversation_flow("customer_onboarding", "Complete customer onboarding workflow")
    
    # Step 1: Initial contact
    flow.step(
        "Hello, I'm a new customer and want to get started",
        criteria=[
            "Response should acknowledge new customer status",
            "Response should begin onboarding process",
            "Response should be welcoming and professional",
            "Response should indicate what information will be needed"
        ]
    )
    
    # Step 2: Information gathering
    flow.step(
        "My name is Sarah Johnson and I need a business account",
        criteria=[
            "Response should acknowledge the name Sarah Johnson",
            "Response should understand business account requirement",
            "Response should ask for business-specific information",
            "Response should maintain professional tone"
        ],
        expect_context_retention=True
    )
    
    # Step 3: Context validation
    flow.context_check(
        "What type of account was I requesting?",
        context_criteria=[
            "Response should remember the business account request",
            "Response should show conversation awareness",
            "Response should not ask for information already provided"
        ]
    )
    
    # Step 4: Process completion
    flow.business_logic_check(
        "I'm ready to complete the setup",
        business_rules=["account_creation", "verification_process"],
        criteria=[
            "Response should outline completion steps",
            "Response should mention verification requirements",
            "Response should provide timeline for account activation"
        ]
    )
    
    result = flow.execute_sync(production_agent)
    
    # Assert comprehensive flow success
    assert result.passed, f"Onboarding flow failed: {result.flow_errors}"
    assert result.context_retention_score >= 0.7, f"Poor context retention: {result.context_retention_score}"
    assert result.business_logic_score >= 0.7, f"Poor business logic: {result.business_logic_score}"


# Example 2: E-commerce Purchase Flow
def test_ecommerce_purchase_flow(production_agent):
    """Test complete e-commerce purchase workflow"""
    flow = BusinessLogicPatterns.purchase_workflow()
    
    result = flow.execute_sync(production_agent)
    
    assert result.passed, f"Purchase flow failed: {result.flow_errors}"
    assert result.business_logic_score >= 0.8, "Purchase logic not meeting standards"
    
    # Verify all critical steps completed
    assert result.steps_executed == result.total_steps, "Not all purchase steps completed"


# Example 3: Multi-System Travel Booking
def test_travel_booking_integration(production_agent):
    """Test complex multi-system travel booking"""
    flow = conversation_flow("travel_booking", "Multi-system travel booking test")
    
    # Step 1: Initial travel request
    flow.tool_usage_check(
        "I need to book a round-trip flight from Seattle to New York for next week",
        expected_tools=["flight_search", "availability_check", "pricing"],
        criteria=[
            "Response should indicate searching for flights",
            "Response should acknowledge Seattle to New York route",
            "Response should ask for specific travel dates",
            "Response should show understanding of round-trip requirement"
        ]
    )
    
    # Step 2: Add complexity
    flow.tool_usage_check(
        "I also need a hotel near Manhattan, preferably under $200/night",
        expected_tools=["hotel_search", "location_filter", "price_filter"],
        criteria=[
            "Response should indicate searching for hotels",
            "Response should understand Manhattan location requirement",
            "Response should acknowledge price constraint",
            "Response should coordinate with previous flight request"
        ]
    )
    
    # Step 3: Coordination check
    flow.business_logic_check(
        "Make sure the hotel checkout aligns with my return flight",
        business_rules=["travel_coordination", "schedule_optimization"],
        criteria=[
            "Response should understand coordination need",
            "Response should reference previous booking information",
            "Response should show travel planning logic"
        ]
    )
    
    result = flow.execute_sync(production_agent)
    
    assert result.passed, f"Travel booking flow failed: {result.flow_errors}"
    assert result.tool_usage_score >= 0.7, "Tool usage patterns not meeting standards"


# Example 4: Error Recovery and Resilience
def test_error_recovery_resilience(production_agent):
    """Test error handling and system resilience"""
    flow = BusinessLogicPatterns.error_handling_workflow()
    
    result = flow.execute_sync(production_agent)
    
    assert result.passed, f"Error recovery flow failed: {result.flow_errors}"
    
    # Verify specific error recovery behaviors
    error_step_results = [r for r in result.step_results if "error" in r.test_id.lower()]
    assert len(error_step_results) >= 2, "Not enough error recovery steps tested"
    
    for error_result in error_step_results:
        assert error_result.passed, f"Error handling failed in step: {error_result.test_id}"


# Example 5: Advanced Context and Memory Testing
def test_advanced_context_management(production_agent):
    """Test sophisticated context retention across complex conversation"""
    flow = ContextPatterns.multi_turn_memory()
    
    # Add additional complex context steps
    flow.step(
        "I also mentioned I need it for machine learning work specifically",
        criteria=[
            "Response should connect to previous software development context",
            "Response should understand ML specialization",
            "Response should maintain all previous context (name, laptop, development)"
        ],
        expect_context_retention=True
    )
    
    flow.context_check(
        "Summarize what you know about my requirements",
        context_criteria=[
            "Response should mention name (John)",
            "Response should mention laptop shopping",
            "Response should mention software development",
            "Response should mention machine learning specialization",
            "Response should show comprehensive context integration"
        ]
    )
    
    result = flow.execute_sync(production_agent)
    
    assert result.passed, f"Context management flow failed: {result.flow_errors}"
    assert result.context_retention_score >= 0.8, f"Context retention below standards: {result.context_retention_score}"


# Example 6: Performance and Scalability Testing
def test_performance_behavior_patterns(production_agent):
    """Test agent behavior under performance constraints"""
    flow = PerformancePatterns.complex_request_handling()
    
    # Add follow-up performance test
    flow.step(
        "Can you give me a quick summary instead?",
        criteria=[
            "Response should understand the request for a simpler alternative",
            "Response should offer reduced scope analysis",
            "Response should maintain helpfulness while managing expectations"
        ],
        expect_context_retention=True
    )
    
    result = flow.execute_sync(production_agent)
    
    assert result.passed, f"Performance behavior flow failed: {result.flow_errors}"


# Example 7: Real-Time Data Integration Testing
def test_realtime_data_integration(production_agent):
    """Test real-time data handling patterns"""
    weather_flow = IntegrationPatterns.real_time_data_pattern("weather")
    
    result = weather_flow.execute_sync(production_agent)
    
    assert result.passed, f"Real-time data flow failed: {result.flow_errors}"
    assert result.tool_usage_score >= 0.7, "Real-time data integration not meeting standards"


# Example 8: Comprehensive System Integration Test
def test_comprehensive_system_integration(production_agent):
    """Test complete system integration across multiple domains"""
    flow = conversation_flow("comprehensive_integration", "Full system integration test")
    
    # Customer service + tool usage
    flow.step(
        "I'm a premium customer and need to check my recent orders",
        criteria=[
            "Response should acknowledge premium status",
            "Response should indicate checking order history",
            "Response should apply appropriate service level"
        ]
    )
    
    # Context + business logic
    flow.step(
        "I want to return the laptop I ordered last week",
        criteria=[
            "Response should reference previous order discussion",
            "Response should understand return request",
            "Response should explain return process for premium customers"
        ],
        expect_context_retention=True
    )
    
    # Error recovery + tool usage
    flow.step(
        "Actually, I can't find my order confirmation email",
        criteria=[
            "Response should offer alternative order lookup methods",
            "Response should maintain helpful tone despite complication",
            "Response should suggest specific solutions"
        ]
    )
    
    # Final integration test
    flow.step(
        "Can you look up my order using my phone number?",
        criteria=[
            "Response should indicate looking up by phone number",
            "Response should maintain conversation context",
            "Response should show system integration capabilities"
        ]
    )
    
    result = flow.execute_sync(production_agent)
    
    # Comprehensive assertions
    assert result.passed, f"Comprehensive integration failed: {result.flow_errors}"
    assert result.overall_score >= 0.75, f"Overall integration score too low: {result.overall_score}"
    assert result.context_retention_score >= 0.7, "Context retention insufficient"
    assert result.business_logic_score >= 0.7, "Business logic compliance insufficient"
    assert result.tool_usage_score >= 0.6, "Tool usage patterns insufficient"
    
    # Verify conversation coherence
    assert len(result.step_results) >= 4, "Not all integration steps completed"
    
    print(f"✅ Comprehensive Integration Test Results:")
    print(f"   Overall Score: {result.overall_score:.2f}")
    print(f"   Context Retention: {result.context_retention_score:.2f}")
    print(f"   Business Logic: {result.business_logic_score:.2f}")
    print(f"   Tool Usage: {result.tool_usage_score:.2f}")
    print(f"   Steps Completed: {result.steps_executed}/{result.total_steps}")


# Example 9: Custom Business Logic Flow
def test_custom_business_workflow(production_agent):
    """Test custom business-specific workflow"""
    flow = conversation_flow("custom_business", "Custom business workflow")
    
    # Define custom business scenario
    flow.business_logic_check(
        "I'm a VIP customer and need urgent support for a critical issue",
        business_rules=["vip_escalation", "priority_handling", "sla_compliance"],
        criteria=[
            "Response should acknowledge VIP status",
            "Response should prioritize the urgent request",
            "Response should indicate expedited handling",
            "Response should follow VIP service protocols"
        ]
    )
    
    flow.business_logic_check(
        "This is affecting my production system right now",
        business_rules=["emergency_response", "escalation_protocol"],
        criteria=[
            "Response should understand production impact",
            "Response should escalate appropriately",
            "Response should provide immediate action steps",
            "Response should offer direct contact options"
        ]
    )
    
    result = flow.execute_sync(production_agent)
    
    assert result.passed, "Custom business workflow failed"
    assert result.business_logic_score >= 0.8, "Business logic not meeting VIP standards"


if __name__ == "__main__":
    """
    Run production flow examples directly for demonstration
    """
    print("🚀 Running Production Flow Testing Examples...")
    print("=" * 60)
    
    # Create demo agent
    def demo_agent(prompt):
        if "new customer" in prompt.lower():
            return "Welcome! I'll help you get started with our onboarding process."
        elif "name is" in prompt.lower():
            return "Nice to meet you! I've noted your information."
        elif "flight" in prompt.lower():
            return "I'm searching for available flights. This may take a moment."
        return "I understand your request. How can I help you further?"
    
    class DemoAgent:
        def __call__(self, content):
            return demo_agent(content)
        
        def send_message(self, content):
            return demo_agent(content)
        
        def reset_conversation(self):
            pass
    
    agent = LocalAgent(model=DemoAgent())
    
    # Run a sample flow
    flow = conversation_flow("demo_onboarding", "Demo customer onboarding")
    flow.step(
        "Hello, I'm a new customer",
        criteria=[
            "Response should acknowledge new customer status",
            "Response should begin onboarding process"
        ]
    )
    
    result = flow.execute_sync(agent)
    
    print(f"Demo Flow Results:")
    print(f"Passed: {result.passed}")
    print(f"Score: {result.overall_score:.2f}")
    print(f"Steps: {result.steps_executed}/{result.total_steps}")
    
    if result.step_results:
        for step_result in result.step_results:
            print(f"\nStep: {step_result.test_id}")
            print(f"Input: {step_result.user_input}")
            print(f"Response: {step_result.agent_response}")
            print(f"Passed: {step_result.passed}")
    
    print("\n✅ Production flow testing framework ready!")
    print("Use these patterns to test your real agentic systems.")