"""
Behavioral Pattern Testing - Specialized testing for common agent behaviors
"""

from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass
from .flows import ConversationFlow, conversation_flow
from .semantic import semantic_test


class ToolUsagePatterns:
    """
    Pre-built behavioral patterns for testing tool usage without internal visibility.
    
    Tests what tool usage LOOKS LIKE in agent responses rather than actual tool calls.
    """
    
    @staticmethod
    def search_pattern(query: str, expected_search_type: str) -> ConversationFlow:
        """
        Test pattern for search-like tool usage.
        
        Args:
            query: The search query to test
            expected_search_type: Type of search (e.g., "web", "database", "files")
            
        Returns:
            ConversationFlow configured for search pattern testing
        """
        flow = conversation_flow(
            f"search_pattern_{expected_search_type}",
            f"Test {expected_search_type} search behavioral pattern"
        )
        
        flow.tool_usage_check(
            query,
            expected_tools=[f"{expected_search_type}_search"],
            criteria=[
                f"Response should indicate performing a {expected_search_type} search",
                "Response should acknowledge the search request",
                "Response should show search processing or results",
                "Response should not expose internal API details"
            ]
        )
        
        return flow
    
    @staticmethod
    def api_integration_pattern(request: str, api_type: str) -> ConversationFlow:
        """
        Test pattern for API integration behaviors.
        
        Args:
            request: The user request that should trigger API usage
            api_type: Type of API (e.g., "weather", "stock", "payment")
            
        Returns:
            ConversationFlow configured for API integration testing
        """
        flow = conversation_flow(
            f"api_integration_{api_type}",
            f"Test {api_type} API integration behavioral pattern"
        )
        
        flow.tool_usage_check(
            request,
            expected_tools=[f"{api_type}_api"],
            criteria=[
                f"Response should indicate checking {api_type} information",
                "Response should show awareness of real-time data needs",
                "Response should provide specific information or explain delays",
                "Response should handle potential API failures gracefully"
            ]
        )
        
        return flow
    
    @staticmethod
    def data_processing_pattern(data_request: str, processing_type: str) -> ConversationFlow:
        """
        Test pattern for data processing behaviors.
        
        Args:
            data_request: Request involving data processing
            processing_type: Type of processing (e.g., "analysis", "transformation", "calculation")
            
        Returns:
            ConversationFlow configured for data processing testing
        """
        flow = conversation_flow(
            f"data_processing_{processing_type}",
            f"Test {processing_type} data processing behavioral pattern"
        )
        
        flow.tool_usage_check(
            data_request,
            expected_tools=[f"data_{processing_type}"],
            criteria=[
                f"Response should indicate performing {processing_type}",
                "Response should acknowledge the complexity if appropriate",
                "Response should provide structured results or progress updates",
                "Response should handle large datasets appropriately"
            ]
        )
        
        return flow


class BusinessLogicPatterns:
    """
    Pre-built patterns for testing business logic compliance.
    """
    
    @staticmethod
    def user_authentication_flow(user_type: str = "new") -> ConversationFlow:
        """
        Test user authentication and onboarding business logic.
        
        Args:
            user_type: Type of user ("new", "returning", "premium")
            
        Returns:
            ConversationFlow configured for authentication testing
        """
        flow = conversation_flow(
            f"auth_flow_{user_type}",
            f"Test authentication flow for {user_type} user"
        )
        
        # Step 1: Initial contact
        flow.business_logic_check(
            f"Hello, I'm a {user_type} user",
            business_rules=["user_identification", "access_level_determination"],
            criteria=[
                f"Response should acknowledge {user_type} user status",
                "Response should apply appropriate access rules",
                "Response should begin appropriate flow for user type"
            ]
        )
        
        # Step 2: Verification
        flow.business_logic_check(
            "I need to access my account",
            business_rules=["identity_verification", "security_protocols"],
            criteria=[
                "Response should request appropriate verification",
                "Response should follow security protocols",
                "Response should not compromise security for convenience"
            ]
        )
        
        return flow
    
    @staticmethod
    def purchase_workflow() -> ConversationFlow:
        """
        Test e-commerce purchase business logic.
        
        Returns:
            ConversationFlow configured for purchase workflow testing
        """
        flow = conversation_flow("purchase_workflow", "Test purchase business logic")
        
        # Step 1: Product inquiry
        flow.business_logic_check(
            "I want to buy your premium plan",
            business_rules=["product_availability", "pricing_rules"],
            criteria=[
                "Response should confirm product availability",
                "Response should provide current pricing",
                "Response should explain premium features"
            ]
        )
        
        # Step 2: Purchase process
        flow.business_logic_check(
            "How do I complete the purchase?",
            business_rules=["payment_processing", "order_management"],
            criteria=[
                "Response should outline purchase steps",
                "Response should mention payment options",
                "Response should address security and billing"
            ]
        )
        
        # Step 3: Confirmation
        flow.business_logic_check(
            "I've completed payment, what's next?",
            business_rules=["order_confirmation", "service_activation"],
            criteria=[
                "Response should confirm order processing",
                "Response should explain next steps",
                "Response should provide timeline for activation"
            ]
        )
        
        return flow
    
    @staticmethod
    def error_handling_workflow() -> ConversationFlow:
        """
        Test error handling and recovery business logic.
        
        Returns:
            ConversationFlow configured for error handling testing
        """
        flow = conversation_flow("error_handling", "Test error handling business logic")
        
        # Step 1: Invalid request
        flow.business_logic_check(
            "Get me data for location INVALID_LOCATION_12345",
            business_rules=["input_validation", "error_recovery"],
            criteria=[
                "Response should handle invalid input gracefully",
                "Response should not crash or return unhelpful errors",
                "Response should suggest alternatives or corrections"
            ]
        )
        
        # Step 2: Recovery attempt
        flow.business_logic_check(
            "I meant Seattle",
            business_rules=["error_recovery", "context_correction"],
            criteria=[
                "Response should understand the correction",
                "Response should apply the fix to previous request",
                "Response should show learning from the error"
            ]
        )
        
        return flow


class ContextPatterns:
    """
    Pre-built patterns for testing context retention and conversation awareness.
    """
    
    @staticmethod
    def multi_turn_memory() -> ConversationFlow:
        """
        Test multi-turn conversation memory and context retention.
        
        Returns:
            ConversationFlow configured for memory testing
        """
        flow = conversation_flow("memory_test", "Test conversation memory")
        
        # Step 1: Establish context
        flow.step(
            "Hi, my name is John and I'm looking for a new laptop",
            criteria=[
                "Response should acknowledge the name John",
                "Response should understand laptop shopping context"
            ]
        )
        
        # Step 2: Add details
        flow.step(
            "I need it for software development work",
            criteria=[
                "Response should connect this to previous laptop request",
                "Response should understand development requirements",
                "Response should maintain John's identity"
            ],
            expect_context_retention=True
        )
        
        # Step 3: Memory check
        flow.context_check(
            "What was my name again?",
            context_criteria=[
                "Response should remember and state the name John",
                "Response should show conversation awareness",
                "Response should not ask for information already provided"
            ]
        )
        
        # Step 4: Context application
        flow.context_check(
            "What was I shopping for?",
            context_criteria=[
                "Response should remember the laptop request",
                "Response should recall the software development purpose",
                "Response should demonstrate comprehensive context retention"
            ]
        )
        
        return flow
    
    @staticmethod
    def preference_tracking() -> ConversationFlow:
        """
        Test preference learning and application across conversation.
        
        Returns:
            ConversationFlow configured for preference tracking
        """
        flow = conversation_flow("preference_tracking", "Test preference learning")
        
        # Step 1: Express preference
        flow.step(
            "I prefer detailed explanations over quick summaries",
            criteria=[
                "Response should acknowledge the preference",
                "Response should demonstrate understanding of detail preference"
            ]
        )
        
        # Step 2: Test preference application
        flow.step(
            "Explain how machine learning works",
            criteria=[
                "Response should provide detailed explanation",
                "Response should not give overly brief summary",
                "Response should apply previously stated preference"
            ],
            expect_context_retention=True
        )
        
        # Step 3: Preference consistency
        flow.context_check(
            "Tell me about quantum computing",
            context_criteria=[
                "Response should again provide detailed explanation",
                "Response should maintain preference consistency",
                "Response should show preference retention across topics"
            ]
        )
        
        return flow


class IntegrationPatterns:
    """
    Pre-built patterns for testing integration behaviors and external system interaction.
    """
    
    @staticmethod
    def real_time_data_pattern(data_type: str) -> ConversationFlow:
        """
        Test real-time data integration patterns.
        
        Args:
            data_type: Type of real-time data (e.g., "stock", "weather", "news")
            
        Returns:
            ConversationFlow configured for real-time data testing
        """
        flow = conversation_flow(
            f"realtime_{data_type}",
            f"Test real-time {data_type} data integration"
        )
        
        flow.tool_usage_check(
            f"What's the current {data_type} information?",
            expected_tools=[f"{data_type}_api", "realtime_data"],
            criteria=[
                f"Response should indicate checking current {data_type} data",
                "Response should show awareness of data freshness",
                "Response should provide timestamp or recency information",
                "Response should handle potential data unavailability"
            ]
        )
        
        # Follow-up to test data staleness awareness
        flow.context_check(
            "Is that the most recent data?",
            context_criteria=[
                "Response should reference when data was obtained",
                "Response should show understanding of data freshness concepts",
                "Response should explain data update frequency if relevant"
            ]
        )
        
        return flow
    
    @staticmethod
    def multi_system_integration() -> ConversationFlow:
        """
        Test coordination across multiple integrated systems.
        
        Returns:
            ConversationFlow configured for multi-system testing
        """
        flow = conversation_flow(
            "multi_system_integration",
            "Test coordination across multiple systems"
        )
        
        flow.tool_usage_check(
            "Book me a flight and reserve a hotel for next week in San Francisco",
            expected_tools=["flight_booking", "hotel_reservation", "calendar_check"],
            criteria=[
                "Response should acknowledge both flight and hotel requests",
                "Response should indicate coordinating multiple bookings",
                "Response should ask for dates or check calendar availability",
                "Response should show understanding of travel coordination"
            ]
        )
        
        flow.business_logic_check(
            "Make sure the hotel is near the airport",
            business_rules=["location_coordination", "travel_optimization"],
            criteria=[
                "Response should understand location coordination need",
                "Response should indicate checking hotel proximity to airport",
                "Response should show travel planning logic"
            ]
        )
        
        return flow


class PerformancePatterns:
    """
    Pre-built patterns for testing performance-related behaviors.
    """
    
    @staticmethod
    def complex_request_handling() -> ConversationFlow:
        """
        Test how agent handles complex, time-consuming requests.
        
        Returns:
            ConversationFlow configured for performance behavior testing
        """
        flow = conversation_flow(
            "complex_request_handling",
            "Test complex request performance behavior"
        )
        
        flow.step(
            "Analyze the last 5 years of stock data for the top 100 companies and create a comprehensive report",
            criteria=[
                "Response should acknowledge the complexity of the request",
                "Response should break down the task or explain the process",
                "Response should set appropriate expectations for timing",
                "Response should not claim instant completion of complex analysis",
                "Response should offer alternatives if the request is too complex"
            ]
        )
        
        return flow
    
    @staticmethod
    def resource_limitation_handling() -> ConversationFlow:
        """
        Test how agent handles resource limitations and constraints.
        
        Returns:
            ConversationFlow configured for resource limitation testing
        """
        flow = conversation_flow(
            "resource_limitation_handling",
            "Test resource limitation behavioral patterns"
        )
        
        flow.step(
            "Process this 50GB dataset and give me results in the next 30 seconds",
            criteria=[
                "Response should recognize unrealistic constraints",
                "Response should explain limitations appropriately",
                "Response should offer realistic alternatives",
                "Response should not promise impossible performance"
            ]
        )
        
        return flow