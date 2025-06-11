"""
Test file for conversation flow functionality
Run with: pytest tests/test_flows.py -v
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from testllm.flows import (
    ConversationFlow, conversation_flow, FlowStep, FlowResult, FlowStepType
)
from testllm.core import AgentUnderTest
from testllm.semantic import SemanticTestResult
from testllm.evaluation_loop import ConsensusResult


class MockFlowAgent(AgentUnderTest):
    """Mock agent for testing flow functionality"""
    
    def __init__(self):
        super().__init__()
        self.conversation_history = []
        self.responses = {
            "Hello, I'm new": "Welcome! I'll help you get started with our onboarding.",
            "My name is John": "Nice to meet you, John! I've noted your name.",
            "What was my name?": "Your name is John, as you mentioned earlier.",
            "Book a flight": "I'm searching for available flights. This may take a moment.",
            "What's the weather?": "Let me check the current weather conditions for you.",
            "Invalid request": "I'm sorry, I couldn't process that request. Could you clarify?"
        }
    
    def send_message(self, message: str) -> str:
        """Mock send_message with realistic responses"""
        response = self.responses.get(message, f"I understand your request: '{message}'. How can I help?")
        self.conversation_history.append({"user": message, "agent": response})
        return response
    
    def reset_conversation(self):
        """Reset conversation state"""
        self.conversation_history = []


@pytest.fixture
def mock_flow_agent():
    """Fixture providing a mock flow agent"""
    return MockFlowAgent()


@pytest.fixture
def mock_consensus_result():
    """Fixture providing a mock consensus result"""
    return ConsensusResult(
        criterion="Response should be helpful",
        consensus_score=1.0,
        passed=True,
        individual_results=[]
    )


class TestConversationFlow:
    """Test the ConversationFlow class"""
    
    def test_flow_initialization(self):
        """Test ConversationFlow initialization"""
        flow = ConversationFlow("test_flow", "Test flow description")
        
        assert flow.flow_id == "test_flow"
        assert flow.description == "Test flow description"
        assert len(flow.steps) == 0
        assert len(flow.conversation_history) == 0
        assert flow.config.evaluator_models == ["claude-sonnet-4"]
        assert flow.config.consensus_threshold == 0.7
    
    def test_flow_initialization_with_config(self):
        """Test ConversationFlow with custom configuration"""
        flow = ConversationFlow(
            "custom_flow",
            "Custom flow",
            evaluator_models=["claude-sonnet-4"],
            consensus_threshold=0.8
        )
        
        assert flow.config.evaluator_models == ["claude-sonnet-4"]
        assert flow.config.consensus_threshold == 0.8
    
    def test_add_step(self):
        """Test adding basic steps to flow"""
        flow = ConversationFlow("test_flow")
        
        flow.step(
            "Hello",
            criteria=["Response should be friendly"],
            step_id="greeting"
        )
        
        assert len(flow.steps) == 1
        step = flow.steps[0]
        assert step.step_id == "greeting"
        assert step.user_input == "Hello"
        assert step.criteria == ["Response should be friendly"]
        assert step.step_type == FlowStepType.USER_INPUT
    
    def test_add_step_with_expectations(self):
        """Test adding steps with behavioral expectations"""
        flow = ConversationFlow("test_flow")
        
        flow.step(
            "Book a flight",
            criteria=["Response should indicate flight search"],
            expect_context_retention=True,
            expect_tool_usage=["flight_search", "booking"],
            expect_business_logic=["availability_check"]
        )
        
        step = flow.steps[0]
        assert step.expect_context_retention == True
        assert step.expect_tool_usage_indicators == ["flight_search", "booking"]
        assert step.expect_business_logic == ["availability_check"]
    
    def test_method_chaining(self):
        """Test that step addition supports method chaining"""
        flow = ConversationFlow("test_flow")
        
        result = flow.step("Hello", ["Be friendly"]).step("Goodbye", ["Be polite"])
        
        assert result is flow
        assert len(flow.steps) == 2
        assert flow.steps[0].user_input == "Hello"
        assert flow.steps[1].user_input == "Goodbye"
    
    def test_context_check(self):
        """Test context check step creation"""
        flow = ConversationFlow("test_flow")
        
        flow.context_check(
            "What was my name?",
            context_criteria=["Should remember name from earlier"],
            step_id="memory_test"
        )
        
        step = flow.steps[0]
        assert step.step_id == "memory_test"
        assert step.user_input == "What was my name?"
        assert step.expect_context_retention == True
        assert "Should remember name from earlier" in step.criteria
    
    def test_tool_usage_check(self):
        """Test tool usage check step creation"""
        flow = ConversationFlow("test_flow")
        
        flow.tool_usage_check(
            "Book me a flight",
            expected_tools=["flight_search"],
            criteria=["Should indicate searching"]
        )
        
        step = flow.steps[0]
        assert step.user_input == "Book me a flight"
        assert step.expect_tool_usage_indicators == ["flight_search"]
        assert "Should indicate searching" in step.criteria
    
    def test_business_logic_check(self):
        """Test business logic check step creation"""
        flow = ConversationFlow("test_flow")
        
        flow.business_logic_check(
            "Process payment",
            business_rules=["payment_validation"],
            criteria=["Should validate payment"]
        )
        
        step = flow.steps[0]
        assert step.user_input == "Process payment"
        assert step.expect_business_logic == ["payment_validation"]
        assert "Should validate payment" in step.criteria
    
    def test_execute_sync_with_mock(self, mock_flow_agent):
        """Test synchronous flow execution with mocked evaluation"""
        flow = ConversationFlow("test_flow")
        flow.step("Hello", ["Response should be friendly"])
        
        # Mock the evaluation loop
        with patch('testllm.flows.EvaluationLoop') as mock_eval_class:
            mock_evaluator = AsyncMock()
            
            def mock_evaluate_response(user_input, agent_response, criteria):
                return [
                    ConsensusResult(criterion.criterion, 1.0, True, [])
                    for criterion in criteria
                ]
            
            mock_evaluator.evaluate_response.side_effect = mock_evaluate_response
            mock_eval_class.return_value = mock_evaluator
            
            result = flow.execute_sync(mock_flow_agent)
        
        assert isinstance(result, FlowResult)
        assert result.flow_id == "test_flow"
        assert result.passed == True
        assert result.steps_executed == 1
        assert result.total_steps == 1
        assert len(result.step_results) == 1
    
    def test_execute_multi_step_flow(self, mock_flow_agent):
        """Test multi-step flow execution"""
        flow = ConversationFlow("multi_step_flow")
        
        flow.step("Hello, I'm new", ["Should acknowledge new user"])
        flow.step("My name is John", ["Should note the name"], expect_context_retention=True)
        flow.context_check("What was my name?", ["Should remember John"])
        
        # Mock the evaluation loop
        with patch('testllm.flows.EvaluationLoop') as mock_eval_class:
            mock_evaluator = AsyncMock()
            
            def mock_evaluate_response(user_input, agent_response, criteria):
                return [
                    ConsensusResult(criterion.criterion, 1.0, True, [])
                    for criterion in criteria
                ]
            
            mock_evaluator.evaluate_response.side_effect = mock_evaluate_response
            mock_eval_class.return_value = mock_evaluator
            
            result = flow.execute_sync(mock_flow_agent)
        
        assert result.passed == True
        assert result.steps_executed == 3
        assert len(result.step_results) == 3
        assert len(flow.conversation_history) == 3
        
        # Check conversation history
        assert flow.conversation_history[0]["user_input"] == "Hello, I'm new"
        assert flow.conversation_history[1]["user_input"] == "My name is John"
        assert flow.conversation_history[2]["user_input"] == "What was my name?"
    
    def test_execute_with_error_handling(self, mock_flow_agent):
        """Test flow execution with error handling"""
        flow = ConversationFlow("error_flow")
        flow.step("Test input", ["Should handle gracefully"])
        
        # Mock evaluation to raise an exception
        with patch('testllm.flows.EvaluationLoop') as mock_eval_class:
            mock_evaluator = AsyncMock()
            mock_evaluator.evaluate_response.side_effect = Exception("Evaluation failed")
            mock_eval_class.return_value = mock_evaluator
            
            result = flow.execute_sync(mock_flow_agent)
        
        # Should handle error gracefully
        assert result.passed == False
        assert len(result.step_results) == 1
        assert len(result.step_results[0].errors) > 0
        assert "Step execution error" in result.step_results[0].errors[0]
    
    def test_enhanced_criteria_generation(self, mock_flow_agent):
        """Test that enhanced criteria are generated based on expectations"""
        flow = ConversationFlow("enhanced_flow")
        
        flow.step(
            "Book a flight",
            criteria=["Should search for flights"],
            expect_context_retention=True,
            expect_tool_usage=["flight_search"],
            expect_business_logic=["availability_check"]
        )
        
        # Mock evaluation and check enhanced criteria
        with patch('testllm.flows.EvaluationLoop') as mock_eval_class:
            mock_evaluator = AsyncMock()
            
            def check_criteria(user_input, agent_response, criteria):
                # Verify enhanced criteria were added
                criterion_texts = [c.criterion for c in criteria]
                assert "Should search for flights" in criterion_texts
                assert any("context" in c for c in criterion_texts)
                assert any("flight_search" in c for c in criterion_texts)
                assert any("availability_check" in c for c in criterion_texts)
                
                return [ConsensusResult(c.criterion, 1.0, True, []) for c in criteria]
            
            mock_evaluator.evaluate_response.side_effect = check_criteria
            mock_eval_class.return_value = mock_evaluator
            
            result = flow.execute_sync(mock_flow_agent)
            assert result.passed


class TestConversationFlowFactory:
    """Test the conversation_flow factory function"""
    
    def test_conversation_flow_factory(self):
        """Test conversation_flow factory function"""
        flow = conversation_flow("factory_test", "Factory created flow")
        
        assert isinstance(flow, ConversationFlow)
        assert flow.flow_id == "factory_test"
        assert flow.description == "Factory created flow"
    
    def test_conversation_flow_factory_with_config(self):
        """Test conversation_flow factory with custom configuration"""
        flow = conversation_flow(
            "custom_factory_test",
            "Custom factory flow",
            evaluator_models=["claude-sonnet-4"],
            consensus_threshold=0.9
        )
        
        assert flow.config.evaluator_models == ["claude-sonnet-4"]
        assert flow.config.consensus_threshold == 0.9


class TestFlowStepTypes:
    """Test different flow step types and functionality"""
    
    def test_flow_step_creation(self):
        """Test FlowStep creation"""
        step = FlowStep(
            step_id="test_step",
            step_type=FlowStepType.USER_INPUT,
            user_input="Test input",
            criteria=["Test criterion"]
        )
        
        assert step.step_id == "test_step"
        assert step.step_type == FlowStepType.USER_INPUT
        assert step.user_input == "Test input"
        assert step.criteria == ["Test criterion"]
    
    def test_flow_step_with_expectations(self):
        """Test FlowStep with behavioral expectations"""
        step = FlowStep(
            step_id="expectation_step",
            step_type=FlowStepType.USER_INPUT,
            user_input="Test",
            criteria=["Test"],
            expect_context_retention=True,
            expect_tool_usage_indicators=["tool1", "tool2"],
            expect_business_logic=["rule1"]
        )
        
        assert step.expect_context_retention == True
        assert step.expect_tool_usage_indicators == ["tool1", "tool2"]
        assert step.expect_business_logic == ["rule1"]


class TestFlowResult:
    """Test FlowResult functionality"""
    
    def test_flow_result_creation(self):
        """Test FlowResult creation"""
        result = FlowResult(
            flow_id="test_flow",
            description="Test flow result",
            steps_executed=2,
            total_steps=3,
            passed=True,
            overall_score=0.85,
            execution_time=1.5
        )
        
        assert result.flow_id == "test_flow"
        assert result.description == "Test flow result"
        assert result.steps_executed == 2
        assert result.total_steps == 3
        assert result.passed == True
        assert result.overall_score == 0.85
        assert result.execution_time == 1.5
    
    def test_flow_result_with_metrics(self):
        """Test FlowResult with flow-specific metrics"""
        result = FlowResult(
            flow_id="metrics_flow",
            description="Flow with metrics",
            steps_executed=1,
            total_steps=1,
            passed=True,
            overall_score=0.9,
            execution_time=1.0,
            context_retention_score=0.85,
            business_logic_score=0.95,
            tool_usage_score=0.8
        )
        
        assert result.context_retention_score == 0.85
        assert result.business_logic_score == 0.95
        assert result.tool_usage_score == 0.8


class TestFlowIntegration:
    """Integration tests for flow functionality"""
    
    def test_complete_flow_workflow(self, mock_flow_agent):
        """Test a complete workflow from creation to execution"""
        # Create flow
        flow = conversation_flow("integration_test", "Complete integration test")
        
        # Add varied steps
        flow.step("Hello", ["Be friendly"])
        flow.tool_usage_check("Book flight", ["flight_search"], ["Search for flights"])
        flow.context_check("What was I booking?", ["Remember flight request"])
        flow.business_logic_check("Complete booking", ["booking_rules"], ["Follow booking process"])
        
        # Mock evaluation
        with patch('testllm.flows.EvaluationLoop') as mock_eval_class:
            mock_evaluator = AsyncMock()
            mock_consensus = ConsensusResult("Test", 1.0, True, [])
            mock_evaluator.evaluate_response.return_value = [mock_consensus]
            mock_eval_class.return_value = mock_evaluator
            
            result = flow.execute_sync(mock_flow_agent)
        
        # Comprehensive assertions
        assert result.passed
        assert result.steps_executed == 4
        assert result.total_steps == 4
        assert len(result.step_results) == 4
        assert len(flow.conversation_history) == 4
        
        # Check step types were executed correctly
        step_ids = [sr.test_id for sr in result.step_results]
        assert any("integration_test_step_1" in sid for sid in step_ids)
        assert any("integration_test_step_2" in sid for sid in step_ids)
    
    def test_flow_score_calculations(self, mock_flow_agent):
        """Test flow score calculation methods"""
        flow = ConversationFlow("score_test")
        
        # Add steps with specific criteria types
        flow.step("Context test", ["Response should show context awareness"])
        flow.step("Tool test", ["Response should show tool usage patterns"])  
        flow.step("Business test", ["Response should follow business logic"])
        
        # Create mock step results
        step_results = [
            SemanticTestResult(
                test_id="context", description="", user_input="", agent_response="",
                criteria=["Response should show context awareness"], passed=True, overall_score=0.8
            ),
            SemanticTestResult(
                test_id="tool", description="", user_input="", agent_response="",
                criteria=["Response should show tool usage patterns"], passed=True, overall_score=0.9
            ),
            SemanticTestResult(
                test_id="business", description="", user_input="", agent_response="",
                criteria=["Response should follow business logic"], passed=True, overall_score=0.7
            )
        ]
        
        # Test score calculation methods
        context_score = flow._calculate_context_retention_score(step_results)
        tool_score = flow._calculate_tool_usage_score(step_results)
        business_score = flow._calculate_business_logic_score(step_results)
        
        assert context_score == 0.8  # Only context step
        assert tool_score == 0.9     # Only tool step
        assert business_score == 0.7 # Only business step


if __name__ == "__main__":
    # Run tests directly
    import sys
    import subprocess
    
    print("Running conversation flow tests...")
    print("=" * 50)
    
    # Run with pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v",
        "--tb=short"
    ], capture_output=False)
    
    sys.exit(result.returncode)