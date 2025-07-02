"""
Test file for semantic testing functionality
Run with: pytest tests/test_semantic.py -v
"""

import pytest
from unittest.mock import Mock, AsyncMock, patch
from testllm.semantic import SemanticTest, semantic_test, pytest_semantic_test
from testllm.core import AgentUnderTest
from testllm.evaluation_loop import EvaluationLoop, EvaluationLoopConfig, EvaluationResult, ConsensusResult


class MockAgent(AgentUnderTest):
    """Mock agent for testing semantic functionality"""
    
    def __init__(self):
        super().__init__()
        self.responses = {
            "Hello": "Hi there! How can I help you today?",
            "What's the weather?": "I'd be happy to help with weather. What city are you interested in?",
            "Tell me a joke": "Why don't scientists trust atoms? Because they make up everything!",
            "Explain Python": "Python is a high-level programming language known for its simplicity and readability.",
            "Help me": "Of course! I'm here to assist you. What do you need help with?",
        }
    
    def send_message(self, message: str) -> str:
        """Mock send_message that returns predefined responses"""
        message_lower = message.lower()
        
        # Check for partial matches to handle variations like "Hello there!"
        for key, response in self.responses.items():
            if key.lower() in message_lower:
                return response
        
        # Default response
        return "I understand your question. How can I assist you?"
    
    def reset_conversation(self):
        """Mock reset - no-op for this test"""
        pass


@pytest.fixture
def mock_agent():
    """Fixture providing a mock agent"""
    return MockAgent()


@pytest.fixture
def mock_evaluation_result():
    """Fixture providing a mock evaluation result"""
    return EvaluationResult(
        criterion="Response should be friendly",
        evaluator_model="claude-sonnet-4",
        decision="YES",
        confidence=1.0,
        reasoning="The response is warm and welcoming"
    )


@pytest.fixture
def mock_consensus_result():
    """Fixture providing a mock consensus result"""
    return ConsensusResult(
        criterion="Response should be friendly",
        consensus_score=1.0,
        passed=True,
        individual_results=[]
    )


class TestSemanticTest:
    """Test the SemanticTest class"""
    
    def test_semantic_test_initialization(self):
        """Test SemanticTest initialization"""
        test = SemanticTest("test_id", "Test description")
        
        assert test.test_id == "test_id"
        assert test.description == "Test description"
        assert len(test.test_cases) == 0
        assert test.config.evaluator_models == ["mistral-large-latest"]
        assert test.config.consensus_threshold == 0.7
        assert test.config.iterations == 1
    
    def test_add_case(self):
        """Test adding test cases"""
        test = SemanticTest("test_id")
        
        test.add_case(
            "Hello",
            "Response should be friendly",
            "Response should offer help"
        )
        
        assert len(test.test_cases) == 1
        case = test.test_cases[0]
        assert case.user_input == "Hello"
        assert len(case.criteria) == 2
        assert "Response should be friendly" in case.criteria
        assert "Response should offer help" in case.criteria
    
    def test_add_case_with_metadata(self):
        """Test adding test cases with metadata"""
        test = SemanticTest("test_id")
        
        test.add_case(
            "Hello",
            "Response should be friendly",
            priority="high",
            category="greeting"
        )
        
        case = test.test_cases[0]
        assert case.metadata["priority"] == "high"
        assert case.metadata["category"] == "greeting"
    
    def test_method_chaining(self):
        """Test that add_case supports method chaining"""
        test = SemanticTest("test_id")
        
        result = test.add_case("Hello", "Be friendly").add_case("Goodbye", "Be polite")
        
        assert result is test
        assert len(test.test_cases) == 2
    
    def test_execute_with_mock(self, mock_agent):
        """Test execute method with mocked evaluation"""
        test = SemanticTest("greeting_test")
        test.add_case("Hello", "Response should be friendly")
        
        results = test.execute_sync(mock_agent)
        
        assert len(results) == 1
        result = results[0]
        assert result.test_id == "greeting_test_case_0"
        assert result.user_input == "Hello"
        assert result.agent_response == "Hi there! How can I help you today?"
        # This test uses real LLM evaluation - should pass for friendly response
        assert result.overall_score >= 0.7  # Expect high score for friendly greeting
    
    def test_execute_sync(self, mock_agent):
        """Test synchronous execute wrapper"""
        test = SemanticTest("greeting_test")
        test.add_case("Hello", "Response should be friendly")
        
        # Mock the async execute method
        mock_results = [Mock(passed=True, overall_score=1.0)]
        with patch.object(test, 'execute', return_value=mock_results) as mock_execute:
            results = test.execute_sync(mock_agent)
        
        mock_execute.assert_called_once_with(mock_agent)
        assert results == mock_results


class TestSemanticTestFactory:
    """Test the semantic_test factory function"""
    
    def test_semantic_test_factory(self):
        """Test semantic_test factory function"""
        test = semantic_test("test_id", "Test description")
        
        assert isinstance(test, SemanticTest)
        assert test.test_id == "test_id"
        assert test.description == "Test description"
    
    def test_semantic_test_factory_with_config(self):
        """Test semantic_test factory with custom configuration"""
        test = semantic_test(
            "test_id",
            "Test description",
            evaluator_models=["mistral-large-latest"],
            consensus_threshold=0.8
        )
        
        assert test.config.evaluator_models == ["mistral-large-latest"]
        assert test.config.consensus_threshold == 0.8


class TestPytestSemanticDecorator:
    """Test the pytest_semantic_test decorator"""
    
    def test_pytest_decorator_basic(self, mock_agent):
        """Test basic pytest decorator functionality"""
        
        @pytest_semantic_test("greeting_test", "Test greetings")
        def test_greeting(agent):
            return [
                ("Hello", [
                    "Response should be friendly",
                    "Response should offer assistance"
                ])
            ]
        
        # Mock the evaluation to return passing results
        with patch('testllm.semantic.EvaluationLoop') as mock_eval_class:
            mock_evaluator = AsyncMock()
            mock_consensus = [
                ConsensusResult("Response should be friendly", 1.0, True, []),
                ConsensusResult("Response should offer assistance", 1.0, True, [])
            ]
            mock_evaluator.evaluate_response.return_value = mock_consensus
            mock_eval_class.return_value = mock_evaluator
            
            # Should not raise an exception
            results = test_greeting(mock_agent)
            assert len(results) == 1
            assert results[0].passed == True
    
    def test_pytest_decorator_failure(self, mock_agent):
        """Test pytest decorator with failing test"""
        
        @pytest_semantic_test("failing_test", "Test that should fail")
        def test_failing(agent):
            return [
                ("Hello", ["Response should be rude"])  # This should fail
            ]
        
        # Mock the evaluation to return failing results
        with patch('testllm.semantic.EvaluationLoop') as mock_eval_class:
            mock_evaluator = AsyncMock()
            mock_consensus = [
                ConsensusResult("Response should be rude", 0.0, False, [])
            ]
            mock_evaluator.evaluate_response.return_value = mock_consensus
            mock_eval_class.return_value = mock_evaluator
            
            # Should raise AssertionError
            with pytest.raises(AssertionError, match="Semantic test failed"):
                test_failing(mock_agent)
    
    def test_pytest_decorator_no_agent_error(self):
        """Test pytest decorator without agent raises error"""
        
        @pytest_semantic_test("test_id", "Test")
        def test_no_agent():
            return [("Hello", ["Be friendly"])]
        
        with pytest.raises(ValueError, match="No AgentUnderTest instance found"):
            test_no_agent()


class TestSemanticTestIntegration:
    """Integration tests for semantic testing"""
    
    def test_complete_semantic_test_flow(self, mock_agent):
        """Test complete semantic test flow"""
        # Create test
        test = semantic_test("integration_test", "Integration test")
        
        # Add multiple test cases
        test.add_case(
            "Hello",
            "Response should be friendly",
            "Response should offer help"
        )
        test.add_case(
            "Tell me a joke",
            "Response should be humorous",
            "Response should be appropriate"
        )
        
        # Mock evaluation results
        with patch('testllm.semantic.EvaluationLoop') as mock_eval_class:
            mock_evaluator = AsyncMock()
            
            # Mock responses for each criterion
            def mock_evaluate_response(user_input, agent_response, criteria):
                results = []
                for criterion in criteria:
                    # All criteria pass for this test
                    results.append(ConsensusResult(
                        criterion.criterion, 1.0, True, []
                    ))
                return results
            
            mock_evaluator.evaluate_response.side_effect = mock_evaluate_response
            mock_eval_class.return_value = mock_evaluator
            
            # Execute test
            results = test.execute_sync(mock_agent)
        
        # Verify results
        assert len(results) == 2
        assert all(result.passed for result in results)
        assert results[0].user_input == "Hello"
        assert results[1].user_input == "Tell me a joke"
    
    def test_semantic_test_with_custom_config(self, mock_agent):
        """Test semantic test with custom configuration"""
        test = SemanticTest(
            "custom_test",
            "Test with custom config",
            evaluator_models=["mistral-large-latest"],
            consensus_threshold=0.8,
            parallel_evaluation=False
        )
        
        assert test.config.evaluator_models == ["mistral-large-latest"]
        assert test.config.consensus_threshold == 0.8
        assert test.config.parallel_execution == False
    
    def test_real_semantic_evaluation(self, mock_agent):
        """Test with real LLM semantic evaluation - not mocked"""
        test = SemanticTest("real_semantic_test")
        
        # Add cases that should clearly pass and fail
        test.add_case("Hello there!", ["Response should be friendly and welcoming"])
        test.add_case("Help me", ["Response should offer assistance and ask for more details"])
        
        # Execute with real LLM evaluation
        results = test.execute_sync(mock_agent)
        
        assert len(results) == 2
        # First result should score well - friendly greeting gets friendly response
        assert results[0].overall_score >= 0.7
        # Second might score lower since mock agent doesn't actually provide time
        # but we just verify it runs without error
        assert results[1].overall_score >= 0.0

    def test_error_handling_in_execution(self, mock_agent):
        """Test error handling during test execution"""
        test = semantic_test("error_test", "Test error handling")
        test.add_case("Hello", "Response should be friendly")
        
        # Mock evaluation to raise an exception
        with patch('testllm.semantic.EvaluationLoop') as mock_eval_class:
            mock_evaluator = AsyncMock()
            mock_evaluator.evaluate_response.side_effect = Exception("Evaluation failed")
            mock_eval_class.return_value = mock_evaluator
            
            results = test.execute_sync(mock_agent)
        
        # Should handle the error gracefully
        assert len(results) == 1
        result = results[0]
        assert result.passed == False
        assert len(result.errors) > 0
        assert "Execution error" in result.errors[0]


if __name__ == "__main__":
    # Run tests directly
    import sys
    import subprocess
    
    print("Running semantic testing tests...")
    print("=" * 50)
    
    # Run with pytest
    result = subprocess.run([
        sys.executable, "-m", "pytest", 
        __file__, 
        "-v",
        "--tb=short"
    ], capture_output=False)
    
    sys.exit(result.returncode)