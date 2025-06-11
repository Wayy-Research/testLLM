"""
Pytest integration for semantic testing with testLLM
"""

import asyncio
import pytest
from typing import List, Optional
from .core import SemanticTest, SemanticTestResult, AgentUnderTest


def semantic_test(test_id: str, description: str = "", evaluator_models: Optional[List[str]] = None):
    """
    Decorator to create a pytest test using semantic evaluation
    
    Usage:
        @semantic_test("greeting_test", "Test greeting functionality")
        async def test_greeting(agent):
            test = SemanticTest("greeting", "Test greetings")
            test.add_test_case(
                "Hello!",
                "Response should be a friendly greeting",
                "Response should offer help"
            )
            return await test.execute(agent)
    """
    def decorator(test_function):
        def wrapper(*args, **kwargs):
            # Extract agent from fixture
            agent = None
            for arg in args:
                if isinstance(arg, AgentUnderTest):
                    agent = arg
                    break
            
            if agent is None:
                agent = kwargs.get('agent')
                if agent is None:
                    raise ValueError("No AgentUnderTest instance found in test arguments")
            
            # Run the async test function
            try:
                loop = asyncio.get_event_loop()
            except RuntimeError:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
            
            if asyncio.iscoroutinefunction(test_function):
                results = loop.run_until_complete(test_function(agent, *args[1:], **kwargs))
            else:
                results = test_function(agent, *args[1:], **kwargs)
            
            # Validate results
            if not isinstance(results, list):
                results = [results]
            
            # Check if any test failed
            failed_tests = [r for r in results if not r.passed]
            if failed_tests:
                failure_messages = []
                for failed_test in failed_tests:
                    failure_messages.append(f"Test {failed_test.test_id} failed (score: {failed_test.consensus_score:.2f})")
                    if failed_test.errors:
                        failure_messages.extend(failed_test.errors)
                
                raise AssertionError(f"Semantic tests failed: {'; '.join(failure_messages)}")
            
            return results
        
        # Preserve original function metadata
        wrapper.__name__ = test_function.__name__
        wrapper.__doc__ = test_function.__doc__
        return wrapper
    
    return decorator


class SemanticTestRunner:
    """Helper class for running semantic tests in pytest"""
    
    def __init__(self, evaluator_models: Optional[List[str]] = None):
        self.evaluator_models = evaluator_models or ["claude-sonnet-4"]
    
    def create_test(self, test_id: str, description: str = "") -> SemanticTest:
        """Create a new semantic test instance"""
        return SemanticTest(
            test_id=test_id,
            description=description,
            evaluator_models=self.evaluator_models
        )
    
    async def run_test(self, agent: AgentUnderTest, test: SemanticTest) -> List[SemanticTestResult]:
        """Run a semantic test and return results"""
        return await test.execute(agent)
    
    def assert_all_passed(self, results: List[SemanticTestResult]) -> None:
        """Assert that all test results passed"""
        failed_tests = [r for r in results if not r.passed]
        if failed_tests:
            failure_messages = []
            for failed_test in failed_tests:
                failure_messages.append(f"Test {failed_test.test_id} failed (score: {failed_test.consensus_score:.2f})")
                if failed_test.errors:
                    failure_messages.extend(failed_test.errors)
            
            raise AssertionError(f"Semantic tests failed: {'; '.join(failure_messages)}")


# Pytest fixtures for semantic testing
@pytest.fixture
def semantic_runner():
    """Fixture providing a SemanticTestRunner instance"""
    return SemanticTestRunner()


@pytest.fixture
def semantic_runner_multi_model():
    """Fixture providing a SemanticTestRunner with multiple evaluator models"""
    return SemanticTestRunner(evaluator_models=["gpt-4o-mini", "claude-3-haiku-20240307"])