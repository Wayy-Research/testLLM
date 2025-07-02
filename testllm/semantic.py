"""
Semantic Testing - Primary testing interface using LLM evaluation
"""

import asyncio
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field

from .core import AgentUnderTest, UserTurn
from .evaluation_loop import EvaluationLoop, EvaluationLoopConfig, SemanticCriterion


@dataclass
class SemanticTestCase:
    """A single test case with semantic evaluation criteria"""
    user_input: str
    criteria: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SemanticTestResult:
    """Result of a semantic test execution"""
    test_id: str
    description: str
    user_input: str
    agent_response: str
    criteria: List[str]
    passed: bool
    overall_score: float
    criterion_results: List[Dict[str, Any]] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)
    execution_time: float = 0.0


class SemanticTest:
    """
    Primary testing interface using LLM-based semantic evaluation.
    
    This replaces traditional assertion-based testing with natural language
    criteria evaluated by LLM judges.
    """
    
    def __init__(
        self, 
        test_id: str, 
        description: str = "",
        evaluator_models: Optional[List[str]] = None,
        consensus_threshold: float = 0.7,
        parallel_evaluation: bool = True
    ):
        """
        Initialize a semantic test.
        
        Args:
            test_id: Unique identifier for the test
            description: Human-readable test description
            evaluator_models: List of LLM models to use for evaluation
            consensus_threshold: Minimum consensus score to pass (0.0-1.0)
            parallel_evaluation: Whether to run evaluators in parallel
        """
        self.test_id = test_id
        self.description = description
        self.test_cases: List[SemanticTestCase] = []
        
        # Evaluation configuration - defaults to fast mode with Mistral
        self.config = EvaluationLoopConfig(
            evaluator_models=evaluator_models or ["mistral-large-latest"],
            consensus_threshold=consensus_threshold,
            parallel_execution=parallel_evaluation,
            iterations=1  # Single iteration for speed
        )
    
    def add_scenario(self, user_input: str, criteria: List[str], **metadata) -> 'SemanticTest':
        """
        Add a test scenario with semantic evaluation criteria.
        
        Args:
            user_input: The message to send to the agent
            criteria: List of natural language criteria for evaluation
            **metadata: Additional metadata for the test scenario
            
        Returns:
            Self for method chaining
            
        Example:
            test.add_scenario(
                user_input="What's the weather like?",
                criteria=[
                    "Response should acknowledge the weather question",
                    "Response should ask for location or provide helpful guidance",
                    "Response should be polite and helpful"
                ]
            )
        """
        test_case = SemanticTestCase(
            user_input=user_input,
            criteria=criteria,
            metadata=metadata
        )
        self.test_cases.append(test_case)
        return self
    
    def add_case(self, user_input: str, *criteria: str, **metadata) -> 'SemanticTest':
        """
        DEPRECATED: Use add_scenario() instead for clearer API.
        Add a test case with semantic evaluation criteria.
        
        Args:
            user_input: The message to send to the agent
            *criteria: Natural language criteria for evaluation
            **metadata: Additional metadata for the test case
            
        Returns:
            Self for method chaining
        """
        return self.add_scenario(user_input, list(criteria), **metadata)
    
    async def execute(self, agent: AgentUnderTest) -> List[SemanticTestResult]:
        """
        Execute all test cases against the agent using LLM evaluation.
        
        Args:
            agent: The agent instance to test
            
        Returns:
            List of test results for each test case
        """
        evaluator = EvaluationLoop(self.config)
        results = []
        
        agent.reset_conversation()
        
        for i, test_case in enumerate(self.test_cases):
            start_time = time.time()
            
            try:
                # Get agent response
                agent_response = agent.send_message(test_case.user_input)
                
                # Convert criteria to SemanticCriterion objects
                semantic_criteria = [
                    SemanticCriterion(criterion=criterion)
                    for criterion in test_case.criteria
                ]
                
                # Evaluate with LLMs
                consensus_results = await evaluator.evaluate_response(
                    test_case.user_input,
                    agent_response,
                    semantic_criteria
                )
                
                # Calculate overall results
                passed = all(result.passed for result in consensus_results)
                overall_score = (
                    sum(result.consensus_score for result in consensus_results) 
                    / len(consensus_results) if consensus_results else 0.0
                )
                
                # Print detailed evaluation results when running tests
                if self._is_in_test_environment():
                    self._print_test_evaluation(i, test_case, agent_response, consensus_results, passed, overall_score)
                
                # Format criterion results
                criterion_results = []
                for consensus_result in consensus_results:
                    criterion_results.append({
                        "criterion": consensus_result.criterion,
                        "passed": consensus_result.passed,
                        "consensus_score": consensus_result.consensus_score,
                        "evaluations": [
                            {
                                "evaluator": eval_result.evaluator_model,
                                "decision": eval_result.decision,
                                "confidence": eval_result.confidence,
                                "reasoning": eval_result.reasoning
                            }
                            for eval_result in consensus_result.individual_results
                        ]
                    })
                
                result = SemanticTestResult(
                    test_id=f"{self.test_id}_case_{i}",
                    description=self.description,
                    user_input=test_case.user_input,
                    agent_response=agent_response,
                    criteria=test_case.criteria,
                    passed=passed,
                    overall_score=overall_score,
                    criterion_results=criterion_results,
                    execution_time=time.time() - start_time
                )
                
            except Exception as e:
                result = SemanticTestResult(
                    test_id=f"{self.test_id}_case_{i}",
                    description=self.description,
                    user_input=test_case.user_input,
                    agent_response="",
                    criteria=test_case.criteria,
                    passed=False,
                    overall_score=0.0,
                    errors=[f"Execution error: {str(e)}"],
                    execution_time=time.time() - start_time
                )
            
            results.append(result)
        
        return results
    
    def execute_sync(self, agent: AgentUnderTest) -> List[SemanticTestResult]:
        """
        Synchronous wrapper for execute() method.
        
        Args:
            agent: The agent instance to test
            
        Returns:
            List of test results for each test case
        """
        return asyncio.run(self.execute(agent))
    
    def _print_test_evaluation(self, test_index: int, test_case, agent_response: str, consensus_results, passed: bool, overall_score: float):
        """Print detailed evaluation results in a readable format"""
        print(f"\n{'='*80}")
        print(f"🧪 TEST CASE EVALUATION: {self.test_id}_case_{test_index}")
        print(f"📄 Test: {self.description}" + (f" | {self.test_id}" if self.description != self.test_id else ""))
        print(f"{'='*80}")
        print(f"📝 User Input: '{test_case.user_input}'")
        print(f"🤖 Agent Response: '{agent_response}'")
        print(f"{'─'*80}")
        
        for i, result in enumerate(consensus_results):
            status_icon = "✅" if result.passed else "❌"
            status_text = "PASS" if result.passed else "FAIL"
            print(f"\n📋 Criterion {i+1}: {status_icon} {status_text} (Score: {result.consensus_score:.2f})")
            print(f"   └── '{result.criterion}'")
            
            if result.individual_results:
                for eval_result in result.individual_results:
                    decision_icon = "✅" if eval_result.decision == "YES" else "❌" if eval_result.decision == "NO" else "⚠️"
                    print(f"   └── {decision_icon} {eval_result.evaluator_model}: {eval_result.decision}")
                    print(f"       💭 {eval_result.reasoning}")
        
        print(f"\n{'─'*80}")
        test_status_icon = "✅" if passed else "❌"
        test_status_text = "PASS" if passed else "FAIL" 
        print(f"🎯 Test Result: {test_status_icon} {test_status_text} (Overall Score: {overall_score:.2f})")
        print(f"{'='*80}\n")
    
    def _is_in_test_environment(self) -> bool:
        """Check if we're running in a pytest environment"""
        import sys
        import os
        return 'pytest' in sys.modules or 'PYTEST_CURRENT_TEST' in os.environ


def semantic_test(
    test_id: str,
    description: str = "",
    evaluator_models: Optional[List[str]] = None,
    consensus_threshold: float = 0.7
) -> SemanticTest:
    """
    Factory function to create a semantic test.
    
    Args:
        test_id: Unique identifier for the test
        description: Human-readable test description  
        evaluator_models: List of LLM models to use for evaluation
        consensus_threshold: Minimum consensus score to pass (0.0-1.0)
        
    Returns:
        SemanticTest instance
        
    Example:
        test = semantic_test("greeting_test", "Test agent greeting behavior")
        test.add_case(
            "Hello!",
            "Response should be a friendly greeting",
            "Response should offer help or ask how to assist"
        )
    """
    return SemanticTest(
        test_id=test_id,
        description=description,
        evaluator_models=evaluator_models,
        consensus_threshold=consensus_threshold
    )


def pytest_semantic_test(
    test_id: str,
    description: str = "",
    evaluator_models: Optional[List[str]] = None,
    consensus_threshold: float = 0.7
):
    """
    Decorator to create a pytest-compatible semantic test.
    
    Args:
        test_id: Unique identifier for the test
        description: Human-readable test description
        evaluator_models: List of LLM models to use for evaluation
        consensus_threshold: Minimum consensus score to pass
        
    Example:
        @pytest_semantic_test("greeting_test", "Test greetings")
        def test_greeting(agent):
            return [
                ("Hello!", [
                    "Response should be friendly",
                    "Response should offer assistance"
                ])
            ]
    """
    def decorator(test_function):
        def wrapper(*args, **kwargs):
            # Find agent in arguments
            agent = None
            for arg in args:
                if isinstance(arg, AgentUnderTest):
                    agent = arg
                    break
            
            if agent is None:
                agent = kwargs.get('agent')
                if agent is None:
                    raise ValueError("No AgentUnderTest instance found in test arguments")
            
            # Get test cases from function
            test_cases = test_function(*args, **kwargs)
            
            # Create and execute semantic test
            test = SemanticTest(
                test_id=test_id,
                description=description,
                evaluator_models=evaluator_models,
                consensus_threshold=consensus_threshold
            )
            
            for user_input, criteria in test_cases:
                test.add_case(user_input, *criteria)
            
            results = test.execute_sync(agent)
            
            # Assert all test cases passed
            failed_cases = [r for r in results if not r.passed]
            if failed_cases:
                error_messages = []
                for failed_case in failed_cases:
                    error_messages.append(
                        f"Test case '{failed_case.user_input}' failed (score: {failed_case.overall_score:.2f})"
                    )
                    if failed_case.errors:
                        error_messages.extend(failed_case.errors)
                
                raise AssertionError(f"Semantic test failed: {'; '.join(error_messages)}")
            
            return results
        
        wrapper.__name__ = test_function.__name__
        wrapper.__doc__ = test_function.__doc__
        return wrapper
    
    return decorator