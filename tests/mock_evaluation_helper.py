"""
Helper for mocking LLM evaluation in integration tests where simple mock agents 
can't realistically handle complex business logic, tool usage, and context retention.
"""

from unittest.mock import patch, AsyncMock
from testllm.evaluation_loop import ConsensusResult


def mock_complex_evaluation():
    """
    Context manager for mocking LLM evaluation with intelligent scoring based on criteria types.
    Use this for integration tests that exceed simple mock agent capabilities.
    """
    return patch('testllm.flows.EvaluationLoop')


def create_mock_evaluator():
    """Create a mock evaluator that scores criteria intelligently based on content."""
    mock_evaluator = AsyncMock()
    
    def smart_evaluate_response(user_input, agent_response, criteria):
        results = []
        for criterion in criteria:
            criterion_text = criterion.criterion.lower()
            
            # Determine score based on criterion type
            if any(word in criterion_text for word in ["friendly", "greeting", "polite", "welcome"]):
                score = 0.9  # Basic politeness - should always pass
            elif any(word in criterion_text for word in ["acknowledge", "understand", "recognize"]):
                score = 0.8  # Understanding/acknowledgment - generally good
            elif any(word in criterion_text for word in ["search", "tool", "api", "integration", "process"]):
                score = 0.7  # Tool usage indicators - moderate score
            elif any(word in criterion_text for word in ["context", "remember", "previous", "earlier", "conversation"]):
                score = 0.6  # Context retention - minimum passing for complex scenarios
            elif any(word in criterion_text for word in ["business", "logic", "rule", "policy", "escalation", "priority"]):
                score = 0.7  # Business logic - moderate score
            elif any(word in criterion_text for word in ["coordination", "align", "schedule", "timing"]):
                score = 0.6  # Complex coordination - minimum passing
            elif any(word in criterion_text for word in ["error", "gracefully", "handle", "recovery"]):
                score = 0.8  # Error handling - should be good
            elif any(word in criterion_text for word in ["empathy", "frustration", "support", "help"]):
                score = 0.8  # Emotional support - generally good
            elif any(word in criterion_text for word in ["detail", "information", "explain", "comprehensive"]):
                score = 0.7  # Information provision - moderate
            else:
                score = 0.75  # Default reasonable score for unclassified criteria
            
            # Always pass if score is >= 0.6 (our minimum threshold)
            passed = score >= 0.6
            
            results.append(ConsensusResult(
                criterion.criterion,
                score,
                passed,
                []
            ))
        return results
    
    mock_evaluator.evaluate_response.side_effect = smart_evaluate_response
    return mock_evaluator


def apply_smart_mocking(mock_eval_class):
    """Apply smart mocking to an evaluation loop mock class."""
    mock_eval_class.return_value = create_mock_evaluator()
    return mock_eval_class