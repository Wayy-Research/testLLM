#!/usr/bin/env python3
"""Debug script for flow testing"""

import sys
sys.path.insert(0, '.')

from testllm.flows import ConversationFlow
from testllm.core import AgentUnderTest
from unittest.mock import patch, AsyncMock
from testllm.evaluation_loop import ConsensusResult

class DebugAgent(AgentUnderTest):
    def send_message(self, message: str) -> str:
        return f"Mock response to: {message}"
    
    def reset_conversation(self):
        pass

# Create a simple test
agent = DebugAgent()
flow = ConversationFlow('test')
flow.step('Hello', ['Be friendly'])

print("=== Basic test ===")

# Mock evaluation
with patch('testllm.flows.EvaluationLoop') as mock_eval_class:
    mock_evaluator = AsyncMock()
    
    def mock_evaluate_response(user_input, agent_response, criteria):
        print(f'Criteria received: {[c.criterion for c in criteria]}')
        results = [ConsensusResult(criterion.criterion, 1.0, True, []) for criterion in criteria]
        print(f'Returning {len(results)} results, all passed: {[r.passed for r in results]}')
        return results
    
    mock_evaluator.evaluate_response.side_effect = mock_evaluate_response
    mock_eval_class.return_value = mock_evaluator
    
    result = flow.execute_sync(agent)
    print(f'Flow result: passed={result.passed}, overall_score={result.overall_score}')
    print(f'Step results: {len(result.step_results)}')
    for sr in result.step_results:
        print(f'  Step {sr.test_id}: passed={sr.passed}, score={sr.overall_score}, errors={sr.errors}')

print("\n=== Test matching the failing integration test ===")

# Replicate the FULL failing test pattern
flow2 = ConversationFlow('e2e_onboarding', 'End-to-end customer onboarding')

# Step 1: Initial contact
flow2.step(
    "Hello, I'm a new customer interested in your services",
    criteria=[
        "Response should acknowledge new customer status",
        "Response should begin onboarding process",
        "Response should be professional and welcoming"
    ]
)

# Step 2: Information gathering
flow2.step(
    "My name is Sarah Johnson and I need a business account",
    criteria=[
        "Response should acknowledge the name Sarah Johnson",
        "Response should understand business account requirement",
        "Response should ask for relevant business information"
    ],
    expect_context_retention=True
)

# Step 3: Memory validation
flow2.context_check(
    "What type of account was I requesting?",
    context_criteria=[
        "Response should remember business account request",
        "Response should demonstrate conversation awareness"
    ]
)

# Step 4: Process completion
flow2.business_logic_check(
    "I'm ready to complete the setup process",
    business_rules=["account_creation", "verification"],
    criteria=[
        "Response should outline completion steps",
        "Response should mention verification requirements"
    ]
)

# Check what all steps look like
print(f"Total steps: {len(flow2.steps)}")
for i, step in enumerate(flow2.steps):
    print(f"Step {i+1}: {step.user_input}")
    print(f"  Business logic expectations: {step.expect_business_logic}")

print("Starting full integration test...")

# Mock evaluation
with patch('testllm.flows.EvaluationLoop') as mock_eval_class:
    mock_evaluator = AsyncMock()
    
    def mock_evaluate_response(user_input, agent_response, criteria):
        print(f'User input: {user_input}')
        print(f'Agent response: {agent_response}')
        print(f'Criteria received: {[c.criterion for c in criteria]}')
        results = [ConsensusResult(criterion.criterion, 1.0, True, []) for criterion in criteria]
        print(f'Returning {len(results)} results, all passed: {[r.passed for r in results]}')
        return results
    
    mock_evaluator.evaluate_response.side_effect = mock_evaluate_response
    mock_eval_class.return_value = mock_evaluator
    
    result = flow2.execute_sync(agent)
    print(f'Flow result: passed={result.passed}, overall_score={result.overall_score}')
    print(f'Business logic score: {result.business_logic_score}')
    print(f'Step results: {len(result.step_results)}')
    for sr in result.step_results:
        print(f'  Step {sr.test_id}: passed={sr.passed}, score={sr.overall_score}, errors={sr.errors}')
        print(f'    Criteria: {sr.criteria}')