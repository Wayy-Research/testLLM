"""
Conversation Flow Testing - Multi-step behavioral testing for production agents
"""

import time
from typing import Dict, List, Any, Optional, Union, Callable
from dataclasses import dataclass, field
from enum import Enum

from .core import AgentUnderTest
from .semantic import SemanticTest, SemanticTestResult
from .evaluation_loop import EvaluationLoop, EvaluationLoopConfig, SemanticCriterion


class FlowStepType(Enum):
    """Types of flow steps"""
    USER_INPUT = "user_input"
    SYSTEM_CHECK = "system_check" 
    CONDITIONAL = "conditional"
    PARALLEL = "parallel"


@dataclass
class FlowStep:
    """A single step in a conversation flow"""
    step_id: str
    step_type: FlowStepType
    user_input: Optional[str] = None
    criteria: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Behavioral expectations
    expect_context_retention: bool = False
    expect_tool_usage_indicators: List[str] = field(default_factory=list)
    expect_business_logic: List[str] = field(default_factory=list)
    
    # Flow control
    conditional_check: Optional[Callable] = None
    parallel_steps: List['FlowStep'] = field(default_factory=list)


@dataclass
class FlowResult:
    """Result of executing a conversation flow"""
    flow_id: str
    description: str
    steps_executed: int
    total_steps: int
    passed: bool
    overall_score: float
    execution_time: float
    step_results: List[SemanticTestResult] = field(default_factory=list)
    flow_errors: List[str] = field(default_factory=list)
    
    # Flow-specific metrics
    context_retention_score: float = 0.0
    business_logic_score: float = 0.0
    tool_usage_score: float = 0.0


class ConversationFlow:
    """
    Multi-step conversation flow testing for production agentic systems.
    
    This class enables testing of complex workflows, context retention,
    tool usage patterns, and business logic validation through behavioral observation.
    """
    
    def __init__(
        self,
        flow_id: str,
        description: str = "",
        evaluator_models: Optional[List[str]] = None,
        consensus_threshold: float = 0.7
    ):
        """
        Initialize a conversation flow test.
        
        Args:
            flow_id: Unique identifier for the flow
            description: Human-readable flow description
            evaluator_models: LLM models for evaluation (default: claude-sonnet-4)
            consensus_threshold: Minimum consensus score to pass
        """
        self.flow_id = flow_id
        self.description = description
        self.steps: List[FlowStep] = []
        self.conversation_history: List[Dict[str, str]] = []
        
        # Evaluation configuration
        self.config = EvaluationLoopConfig(
            evaluator_models=evaluator_models or ["claude-sonnet-4"],
            consensus_threshold=consensus_threshold,
            parallel_execution=True,
            iterations=3
        )
    
    def step(
        self, 
        user_input: str, 
        criteria: List[str],
        step_id: Optional[str] = None,
        expect_context_retention: bool = False,
        expect_tool_usage: Optional[List[str]] = None,
        expect_business_logic: Optional[List[str]] = None,
        **metadata
    ) -> 'ConversationFlow':
        """
        Add a conversation step to the flow.
        
        Args:
            user_input: What the user says in this step
            criteria: Semantic criteria for evaluating the agent's response
            step_id: Optional step identifier
            expect_context_retention: Whether this step should show context awareness
            expect_tool_usage: Expected tool usage indicators in response
            expect_business_logic: Expected business logic patterns
            **metadata: Additional step metadata
            
        Returns:
            Self for method chaining
            
        Example:
            flow.step(
                "Hello, I'm a new customer",
                criteria=[
                    "Response should acknowledge new customer status",
                    "Response should begin onboarding process"
                ],
                expect_context_retention=False,
                expect_business_logic=["customer_onboarding"]
            )
        """
        step_id = step_id or f"step_{len(self.steps) + 1}"
        
        step = FlowStep(
            step_id=step_id,
            step_type=FlowStepType.USER_INPUT,
            user_input=user_input,
            criteria=criteria,
            expect_context_retention=expect_context_retention,
            expect_tool_usage_indicators=expect_tool_usage or [],
            expect_business_logic=expect_business_logic or [],
            metadata=metadata
        )
        
        self.steps.append(step)
        return self
    
    def conditional_step(
        self,
        condition_check: Callable[[str], bool],
        true_step: FlowStep,
        false_step: Optional[FlowStep] = None
    ) -> 'ConversationFlow':
        """
        Add a conditional step based on previous agent response.
        
        Args:
            condition_check: Function that takes agent response and returns bool
            true_step: Step to execute if condition is true
            false_step: Step to execute if condition is false
            
        Returns:
            Self for method chaining
        """
        conditional_step = FlowStep(
            step_id=f"conditional_{len(self.steps) + 1}",
            step_type=FlowStepType.CONDITIONAL,
            conditional_check=condition_check,
            metadata={"true_step": true_step, "false_step": false_step}
        )
        
        self.steps.append(conditional_step)
        return self
    
    def parallel_steps(self, *steps: FlowStep) -> 'ConversationFlow':
        """
        Add multiple steps that can be executed in parallel.
        
        Args:
            *steps: Multiple FlowStep objects to execute concurrently
            
        Returns:
            Self for method chaining
        """
        parallel_step = FlowStep(
            step_id=f"parallel_{len(self.steps) + 1}",
            step_type=FlowStepType.PARALLEL,
            parallel_steps=list(steps)
        )
        
        self.steps.append(parallel_step)
        return self
    
    def context_check(
        self,
        user_input: str,
        context_criteria: List[str],
        step_id: Optional[str] = None
    ) -> 'ConversationFlow':
        """
        Add a step specifically to test context retention.
        
        Args:
            user_input: Input that requires context from previous steps
            context_criteria: Criteria specifically about context awareness
            step_id: Optional step identifier
            
        Returns:
            Self for method chaining
            
        Example:
            flow.context_check(
                "What was my name again?",
                criteria=[
                    "Response should remember the name from earlier",
                    "Response should show conversation awareness"
                ]
            )
        """
        return self.step(
            user_input=user_input,
            criteria=context_criteria,
            step_id=step_id,
            expect_context_retention=True
        )
    
    def tool_usage_check(
        self,
        user_input: str,
        expected_tools: List[str],
        criteria: List[str],
        step_id: Optional[str] = None
    ) -> 'ConversationFlow':
        """
        Add a step to test tool usage patterns.
        
        Args:
            user_input: Input that should trigger tool usage
            expected_tools: List of expected tool usage indicators
            criteria: Criteria for evaluating tool usage in response
            step_id: Optional step identifier
            
        Returns:
            Self for method chaining
            
        Example:
            flow.tool_usage_check(
                "Book me a flight to NYC",
                expected_tools=["flight_search", "availability_check"],
                criteria=[
                    "Response should indicate searching for flights",
                    "Response should show tool usage without exposing internals"
                ]
            )
        """
        return self.step(
            user_input=user_input,
            criteria=criteria,
            step_id=step_id,
            expect_tool_usage=expected_tools
        )
    
    def business_logic_check(
        self,
        user_input: str,
        business_rules: List[str],
        criteria: List[str],
        step_id: Optional[str] = None
    ) -> 'ConversationFlow':
        """
        Add a step to test business logic compliance.
        
        Args:
            user_input: Input that should trigger business logic
            business_rules: List of business rules that should be applied
            criteria: Criteria for evaluating business logic in response
            step_id: Optional step identifier
            
        Returns:
            Self for method chaining
        """
        return self.step(
            user_input=user_input,
            criteria=criteria,
            step_id=step_id,
            expect_business_logic=business_rules
        )
    
    async def execute(self, agent: AgentUnderTest) -> FlowResult:
        """
        Execute the conversation flow against the agent.
        
        Args:
            agent: The agent instance to test
            
        Returns:
            FlowResult with detailed flow execution results
        """
        start_time = time.time()
        evaluator = EvaluationLoop(self.config)
        
        # Reset agent and conversation history
        agent.reset_conversation()
        self.conversation_history = []
        
        step_results = []
        flow_errors = []
        steps_executed = 0
        
        try:
            for step in self.steps:
                step_result = await self._execute_step(agent, step, evaluator)
                
                if step_result:
                    step_results.append(step_result)
                    steps_executed += 1
                    
                    # Add to conversation history
                    self.conversation_history.append({
                        "step_id": step.step_id,
                        "user_input": step.user_input or "",
                        "agent_response": step_result.agent_response,
                        "passed": step_result.passed
                    })
        
        except Exception as e:
            flow_errors.append(f"Flow execution error: {str(e)}")
        
        # Calculate overall metrics
        overall_score = (
            sum(r.overall_score for r in step_results) / len(step_results)
            if step_results else 0.0
        )
        
        passed = all(r.passed for r in step_results) and not flow_errors
        
        # Calculate flow-specific metrics
        context_score = self._calculate_context_retention_score(step_results)
        business_score = self._calculate_business_logic_score(step_results)
        tool_score = self._calculate_tool_usage_score(step_results)
        
        return FlowResult(
            flow_id=self.flow_id,
            description=self.description,
            steps_executed=steps_executed,
            total_steps=len(self.steps),
            passed=passed,
            overall_score=overall_score,
            execution_time=time.time() - start_time,
            step_results=step_results,
            flow_errors=flow_errors,
            context_retention_score=context_score,
            business_logic_score=business_score,
            tool_usage_score=tool_score
        )
    
    def execute_sync(self, agent: AgentUnderTest) -> FlowResult:
        """
        Synchronous wrapper for execute() method.
        
        Args:
            agent: The agent instance to test
            
        Returns:
            FlowResult with detailed flow execution results
        """
        import asyncio
        return asyncio.run(self.execute(agent))
    
    async def _execute_step(
        self, 
        agent: AgentUnderTest, 
        step: FlowStep, 
        evaluator: EvaluationLoop
    ) -> Optional[SemanticTestResult]:
        """Execute a single flow step"""
        
        if step.step_type == FlowStepType.USER_INPUT:
            return await self._execute_user_input_step(agent, step, evaluator)
        elif step.step_type == FlowStepType.CONDITIONAL:
            return await self._execute_conditional_step(agent, step, evaluator)
        elif step.step_type == FlowStepType.PARALLEL:
            return await self._execute_parallel_steps(agent, step, evaluator)
        else:
            return None
    
    async def _execute_user_input_step(
        self,
        agent: AgentUnderTest,
        step: FlowStep,
        evaluator: EvaluationLoop
    ) -> SemanticTestResult:
        """Execute a user input step"""
        start_time = time.time()
        
        try:
            # Get agent response
            agent_response = agent.send_message(step.user_input)
            
            # Build enhanced criteria including flow-specific checks
            enhanced_criteria = step.criteria.copy()
            
            # Add context retention criteria if expected
            if step.expect_context_retention and self.conversation_history:
                enhanced_criteria.append(
                    "Response should demonstrate awareness of previous conversation context"
                )
            
            # Add tool usage criteria if expected
            if step.expect_tool_usage_indicators:
                for tool in step.expect_tool_usage_indicators:
                    enhanced_criteria.append(
                        f"Response should indicate usage of {tool} functionality"
                    )
            
            # Add business logic criteria if expected
            if step.expect_business_logic:
                for rule in step.expect_business_logic:
                    enhanced_criteria.append(
                        f"Response should demonstrate {rule} business logic"
                    )
            
            # Convert to SemanticCriterion objects
            semantic_criteria = [
                SemanticCriterion(criterion=criterion)
                for criterion in enhanced_criteria
            ]
            
            # Evaluate with LLMs
            consensus_results = await evaluator.evaluate_response(
                step.user_input,
                agent_response,
                semantic_criteria
            )
            
            # Calculate results
            passed = all(result.passed for result in consensus_results)
            overall_score = (
                sum(result.consensus_score for result in consensus_results) 
                / len(consensus_results) if consensus_results else 0.0
            )
            
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
            
            return SemanticTestResult(
                test_id=f"{self.flow_id}_{step.step_id}",
                description=f"Flow step: {step.step_id}",
                user_input=step.user_input,
                agent_response=agent_response,
                criteria=enhanced_criteria,
                passed=passed,
                overall_score=overall_score,
                criterion_results=criterion_results,
                execution_time=time.time() - start_time
            )
            
        except Exception as e:
            return SemanticTestResult(
                test_id=f"{self.flow_id}_{step.step_id}",
                description=f"Flow step: {step.step_id}",
                user_input=step.user_input or "",
                agent_response="",
                criteria=step.criteria,
                passed=False,
                overall_score=0.0,
                errors=[f"Step execution error: {str(e)}"],
                execution_time=time.time() - start_time
            )
    
    async def _execute_conditional_step(
        self,
        agent: AgentUnderTest,
        step: FlowStep,
        evaluator: EvaluationLoop
    ) -> Optional[SemanticTestResult]:
        """Execute a conditional step based on previous response"""
        if not self.conversation_history:
            return None
        
        last_response = self.conversation_history[-1]["agent_response"]
        condition_result = step.conditional_check(last_response)
        
        next_step = (
            step.metadata["true_step"] if condition_result 
            else step.metadata["false_step"]
        )
        
        if next_step:
            return await self._execute_step(agent, next_step, evaluator)
        return None
    
    async def _execute_parallel_steps(
        self,
        agent: AgentUnderTest,
        step: FlowStep,
        evaluator: EvaluationLoop
    ) -> Optional[SemanticTestResult]:
        """Execute parallel steps (for now, execute sequentially)"""
        # For simplicity, execute sequentially
        # In production, this could be enhanced for true parallel execution
        results = []
        for parallel_step in step.parallel_steps:
            result = await self._execute_step(agent, parallel_step, evaluator)
            if result:
                results.append(result)
        
        # Return combined result
        if results:
            return results[0]  # For now, return first result
        return None
    
    def _calculate_context_retention_score(self, step_results: List[SemanticTestResult]) -> float:
        """Calculate context retention score across the flow"""
        context_steps = [
            r for r in step_results 
            if any("context" in criterion.lower() for criterion in r.criteria)
        ]
        
        if not context_steps:
            return 1.0
            
        return sum(r.overall_score for r in context_steps) / len(context_steps)
    
    def _calculate_business_logic_score(self, step_results: List[SemanticTestResult]) -> float:
        """Calculate business logic compliance score"""
        business_steps = [
            r for r in step_results 
            if any("business" in criterion.lower() for criterion in r.criteria)
        ]
        
        if not business_steps:
            return 1.0
            
        return sum(r.overall_score for r in business_steps) / len(business_steps)
    
    def _calculate_tool_usage_score(self, step_results: List[SemanticTestResult]) -> float:
        """Calculate tool usage pattern score"""
        tool_steps = [
            r for r in step_results 
            if any("tool" in criterion.lower() or "usage" in criterion.lower() for criterion in r.criteria)
        ]
        
        if not tool_steps:
            return 1.0
            
        return sum(r.overall_score for r in tool_steps) / len(tool_steps)


def conversation_flow(
    flow_id: str,
    description: str = "",
    evaluator_models: Optional[List[str]] = None,
    consensus_threshold: float = 0.7
) -> ConversationFlow:
    """
    Factory function to create a conversation flow.
    
    Args:
        flow_id: Unique identifier for the flow
        description: Human-readable flow description
        evaluator_models: LLM models for evaluation (default: claude-sonnet-4)
        consensus_threshold: Minimum consensus score to pass
        
    Returns:
        ConversationFlow instance
        
    Example:
        flow = conversation_flow("onboarding", "Test user onboarding process")
        flow.step(
            "Hello, I'm new here",
            criteria=["Response should begin onboarding process"]
        )
    """
    return ConversationFlow(
        flow_id=flow_id,
        description=description,
        evaluator_models=evaluator_models,
        consensus_threshold=consensus_threshold
    )