"""
Evaluation Loop - Multi-LLM semantic evaluation system
"""

import asyncio
import json
import os
import time
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from enum import Enum
import requests

# Load environment variables from .env file if available
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from .core import AgentUnderTest


class EvaluatorType(Enum):
    """Types of evaluator models"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    CUSTOM = "custom"


@dataclass
class SemanticCriterion:
    """A semantic criterion for evaluation"""
    criterion: str
    weight: float = 1.0
    description: Optional[str] = None


@dataclass
class EvaluationResult:
    """Result from a single evaluator for a single criterion"""
    criterion: str
    evaluator_model: str
    decision: str  # "YES", "NO", "MAYBE"
    confidence: float  # 0.0 to 1.0
    reasoning: str
    execution_time: float = 0.0


@dataclass
class ConsensusResult:
    """Consensus result across multiple evaluators"""
    criterion: str
    consensus_score: float  # 0.0 to 1.0
    passed: bool
    individual_results: List[EvaluationResult] = field(default_factory=list)
    weighted_score: float = 0.0


@dataclass
class EvaluationLoopConfig:
    """Configuration for the evaluation loop"""
    iterations: int = 3
    evaluator_models: List[str] = field(default_factory=lambda: ["claude-sonnet-4"])
    consensus_threshold: float = 0.67
    timeout: int = 30
    parallel_execution: bool = True
    retry_count: int = 2


class EvaluatorClient:
    """Base class for evaluator model clients"""
    
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.evaluator_type = self._detect_evaluator_type(model_name)
    
    def _detect_evaluator_type(self, model_name: str) -> EvaluatorType:
        """Detect evaluator type from model name"""
        if model_name.startswith(("gpt-", "o1-")):
            return EvaluatorType.OPENAI
        elif model_name.startswith(("claude-", "sonnet", "haiku", "opus")):
            return EvaluatorType.ANTHROPIC
        elif model_name.startswith(("llama", "mistral", "local-")):
            return EvaluatorType.LOCAL
        else:
            return EvaluatorType.CUSTOM
    
    async def evaluate(self, user_input: str, agent_response: str, 
                      criterion: SemanticCriterion) -> EvaluationResult:
        """Evaluate agent response against criterion"""
        start_time = time.time()
        
        try:
            prompt = self._build_evaluation_prompt(user_input, agent_response, criterion)
            response = await self._call_model(prompt)
            parsed_result = self._parse_evaluation_response(response)
            
            return EvaluationResult(
                criterion=criterion.criterion,
                evaluator_model=self.model_name,
                decision=parsed_result["decision"],
                confidence=parsed_result["confidence"],
                reasoning=parsed_result["reasoning"],
                execution_time=time.time() - start_time
            )
        
        except Exception as e:
            return EvaluationResult(
                criterion=criterion.criterion,
                evaluator_model=self.model_name,
                decision="ERROR",
                confidence=0.0,
                reasoning=f"Evaluation failed: {str(e)}",
                execution_time=time.time() - start_time
            )
    
    def _build_evaluation_prompt(self, user_input: str, agent_response: str, 
                                criterion: SemanticCriterion) -> str:
        """Build evaluation prompt for the model"""
        return f"""You are an expert evaluator assessing AI agent responses.

USER INPUT: "{user_input}"
AGENT RESPONSE: "{agent_response}"
EVALUATION CRITERION: "{criterion.criterion}"

Evaluate whether the agent response meets the criterion. Consider:
- Semantic meaning, not just exact words
- Overall intent and appropriateness
- Context and tone

Respond in this exact JSON format:
{{
    "decision": "YES|NO",
    "reasoning": "Brief explanation of your evaluation"
}}

Decision guide:
- YES: The response meets the criterion
- NO: The response does not meet the criterion

You must choose either YES or NO. Be precise and objective in your evaluation."""
    
    async def _call_model(self, prompt: str) -> str:
        """Call the evaluator model - to be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement _call_model")
    
    def _parse_evaluation_response(self, response: str) -> Dict[str, Any]:
        """Parse model response into structured result"""
        try:
            # Try to extract JSON from response
            import re
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group())
                
                # Validate required fields
                decision = result.get("decision", "NO").upper()
                if decision not in ["YES", "NO"]:
                    decision = "NO"
                
                confidence = 1.0  # No longer using confidence scoring
                
                reasoning = result.get("reasoning", "No reasoning provided")
                
                return {
                    "decision": decision,
                    "confidence": confidence,
                    "reasoning": reasoning
                }
        except:
            pass
        
        # Fallback parsing if JSON fails
        response_upper = response.upper()
        if "YES" in response_upper:
            decision = "YES"
        else:
            decision = "NO"
        
        confidence = 1.0  # No longer using confidence scoring
        
        return {
            "decision": decision,
            "confidence": confidence,
            "reasoning": "Parsed from non-JSON response"
        }


class OpenAIEvaluator(EvaluatorClient):
    """OpenAI model evaluator"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key or self._get_api_key()
    
    def _get_api_key(self) -> str:
        return os.getenv("OPENAI_API_KEY", "")
    
    async def _call_model(self, prompt: str) -> str:
        """Call OpenAI API"""
        if not self.api_key:
            raise ValueError("OpenAI API key not provided")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 200
        }
        
        # Use requests for now - could be made async with aiohttp
        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result["choices"][0]["message"]["content"]


class AnthropicEvaluator(EvaluatorClient):
    """Anthropic Claude evaluator"""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None):
        super().__init__(model_name)
        self.api_key = api_key or self._get_api_key()
    
    def _get_api_key(self) -> str:
        return os.getenv("ANTHROPIC_API_KEY", "")
    
    async def _call_model(self, prompt: str) -> str:
        """Call Anthropic API"""
        if not self.api_key:
            raise ValueError("Anthropic API key not provided")
        
        headers = {
            "x-api-key": self.api_key,
            "Content-Type": "application/json",
            "anthropic-version": "2023-06-01"
        }
        
        payload = {
            "model": self.model_name,
            "max_tokens": 200,
            "temperature": 0.1,
            "messages": [{"role": "user", "content": prompt}]
        }
        
        response = requests.post(
            "https://api.anthropic.com/v1/messages",
            headers=headers,
            json=payload,
            timeout=30
        )
        response.raise_for_status()
        
        result = response.json()
        return result["content"][0]["text"]


class LocalEvaluator(EvaluatorClient):
    """Local model evaluator (e.g., Ollama)"""
    
    def __init__(self, model_name: str, endpoint: str = "http://localhost:11434"):
        super().__init__(model_name)
        self.endpoint = endpoint
    
    async def _call_model(self, prompt: str) -> str:
        """Call local model via Ollama API"""
        payload = {
            "model": self.model_name.replace("local-", ""),
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.1,
                "num_predict": 200
            }
        }
        
        response = requests.post(
            f"{self.endpoint}/api/generate",
            json=payload,
            timeout=60
        )
        response.raise_for_status()
        
        result = response.json()
        return result["response"]


class EvaluationLoop:
    """Main evaluation loop orchestrator"""
    
    def __init__(self, config: EvaluationLoopConfig):
        self.config = config
        self.evaluators = self._create_evaluators()
    
    def _create_evaluators(self) -> List[EvaluatorClient]:
        """Create evaluator clients based on configuration"""
        evaluators = []
        
        for model_name in self.config.evaluator_models:
            if model_name.startswith(("gpt-", "o1-")):
                evaluators.append(OpenAIEvaluator(model_name))
            elif model_name.startswith(("claude-", "sonnet", "haiku", "opus")):
                evaluators.append(AnthropicEvaluator(model_name))
            elif model_name.startswith(("llama", "mistral", "local-")):
                evaluators.append(LocalEvaluator(model_name))
            else:
                # Custom evaluator - could be extended
                evaluators.append(EvaluatorClient(model_name))
        
        return evaluators
    
    async def evaluate_response(self, user_input: str, agent_response: str, 
                               criteria: List[SemanticCriterion]) -> List[ConsensusResult]:
        """Run evaluation loop for agent response against all criteria"""
        results = []
        
        for criterion in criteria:
            consensus_result = await self._evaluate_single_criterion(
                user_input, agent_response, criterion
            )
            results.append(consensus_result)
        
        return results
    
    async def _evaluate_single_criterion(self, user_input: str, agent_response: str,
                                        criterion: SemanticCriterion) -> ConsensusResult:
        """Evaluate single criterion across all evaluators with iterations"""
        all_evaluations = []
        
        # Run for specified iterations
        for iteration in range(self.config.iterations):
            if self.config.parallel_execution:
                # Run all evaluators in parallel
                tasks = [
                    evaluator.evaluate(user_input, agent_response, criterion)
                    for evaluator in self.evaluators
                ]
                iteration_results = await asyncio.gather(*tasks, return_exceptions=True)
            else:
                # Run evaluators sequentially
                iteration_results = []
                for evaluator in self.evaluators:
                    result = await evaluator.evaluate(user_input, agent_response, criterion)
                    iteration_results.append(result)
            
            # Filter out exceptions and add to all evaluations
            for result in iteration_results:
                if isinstance(result, EvaluationResult):
                    all_evaluations.append(result)
        
        # Calculate consensus
        return self._calculate_consensus(criterion, all_evaluations)
    
    def _calculate_consensus(self, criterion: SemanticCriterion, 
                           evaluations: List[EvaluationResult]) -> ConsensusResult:
        """Calculate consensus from multiple evaluation results"""
        if not evaluations:
            return ConsensusResult(
                criterion=criterion.criterion,
                consensus_score=0.0,
                passed=False,
                individual_results=[]
            )
        
        # Convert decisions to scores
        decision_scores = []
        for eval_result in evaluations:
            if eval_result.decision == "YES":
                score = 1.0
            else:  # NO
                score = 0.0
            
            decision_scores.append(score)
        
        # Calculate consensus score (average)
        consensus_score = sum(decision_scores) / len(decision_scores)
        
        # Apply criterion weight
        weighted_score = consensus_score * criterion.weight
        
        # Determine if passed based on threshold
        passed = consensus_score >= self.config.consensus_threshold
        
        return ConsensusResult(
            criterion=criterion.criterion,
            consensus_score=consensus_score,
            passed=passed,
            individual_results=evaluations,
            weighted_score=weighted_score
        )


def create_evaluation_loop(config_dict: Dict[str, Any]) -> EvaluationLoop:
    """Create evaluation loop from configuration dictionary"""
    config = EvaluationLoopConfig(
        iterations=config_dict.get("iterations", 1),
        evaluator_models=config_dict.get("evaluator_models", ["claude-sonnet-4"]),
        consensus_threshold=config_dict.get("consensus_threshold", 0.67),
        timeout=config_dict.get("timeout", 30),
        parallel_execution=config_dict.get("parallel_execution", True),
        retry_count=config_dict.get("retry_count", 2)
    )
    
    return EvaluationLoop(config)