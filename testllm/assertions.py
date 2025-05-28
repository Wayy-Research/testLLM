"""
Assertion library for testLLM Framework
"""

import re
import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Union


@dataclass
class AssertionResult:
    """Result of an assertion check"""
    assertion_type: str
    passed: bool
    expected: Any = None
    actual: Any = None
    message: str = ""


class BaseAssertion(ABC):
    """Base class for all assertions"""
    
    @abstractmethod
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        """Check the assertion against the response"""
        pass


class ContainsAssertion(BaseAssertion):
    """Assert that response contains a specific pattern"""
    
    def __init__(self, pattern: str, case_sensitive: bool = False):
        self.pattern = pattern
        self.case_sensitive = case_sensitive
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        if self.case_sensitive:
            contains = self.pattern in response
        else:
            contains = self.pattern.lower() in response.lower()
        
        return AssertionResult(
            assertion_type="contains",
            passed=contains,
            expected=self.pattern,
            actual=response,
            message=f"Expected response to contain '{self.pattern}'" if not contains else ""
        )


class ExcludesAssertion(BaseAssertion):
    """Assert that response excludes a specific pattern"""
    
    def __init__(self, pattern: str, case_sensitive: bool = False):
        self.pattern = pattern
        self.case_sensitive = case_sensitive
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        if self.case_sensitive:
            excludes = self.pattern not in response
        else:
            excludes = self.pattern.lower() not in response.lower()
        
        return AssertionResult(
            assertion_type="excludes",
            passed=excludes,
            expected=f"not {self.pattern}",
            actual=response,
            message=f"Expected response to not contain '{self.pattern}'" if not excludes else ""
        )


class RegexAssertion(BaseAssertion):
    """Assert that response matches a regex pattern"""
    
    def __init__(self, pattern: str, flags: int = 0):
        self.pattern = pattern
        self.flags = flags
        self.compiled_pattern = re.compile(pattern, flags)
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        match = self.compiled_pattern.search(response)
        
        return AssertionResult(
            assertion_type="regex",
            passed=match is not None,
            expected=self.pattern,
            actual=response,
            message=f"Expected response to match regex '{self.pattern}'" if not match else ""
        )


class MaxLengthAssertion(BaseAssertion):
    """Assert maximum response length"""
    
    def __init__(self, max_length: int):
        self.max_length = max_length
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        actual_length = len(response)
        passed = actual_length <= self.max_length
        
        return AssertionResult(
            assertion_type="max_length",
            passed=passed,
            expected=self.max_length,
            actual=actual_length,
            message=f"Expected response length <= {self.max_length}, got {actual_length}" if not passed else ""
        )


class MinLengthAssertion(BaseAssertion):
    """Assert minimum response length"""
    
    def __init__(self, min_length: int):
        self.min_length = min_length
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        actual_length = len(response)
        passed = actual_length >= self.min_length
        
        return AssertionResult(
            assertion_type="min_length",
            passed=passed,
            expected=self.min_length,
            actual=actual_length,
            message=f"Expected response length >= {self.min_length}, got {actual_length}" if not passed else ""
        )


class SentimentAssertion(BaseAssertion):
    """Assert response sentiment using keyword-based analysis"""
    
    POSITIVE_KEYWORDS = {
        'good', 'great', 'excellent', 'wonderful', 'amazing', 'fantastic', 
        'awesome', 'brilliant', 'perfect', 'love', 'happy', 'glad', 'pleased',
        'satisfied', 'delighted', 'thrilled', 'excited', 'yes', 'sure', 'absolutely',
        'help', 'assist', 'support', 'welcome', 'here', 'ready', 'available'
    }
    
    NEGATIVE_KEYWORDS = {
        'bad', 'terrible', 'awful', 'horrible', 'disgusting', 'hate', 'angry',
        'sad', 'disappointed', 'frustrated', 'annoyed', 'upset', 'mad', 'furious',
        'no', 'never', 'impossible', 'can\'t', 'won\'t', 'refuse', 'error', 'fail'
    }
    
    def __init__(self, expected_sentiment: str):
        if expected_sentiment not in ['positive', 'negative', 'neutral']:
            raise ValueError("Sentiment must be 'positive', 'negative', or 'neutral'")
        self.expected_sentiment = expected_sentiment
    
    def _analyze_sentiment(self, text: str) -> str:
        """Simple keyword-based sentiment analysis"""
        text_lower = text.lower()
        words = re.findall(r'\b\w+\b', text_lower)
        
        positive_count = sum(1 for word in words if word in self.POSITIVE_KEYWORDS)
        negative_count = sum(1 for word in words if word in self.NEGATIVE_KEYWORDS)
        
        if positive_count > negative_count:
            return 'positive'
        elif negative_count > positive_count:
            return 'negative'
        else:
            return 'neutral'
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        actual_sentiment = self._analyze_sentiment(response)
        passed = actual_sentiment == self.expected_sentiment
        
        return AssertionResult(
            assertion_type="sentiment",
            passed=passed,
            expected=self.expected_sentiment,
            actual=actual_sentiment,
            message=f"Expected {self.expected_sentiment} sentiment, got {actual_sentiment}" if not passed else ""
        )


class JsonValidAssertion(BaseAssertion):
    """Assert that response is valid JSON"""
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        try:
            json.loads(response)
            return AssertionResult(
                assertion_type="json_valid",
                passed=True,
                expected="valid JSON",
                actual="valid JSON"
            )
        except json.JSONDecodeError as e:
            return AssertionResult(
                assertion_type="json_valid",
                passed=False,
                expected="valid JSON",
                actual=f"invalid JSON: {e}",
                message=f"Expected valid JSON, got parse error: {e}"
            )


class JsonSchemaAssertion(BaseAssertion):
    """Assert that response matches a JSON schema"""
    
    def __init__(self, schema: Dict[str, Any]):
        self.schema = schema
    
    def _validate_schema(self, data: Any, schema: Dict[str, Any]) -> tuple[bool, str]:
        """Simple JSON schema validation"""
        if 'type' in schema:
            expected_type = schema['type']
            if expected_type == 'object' and not isinstance(data, dict):
                return False, f"Expected object, got {type(data).__name__}"
            elif expected_type == 'array' and not isinstance(data, list):
                return False, f"Expected array, got {type(data).__name__}"
            elif expected_type == 'string' and not isinstance(data, str):
                return False, f"Expected string, got {type(data).__name__}"
            elif expected_type == 'number' and not isinstance(data, (int, float)):
                return False, f"Expected number, got {type(data).__name__}"
            elif expected_type == 'boolean' and not isinstance(data, bool):
                return False, f"Expected boolean, got {type(data).__name__}"
        
        if 'properties' in schema and isinstance(data, dict):
            for prop, prop_schema in schema['properties'].items():
                if 'required' in schema and prop in schema['required'] and prop not in data:
                    return False, f"Required property '{prop}' missing"
                if prop in data:
                    valid, msg = self._validate_schema(data[prop], prop_schema)
                    if not valid:
                        return False, f"Property '{prop}': {msg}"
        
        return True, ""
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        try:
            data = json.loads(response)
            valid, message = self._validate_schema(data, self.schema)
            
            return AssertionResult(
                assertion_type="json_schema",
                passed=valid,
                expected="data matching schema",
                actual=data,
                message=message if not valid else ""
            )
        except json.JSONDecodeError as e:
            return AssertionResult(
                assertion_type="json_schema",
                passed=False,
                expected="valid JSON matching schema",
                actual=f"invalid JSON: {e}",
                message=f"Expected valid JSON, got parse error: {e}"
            )


class ToolUsageAssertion(BaseAssertion):
    """Assert that agent used a specific tool"""
    
    def __init__(self, tool_name: str):
        self.tool_name = tool_name
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        tool_calls = agent.get_tool_calls()
        used_tools = [call.get('tool_name', call.get('name', '')) for call in tool_calls]
        passed = self.tool_name in used_tools
        
        return AssertionResult(
            assertion_type="tool_usage",
            passed=passed,
            expected=self.tool_name,
            actual=used_tools,
            message=f"Expected agent to use tool '{self.tool_name}', used: {used_tools}" if not passed else ""
        )


class TokenCountAssertion(BaseAssertion):
    """Assert approximate token count (rough estimation)"""
    
    def __init__(self, max_tokens: int):
        self.max_tokens = max_tokens
    
    def _estimate_tokens(self, text: str) -> int:
        """Rough token estimation: ~4 characters per token on average"""
        return len(text) // 4
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        estimated_tokens = self._estimate_tokens(response)
        passed = estimated_tokens <= self.max_tokens
        
        return AssertionResult(
            assertion_type="token_count",
            passed=passed,
            expected=self.max_tokens,
            actual=estimated_tokens,
            message=f"Expected <= {self.max_tokens} tokens, estimated {estimated_tokens}" if not passed else ""
        )


class AllOfAssertion(BaseAssertion):
    """Assert that all given assertions pass"""
    
    def __init__(self, assertions: List[BaseAssertion]):
        self.assertions = assertions
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        failed_assertions = []
        
        for assertion in self.assertions:
            result = assertion.check(response, agent)
            if not result.passed:
                failed_assertions.append(f"{result.assertion_type}: {result.message}")
        
        passed = len(failed_assertions) == 0
        
        return AssertionResult(
            assertion_type="all_of",
            passed=passed,
            expected="all assertions to pass",
            actual=f"{len(self.assertions) - len(failed_assertions)}/{len(self.assertions)} passed",
            message=f"Failed assertions: {'; '.join(failed_assertions)}" if not passed else ""
        )


class AnyOfAssertion(BaseAssertion):
    """Assert that at least one of the given assertions passes"""
    
    def __init__(self, assertions: List[BaseAssertion]):
        self.assertions = assertions
    
    def check(self, response: str, agent: 'AgentUnderTest') -> AssertionResult:
        passed_assertions = []
        failed_assertions = []
        
        for assertion in self.assertions:
            result = assertion.check(response, agent)
            if result.passed:
                passed_assertions.append(result.assertion_type)
            else:
                failed_assertions.append(f"{result.assertion_type}: {result.message}")
        
        passed = len(passed_assertions) > 0
        
        return AssertionResult(
            assertion_type="any_of",
            passed=passed,
            expected="at least one assertion to pass",
            actual=f"{len(passed_assertions)}/{len(self.assertions)} passed",
            message=f"No assertions passed. Failures: {'; '.join(failed_assertions)}" if not passed else ""
        )


def create_assertion_from_dict(assertion_dict: Dict[str, Any]) -> BaseAssertion:
    """Factory function to create assertions from dictionary definitions"""
    assertion_type = assertion_dict.get("type")
    
    if assertion_type == "contains":
        return ContainsAssertion(
            assertion_dict["value"],
            assertion_dict.get("case_sensitive", False)
        )
    elif assertion_type == "excludes":
        return ExcludesAssertion(
            assertion_dict["value"],
            assertion_dict.get("case_sensitive", False)
        )
    elif assertion_type == "regex":
        return RegexAssertion(
            assertion_dict["value"],
            assertion_dict.get("flags", 0)
        )
    elif assertion_type == "max_length":
        return MaxLengthAssertion(assertion_dict["value"])
    elif assertion_type == "min_length":
        return MinLengthAssertion(assertion_dict["value"])
    elif assertion_type == "sentiment":
        return SentimentAssertion(assertion_dict["value"])
    elif assertion_type == "json_valid":
        return JsonValidAssertion()
    elif assertion_type == "json_schema":
        return JsonSchemaAssertion(assertion_dict["value"])
    elif assertion_type == "tool_usage":
        return ToolUsageAssertion(assertion_dict["value"])
    elif assertion_type == "token_count":
        return TokenCountAssertion(assertion_dict["value"])
    elif assertion_type == "all_of":
        sub_assertions = [create_assertion_from_dict(a) for a in assertion_dict["value"]]
        return AllOfAssertion(sub_assertions)
    elif assertion_type == "any_of":
        sub_assertions = [create_assertion_from_dict(a) for a in assertion_dict["value"]]
        return AnyOfAssertion(sub_assertions)
    else:
        raise ValueError(f"Unknown assertion type: {assertion_type}")