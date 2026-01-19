"""
Framework Adapters - Parse tool calls from different LLM framework formats.

This module provides adapters for parsing tool calls from various formats
used by different LLM frameworks (Anthropic, LangChain, etc.)

These adapters are framework-agnostic by design - they work with the data
format, not by importing the framework libraries.

Example usage:
    from testllm.tool_testing import GenericToolAdapter, AnthropicAdapter

    # Parse from generic dict format
    call = GenericToolAdapter.parse_call({"tool_name": "search", "arguments": {...}})

    # Parse from Anthropic format
    call = AnthropicAdapter.parse_call(anthropic_tool_call)
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union
from datetime import datetime
import json
import uuid

from .types import ToolCall, ToolResponse, ToolCallStatus


class BaseToolAdapter(ABC):
    """Abstract base class for tool call adapters"""

    @classmethod
    @abstractmethod
    def parse_call(cls, data: Any) -> ToolCall:
        """Parse a tool call from framework-specific format"""
        pass

    @classmethod
    @abstractmethod
    def parse_response(cls, data: Any) -> ToolResponse:
        """Parse a tool response from framework-specific format"""
        pass

    @classmethod
    @abstractmethod
    def format_call(cls, call: ToolCall) -> Any:
        """Format a ToolCall into framework-specific format"""
        pass

    @classmethod
    @abstractmethod
    def format_response(cls, response: ToolResponse) -> Any:
        """Format a ToolResponse into framework-specific format"""
        pass

    @classmethod
    def can_parse(cls, data: Any) -> bool:
        """Check if this adapter can parse the given data"""
        return False


class GenericToolAdapter(BaseToolAdapter):
    """
    Generic adapter for dict-based tool calls.

    Handles the most common key patterns:
    - tool_name/name for tool name
    - arguments/input/parameters for arguments
    - id/call_id for call ID
    """

    @classmethod
    def parse_call(cls, data: Dict[str, Any]) -> ToolCall:
        """Parse a tool call from a generic dict format"""
        if isinstance(data, ToolCall):
            return data

        tool_name = data.get("tool_name") or data.get("name") or data.get("function", {}).get("name", "")

        # Try multiple argument key patterns
        arguments = (
            data.get("arguments") or
            data.get("input") or
            data.get("parameters") or
            data.get("function", {}).get("arguments") or
            {}
        )

        # Parse arguments if string
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"raw": arguments}

        call_id = data.get("id") or data.get("call_id") or str(uuid.uuid4())

        # Parse timestamp if present
        timestamp = None
        if "timestamp" in data:
            ts = data["timestamp"]
            if isinstance(ts, str):
                try:
                    timestamp = datetime.fromisoformat(ts.replace("Z", "+00:00"))
                except ValueError:
                    pass
            elif isinstance(ts, datetime):
                timestamp = ts

        # Parse status if present
        status = ToolCallStatus.PENDING
        if "status" in data:
            try:
                status = ToolCallStatus(data["status"])
            except ValueError:
                pass

        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            call_id=call_id,
            timestamp=timestamp,
            status=status,
            metadata=data.get("metadata", {})
        )

    @classmethod
    def parse_response(cls, data: Dict[str, Any]) -> ToolResponse:
        """Parse a tool response from a generic dict format"""
        if isinstance(data, ToolResponse):
            return data

        tool_name = data.get("tool_name") or data.get("name") or ""
        result = data.get("result") or data.get("output") or data.get("content")
        success = data.get("success", True)
        error_message = data.get("error_message") or data.get("error")

        return ToolResponse(
            tool_name=tool_name,
            result=result,
            success=success,
            error_message=error_message,
            metadata=data.get("metadata", {})
        )

    @classmethod
    def format_call(cls, call: ToolCall) -> Dict[str, Any]:
        """Format a ToolCall into generic dict format"""
        return call.to_dict()

    @classmethod
    def format_response(cls, response: ToolResponse) -> Dict[str, Any]:
        """Format a ToolResponse into generic dict format"""
        return response.to_dict()

    @classmethod
    def can_parse(cls, data: Any) -> bool:
        """Check if this is a dict with tool call indicators"""
        if not isinstance(data, dict):
            return False
        return any(key in data for key in ["tool_name", "name", "function"])


class AnthropicAdapter(BaseToolAdapter):
    """
    Adapter for Anthropic API tool call format.

    Anthropic format:
    {
        "type": "tool_use",
        "id": "toolu_123",
        "name": "search_flights",
        "input": {"destination": "NYC"}
    }
    """

    @classmethod
    def parse_call(cls, data: Dict[str, Any]) -> ToolCall:
        """Parse an Anthropic format tool call"""
        if isinstance(data, ToolCall):
            return data

        tool_name = data.get("name", "")
        arguments = data.get("input", {})
        call_id = data.get("id", str(uuid.uuid4()))

        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            call_id=call_id,
            timestamp=datetime.now(),
            status=ToolCallStatus.PENDING,
            metadata={"type": data.get("type", "tool_use")}
        )

    @classmethod
    def parse_response(cls, data: Dict[str, Any]) -> ToolResponse:
        """Parse an Anthropic format tool response"""
        # Anthropic tool responses are in tool_result format
        tool_use_id = data.get("tool_use_id", "")
        content = data.get("content", "")
        is_error = data.get("is_error", False)

        # Handle content blocks
        if isinstance(content, list):
            # Combine text blocks
            text_content = ""
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    text_content += block.get("text", "")
                elif isinstance(block, str):
                    text_content += block
            content = text_content

        return ToolResponse(
            tool_name="",  # Not always available in response
            result=content,
            success=not is_error,
            error_message=content if is_error else None,
            metadata={"tool_use_id": tool_use_id}
        )

    @classmethod
    def format_call(cls, call: ToolCall) -> Dict[str, Any]:
        """Format a ToolCall into Anthropic format"""
        return {
            "type": "tool_use",
            "id": call.call_id or f"toolu_{uuid.uuid4().hex[:8]}",
            "name": call.tool_name,
            "input": call.arguments
        }

    @classmethod
    def format_response(cls, response: ToolResponse) -> Dict[str, Any]:
        """Format a ToolResponse into Anthropic message format"""
        return {
            "type": "tool_result",
            "tool_use_id": response.metadata.get("tool_use_id", ""),
            "content": response.result if isinstance(response.result, str) else json.dumps(response.result),
            "is_error": not response.success
        }

    @classmethod
    def can_parse(cls, data: Any) -> bool:
        """Check if this is Anthropic format"""
        if not isinstance(data, dict):
            return False
        return data.get("type") == "tool_use" and "input" in data


class LangChainAdapter(BaseToolAdapter):
    """
    Adapter for LangChain tool call format.

    LangChain format (varies by version):
    {
        "tool": "search_flights",
        "tool_input": {"destination": "NYC"},
        "log": "...",
        "message_log": [...]
    }
    """

    @classmethod
    def parse_call(cls, data: Dict[str, Any]) -> ToolCall:
        """Parse a LangChain format tool call"""
        if isinstance(data, ToolCall):
            return data

        tool_name = data.get("tool") or data.get("name") or ""
        arguments = data.get("tool_input") or data.get("input") or {}

        # Handle string input
        if isinstance(arguments, str):
            try:
                arguments = json.loads(arguments)
            except json.JSONDecodeError:
                arguments = {"input": arguments}

        return ToolCall(
            tool_name=tool_name,
            arguments=arguments,
            call_id=str(uuid.uuid4()),
            timestamp=datetime.now(),
            status=ToolCallStatus.PENDING,
            metadata={"log": data.get("log", "")}
        )

    @classmethod
    def parse_response(cls, data: Dict[str, Any]) -> ToolResponse:
        """Parse a LangChain format tool response"""
        return ToolResponse(
            tool_name=data.get("tool", ""),
            result=data.get("output") or data.get("result"),
            success=True,
            metadata={}
        )

    @classmethod
    def format_call(cls, call: ToolCall) -> Dict[str, Any]:
        """Format a ToolCall into LangChain format"""
        return {
            "tool": call.tool_name,
            "tool_input": call.arguments,
            "log": call.metadata.get("log", "")
        }

    @classmethod
    def format_response(cls, response: ToolResponse) -> Dict[str, Any]:
        """Format a ToolResponse into LangChain format"""
        return {
            "tool": response.tool_name,
            "output": response.result
        }

    @classmethod
    def can_parse(cls, data: Any) -> bool:
        """Check if this is LangChain format"""
        if not isinstance(data, dict):
            return False
        return "tool" in data and "tool_input" in data


class AutoAdapter:
    """
    Automatically detect and use the appropriate adapter.

    Usage:
        call = AutoAdapter.parse_call(data)  # Auto-detects format
    """

    _adapters = [
        AnthropicAdapter,
        LangChainAdapter,
        GenericToolAdapter,  # Fallback
    ]

    @classmethod
    def parse_call(cls, data: Any) -> ToolCall:
        """Parse a tool call, auto-detecting the format"""
        if isinstance(data, ToolCall):
            return data

        for adapter in cls._adapters:
            if adapter.can_parse(data):
                return adapter.parse_call(data)

        # Last resort: try generic adapter
        return GenericToolAdapter.parse_call(data)

    @classmethod
    def parse_calls(cls, data: List[Any]) -> List[ToolCall]:
        """Parse multiple tool calls"""
        return [cls.parse_call(item) for item in data]

    @classmethod
    def parse_response(cls, data: Any) -> ToolResponse:
        """Parse a tool response, auto-detecting the format"""
        if isinstance(data, ToolResponse):
            return data

        # Use generic adapter for responses (most flexible)
        return GenericToolAdapter.parse_response(data)

    @classmethod
    def detect_format(cls, data: Any) -> Optional[str]:
        """Detect which format the data is in"""
        if AnthropicAdapter.can_parse(data):
            return "anthropic"
        elif LangChainAdapter.can_parse(data):
            return "langchain"
        elif GenericToolAdapter.can_parse(data):
            return "generic"
        return None


def parse_tool_calls(
    data: Union[Dict, List],
    adapter: Optional[type] = None
) -> List[ToolCall]:
    """
    Convenience function to parse tool calls from various formats.

    Args:
        data: Tool call data (single dict or list)
        adapter: Optional specific adapter to use

    Returns:
        List of ToolCall objects
    """
    if adapter is None:
        adapter_func = AutoAdapter.parse_call
    else:
        adapter_func = adapter.parse_call

    if isinstance(data, list):
        return [adapter_func(item) for item in data]
    else:
        return [adapter_func(data)]
