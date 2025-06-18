"""
Zapier MCP Gmail Agent Example

This agent demonstrates real integration with Zapier's MCP server to send Gmail messages.
Uses the actual Zapier MCP API for email functionality.

Prerequisites:
- pip install fastmcp pydantic-ai (or install from requirements.txt)
- ZAPIER_MCP_URL environment variable set to your Zapier MCP server URL
- ANTHROPIC_API_KEY environment variable set
"""

import os
import asyncio
import json
from typing import Optional, Dict, Any
from pydantic import BaseModel
from pydantic_ai import Agent
from pydantic_ai.models.anthropic import AnthropicModel
from fastmcp import Client
from fastmcp.client.transports import StreamableHttpTransport


class EmailResult(BaseModel):
    """Model for email sending results"""
    success: bool
    message: str
    details: Optional[Dict[str, Any]] = None


class ZapierMCPClient:
    """Client for interacting with Zapier MCP server"""
    
    def __init__(self, server_url: str):
        self.server_url = server_url
        self.transport = StreamableHttpTransport(server_url)
        self.client = Client(transport=self.transport)
    
    async def send_email(self, to: str, subject: str, body: str, cc: str = None, bcc: str = None) -> EmailResult:
        """Send email through Zapier MCP Gmail integration"""
        try:
            async with self.client:
                # Prepare parameters for the gmail_send_email tool
                params = {
                    "instructions": "Send email through Gmail",
                    "to": to,
                    "subject": subject,
                    "body": body
                }
                
                # Add optional parameters
                if cc:
                    params["cc"] = cc
                if bcc:
                    params["bcc"] = bcc
                
                # Call the Gmail send email tool
                result = await self.client.call_tool("gmail_send_email", params)
                
                # Parse result
                if result and len(result) > 0:
                    json_result = json.loads(result[0].text)
                    return EmailResult(
                        success=True,
                        message=f"Email sent successfully to {to}",
                        details=json_result
                    )
                else:
                    return EmailResult(
                        success=False,
                        message="No response from email service"
                    )
                    
        except Exception as e:
            return EmailResult(
                success=False,
                message=f"Failed to send email: {str(e)}"
            )
    
    async def list_available_tools(self) -> list:
        """List all available tools from Zapier MCP"""
        try:
            async with self.client:
                tools = await self.client.list_tools()
                return [{"name": t.name, "description": t.description} for t in tools]
        except Exception as e:
            return [{"error": f"Failed to list tools: {str(e)}"}]


# Initialize components
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
zapier_mcp_url = os.getenv('ZAPIER_MCP_URL')

if not anthropic_api_key:
    raise ValueError("ANTHROPIC_API_KEY environment variable is required")

if not zapier_mcp_url:
    # Use the example URL from the documentation as fallback
    zapier_mcp_url = "https://mcp.zapier.com/api/mcp/s/MzIxNTg4ODQtYjIzMy00Zjk3LTkyZDYtMDI2NmJkODE4OGZkOjdjNDJhY2MwLTYwNzMtNDdiMS05YjAyLWYyZjQ2MTkyYzIzZg==/mcp"
    print(f"Using example Zapier MCP URL: {zapier_mcp_url}")

# Initialize Claude model and MCP client
claude_model = AnthropicModel('claude-sonnet-4-20250514')
mcp_client = ZapierMCPClient(zapier_mcp_url)

# Create Gmail agent
gmail_agent = Agent(
    model=claude_model,
    system_prompt="""You are a helpful Gmail assistant that can send emails through Zapier integration.

    When users want to send emails:
    1. Extract the recipient email address, subject, and message content
    2. Use the send_gmail_email tool to send the message
    3. Provide clear confirmation of success or failure
    
    For email validation, use the validate_email tool.
    
    Always be helpful and ask for clarification if email details are missing or unclear.
    Be conversational and friendly in your responses."""
)


@gmail_agent.tool_plain
def send_gmail_email(to_address: str, subject: str, message_body: str, cc_address: str = None) -> EmailResult:
    """Send an email through Zapier MCP Gmail integration"""
    
    # Validate email format
    if "@" not in to_address or "." not in to_address:
        return EmailResult(
            success=False,
            message="Invalid email address format"
        )
    
    # Use asyncio to run the async MCP call
    try:
        loop = asyncio.get_event_loop()
        result = loop.run_until_complete(
            mcp_client.send_email(to_address, subject, message_body, cc_address)
        )
        return result
    except Exception as e:
        return EmailResult(
            success=False,
            message=f"Error sending email: {str(e)}"
        )


@gmail_agent.tool_plain
def validate_email(email_address: str) -> dict:
    """Validate email address format"""
    import re
    
    # Basic email validation regex
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    is_valid = bool(re.match(pattern, email_address))
    
    return {
        "email": email_address,
        "is_valid": is_valid,
        "message": "Valid email format" if is_valid else "Invalid email format - please check the address"
    }


@gmail_agent.tool_plain
def list_zapier_tools() -> dict:
    """List available tools from Zapier MCP server"""
    try:
        loop = asyncio.get_event_loop()
        tools = loop.run_until_complete(mcp_client.list_available_tools())
        return {
            "available_tools": tools,
            "count": len(tools)
        }
    except Exception as e:
        return {"error": f"Failed to list tools: {str(e)}"}


class ZapierGmailWrapper:
    """Wrapper for synchronous usage of the Gmail agent"""
    
    def __init__(self, agent):
        self.agent = agent
    
    def __call__(self, prompt: str) -> str:
        """Synchronous interface"""
        return asyncio.run(self._async_call(prompt))
    
    async def _async_call(self, prompt: str) -> str:
        """Async call to agent"""
        try:
            result = await self.agent.run(prompt)
            return result.data
        except Exception as e:
            return f"Sorry, I encountered an error: {str(e)}"


# Create wrapped agent for easy use
zapier_gmail_agent = ZapierGmailWrapper(gmail_agent)


async def demo():
    """Demonstrate Zapier Gmail agent capabilities"""
    print("📧 Zapier MCP Gmail Agent Demo\n")
    
    print("🔧 Available Tools:")
    tools_response = await gmail_agent.run("What tools are available through Zapier?")
    print(f"{tools_response.data}\n")
    
    print("🔍 Email Validation:")
    validation_response = await gmail_agent.run("Is 'test@example.com' a valid email address?")
    print(f"{validation_response.data}\n")
    
    print("✉️ Send Test Email:")
    email_response = await gmail_agent.run(
        "Send an email to test@example.com with the subject 'Hello from Zapier MCP' and message 'This is a test email sent through the Zapier MCP integration!'"
    )
    print(f"{email_response.data}\n")
    
    print("📝 Complex Email Request:")
    complex_response = await gmail_agent.run(
        "I need to send a professional email to client@company.com about our project update. The subject should be 'Project Status Update' and mention that we're on track for the deadline."
    )
    print(f"{complex_response.data}\n")


if __name__ == "__main__":
    print("🚀 Starting Zapier MCP Gmail Agent Demo...")
    print("This agent uses real Zapier MCP integration for Gmail\n")
    
    try:
        asyncio.run(demo())
    except KeyboardInterrupt:
        print("\n👋 Demo interrupted by user")
    except Exception as e:
        print(f"\n❌ Demo failed: {str(e)}")