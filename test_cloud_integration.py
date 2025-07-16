#!/usr/bin/env python3
"""
Test the complete cloud integration
"""

import sys
import time
from pathlib import Path

# Add testLLM to path
sys.path.insert(0, str(Path(__file__).parent))

from testllm import semantic_test, LocalAgent
from testllm.config import get_config

class WeatherAgent:
    """Simple test agent"""
    def __call__(self, prompt):
        if "weather" in prompt.lower():
            return "I'll check the current weather conditions for you. What city would you like to know about?"
        elif "hello" in prompt.lower():
            return "Hello! How can I help you today?"
        else:
            return "I understand your request. Let me help you with that."

def main():
    print("🧪 Testing testLLM Cloud Integration")
    print("=" * 50)
    
    # Check configuration
    config = get_config()
    print(f"✓ Dashboard URL: {config.dashboard_url}")
    print(f"✓ API Base URL: {config.api_base_url}")
    print(f"✓ Telemetry Enabled: {config.telemetry_enabled}")
    print(f"✓ API Key Set: {'Yes' if config.api_key else 'No'}")
    
    if not config.api_key:
        print("\n❌ No API key set!")
        print("Please set your API key:")
        print("export TESTLLM_API_KEY='your_api_key_here'")
        return
    
    # Create test agent
    agent = LocalAgent(model=WeatherAgent())
    
    # Create semantic test
    test = semantic_test("weather_integration_test", "Test weather agent integration")
    test.add_scenario(
        "What's the weather like in Seattle?",
        criteria=[
            "Response should acknowledge the weather question",
            "Response should mention checking or retrieving weather data",
            "Response should be helpful and professional"
        ]
    )
    
    test.add_scenario(
        "Hello there!",
        criteria=[
            "Response should be a friendly greeting",
            "Response should offer assistance"
        ]
    )
    
    print("\n🏃 Running semantic test...")
    
    try:
        results = test.execute_sync(agent)
        
        print(f"\n✅ Test completed successfully!")
        print(f"   Total scenarios: {len(results)}")
        
        for i, result in enumerate(results, 1):
            status = "✅ PASSED" if result.passed else "❌ FAILED"
            print(f"   Scenario {i}: {status} (Score: {result.overall_score:.2f})")
            print(f"   Input: {result.user_input}")
            print(f"   Response: {result.agent_response}")
            print()
        
        # The telemetry should have automatically sent data to the dashboard
        # and printed a dashboard URL
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()