#!/usr/bin/env python3
"""
Performance demonstration script showing different testLLM configuration modes.

This script demonstrates the speed differences between Mistral and Claude evaluators,
and how different configuration modes affect performance.

Usage:
    python examples/performance_demo.py

Requirements:
    - MISTRAL_API_KEY in environment (for fast demos)
    - ANTHROPIC_API_KEY in environment (for thorough demos)
"""

import time
from testllm import LocalAgent, conversation_flow


class ExampleAgent:
    """Simple example agent for demonstration"""
    
    def __call__(self, prompt: str) -> str:
        """Simple responses for testing"""
        if "hello" in prompt.lower():
            return "Hello! I'm here to help you with anything you need."
        elif "weather" in prompt.lower():
            return "I'd be happy to help with weather information. What city are you interested in?"
        elif "help" in prompt.lower():
            return "Of course! I'm here to assist you. What can I help you with today?"
        else:
            return "I understand your request. How can I assist you further?"


def demo_configuration_modes():
    """Demonstrate different configuration modes and their performance"""
    
    # Create test agent
    agent = LocalAgent(model=ExampleAgent())
    
    print("🚀 testLLM Performance Demo")
    print("=" * 50)
    
    configurations = [
        ("fast", "⚡ Fast Mode (Mistral, 1 iteration)"),
        ("production", "🏭 Production Mode (Mistral + Claude, 2 iterations)"), 
        ("thorough", "🔍 Thorough Mode (Mistral + Claude, 3 iterations, debug)")
    ]
    
    for config_mode, description in configurations:
        print(f"\n{description}")
        print("-" * 40)
        
        try:
            # Create flow with specific configuration
            flow = conversation_flow(
                f"demo_{config_mode}", 
                f"Demo flow in {config_mode} mode",
                config_mode=config_mode
            )
            
            # Add a simple test step
            flow.step(
                "Hello, can you help me?",
                criteria=[
                    "Response should be friendly and welcoming",
                    "Response should offer assistance",
                    "Response should be professional"
                ]
            )
            
            # Time the execution
            start_time = time.time()
            result = flow.execute_sync(agent)
            end_time = time.time()
            
            # Report results
            duration = end_time - start_time
            print(f"✅ Completed in {duration:.1f} seconds")
            print(f"   Passed: {result.passed}")
            print(f"   Score: {result.overall_score:.2f}")
            print(f"   Steps executed: {result.steps_executed}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")
            if "API key" in str(e):
                print("   💡 Add the required API key to your .env file")


def demo_evaluator_comparison():
    """Compare performance between different evaluators"""
    
    agent = LocalAgent(model=ExampleAgent())
    
    print("\n\n🔬 Evaluator Performance Comparison")
    print("=" * 50)
    
    evaluator_configs = [
        (["mistral-large-latest"], "Mistral Large (fastest)"),
        (["claude-sonnet-4-20250514"], "Claude Sonnet 4 (thorough)"),
        (["mistral-large-latest", "claude-sonnet-4-20250514"], "Both evaluators")
    ]
    
    for evaluators, description in evaluator_configs:
        print(f"\n{description}")
        print("-" * 30)
        
        try:
            flow = conversation_flow(
                f"eval_demo",
                f"Evaluator comparison: {description}",
                evaluator_models=evaluators,
                config_mode="fast"  # Use fast mode settings but with custom evaluators
            )
            
            flow.step(
                "What's the weather like today?",
                criteria=["Response should acknowledge the weather question"]
            )
            
            start_time = time.time()
            result = flow.execute_sync(agent)
            end_time = time.time()
            
            duration = end_time - start_time
            print(f"✅ Completed in {duration:.1f} seconds")
            print(f"   Evaluators used: {len(evaluators)}")
            print(f"   Score: {result.overall_score:.2f}")
            
        except Exception as e:
            print(f"❌ Failed: {e}")


def demo_optimization_tips():
    """Demonstrate optimization techniques"""
    
    agent = LocalAgent(model=ExampleAgent())
    
    print("\n\n💡 Optimization Tips Demo")
    print("=" * 50)
    
    # Example 1: Too many criteria (slower)
    print("\n❌ Inefficient: Too many criteria")
    print("-" * 35)
    
    try:
        flow = conversation_flow("slow_demo", config_mode="fast")
        flow.step(
            "Hello there!",
            criteria=[
                "Response should be friendly",
                "Response should be welcoming", 
                "Response should be professional",
                "Response should offer help",
                "Response should be conversational",
                "Response should be appropriate"
            ]
        )
        
        start_time = time.time()
        result = flow.execute_sync(agent)
        end_time = time.time()
        
        print(f"⏱️  {end_time - start_time:.1f} seconds (6 criteria)")
        
    except Exception as e:
        print(f"❌ Failed: {e}")
    
    # Example 2: Optimized criteria (faster)
    print("\n✅ Efficient: Consolidated criteria")
    print("-" * 35)
    
    try:
        flow = conversation_flow("fast_demo", config_mode="fast")
        flow.step(
            "Hello there!",
            criteria=[
                "Response should be friendly, professional, and offer help"
            ]
        )
        
        start_time = time.time()
        result = flow.execute_sync(agent)
        end_time = time.time()
        
        print(f"⏱️  {end_time - start_time:.1f} seconds (1 consolidated criterion)")
        
    except Exception as e:
        print(f"❌ Failed: {e}")


if __name__ == "__main__":
    print("Starting testLLM performance demonstrations...")
    print("Make sure you have MISTRAL_API_KEY and/or ANTHROPIC_API_KEY in your .env file")
    
    demo_configuration_modes()
    demo_evaluator_comparison() 
    demo_optimization_tips()
    
    print("\n\n🎉 Demo complete!")
    print("\nKey takeaways:")
    print("• Mistral Large is 3-5x faster than Claude Sonnet 4")
    print("• Use 'fast' mode for development (default)")
    print("• Use 'production' mode for CI/CD pipelines")
    print("• Consolidate criteria for better performance")
    print("• testLLM automatically falls back between evaluators")