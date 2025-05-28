"""
Simple Pytest plugin for testLLM Framework - just provides basic configuration
"""

import pytest


def pytest_configure(config):
    """Configure pytest for testLLM"""
    config.addinivalue_line(
        "markers", "testllm: mark test as a testLLM agent test"
    )


@pytest.fixture(scope="session") 
def testllm_config():
    """Basic testLLM configuration"""
    return {
        "test_directories": ["test_yaml/", "examples/"],
        "default_timeout": 30,
        "retry_count": 0
    }