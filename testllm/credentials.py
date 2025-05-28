"""
Credential management for testLLM Framework
"""

import os
from typing import Optional, Dict
from pathlib import Path


def load_dotenv_if_available():
    """Load .env file if it exists and python-dotenv is available"""
    try:
        from dotenv import load_dotenv
        
        # Look for .env file in current directory and parent directories
        env_path = Path.cwd() / ".env"
        if env_path.exists():
            load_dotenv(env_path)
            return True
        
        # Also check in testLLM project root if we're in a subdirectory
        for parent in Path.cwd().parents:
            env_path = parent / ".env"
            if env_path.exists():
                load_dotenv(env_path)
                return True
                
    except ImportError:
        # python-dotenv not installed, fall back to environment variables only
        pass
    
    return False


def get_api_key(service: str, key_name: Optional[str] = None) -> Optional[str]:
    """
    Get API key for a service with .env file support
    
    Args:
        service: Service name (e.g., 'openai', 'anthropic')
        key_name: Override the default environment variable name
    
    Returns:
        API key if found, None otherwise
    """
    # Load .env file if available
    load_dotenv_if_available()
    
    # Default key names for common services
    default_keys = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'claude': 'ANTHROPIC_API_KEY',  # Alias for anthropic
        'together': 'TOGETHER_API_KEY',
        'replicate': 'REPLICATE_API_TOKEN',
        'huggingface': 'HUGGINGFACE_API_TOKEN',
        'cohere': 'COHERE_API_KEY',
    }
    
    # Use provided key name or default
    env_var = key_name or default_keys.get(service.lower())
    if not env_var:
        raise ValueError(f"Unknown service '{service}' and no key_name provided")
    
    return os.getenv(env_var)


def get_all_api_keys() -> Dict[str, str]:
    """
    Get all available API keys
    
    Returns:
        Dictionary mapping service names to API keys
    """
    load_dotenv_if_available()
    
    api_keys = {}
    
    # Check for common API keys
    key_mappings = {
        'openai': 'OPENAI_API_KEY',
        'anthropic': 'ANTHROPIC_API_KEY',
        'together': 'TOGETHER_API_KEY',
        'replicate': 'REPLICATE_API_TOKEN',
        'huggingface': 'HUGGINGFACE_API_TOKEN',
        'cohere': 'COHERE_API_KEY',
    }
    
    for service, env_var in key_mappings.items():
        key = os.getenv(env_var)
        if key:
            api_keys[service] = key
    
    return api_keys


def validate_credentials_for_models(model_names: list) -> Dict[str, bool]:
    """
    Validate that required credentials are available for given models
    
    Args:
        model_names: List of model names to check
    
    Returns:
        Dictionary mapping model names to credential availability
    """
    load_dotenv_if_available()
    
    results = {}
    
    for model in model_names:
        model_lower = model.lower()
        
        if model_lower.startswith(('gpt-', 'o1-')):
            results[model] = bool(get_api_key('openai'))
        elif model_lower.startswith(('claude-', 'sonnet', 'haiku', 'opus')):
            results[model] = bool(get_api_key('anthropic'))
        elif model_lower.startswith(('llama', 'mistral', 'local-')):
            # Local models don't need credentials
            results[model] = True
        else:
            # Unknown model type - assume credentials needed but not available
            results[model] = False
    
    return results


def ensure_credentials(service: str, key_name: Optional[str] = None) -> str:
    """
    Ensure API key is available for a service, raise error if not found
    
    Args:
        service: Service name
        key_name: Override the default environment variable name
    
    Returns:
        API key
    
    Raises:
        ValueError: If API key is not found
    """
    api_key = get_api_key(service, key_name)
    if not api_key:
        env_var = key_name or {
            'openai': 'OPENAI_API_KEY',
            'anthropic': 'ANTHROPIC_API_KEY',
            'claude': 'ANTHROPIC_API_KEY',
        }.get(service.lower(), f"{service.upper()}_API_KEY")
        
        raise ValueError(
            f"API key for {service} not found. "
            f"Please set {env_var} in your environment or .env file."
        )
    
    return api_key