"""
Configuration management for testLLM cloud services
"""

import os
from typing import Optional, Dict, Any
from dataclasses import dataclass, field


@dataclass
class CloudConfig:
    """Configuration for testLLM cloud services"""
    
    # Dashboard API endpoints
    dashboard_url: str = "http://localhost:3000"
    api_base_url: str = "http://localhost:3000/api"
    
    # Authentication
    api_key: Optional[str] = None
    user_id: Optional[str] = None
    
    # Feature flags
    telemetry_enabled: bool = True
    cloud_reporting_enabled: bool = True
    custom_models_enabled: bool = False
    
    # Custom model endpoints (RunPod)
    models_api_url: str = "https://api.runpod.ai/v2/testllm-models"
    models_api_key: Optional[str] = None
    
    # Request configuration
    timeout: int = 30
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Data collection preferences
    collect_inputs: bool = True
    collect_outputs: bool = True
    collect_metadata: bool = True
    
    # Local settings
    local_storage_enabled: bool = True
    local_storage_path: str = "~/.testllm"
    
    # Advanced settings
    batch_size: int = 100
    flush_interval: int = 30  # seconds
    debug_mode: bool = False
    
    def __post_init__(self):
        """Initialize configuration from environment variables"""
        # Load from environment variables
        self.api_key = os.getenv("TESTLLM_API_KEY", self.api_key)
        self.user_id = os.getenv("TESTLLM_USER_ID", self.user_id)
        
        # Dashboard configuration
        self.dashboard_url = os.getenv("TESTLLM_DASHBOARD_URL", self.dashboard_url)
        self.api_base_url = os.getenv("TESTLLM_API_BASE_URL", self.api_base_url)
        
        # Custom models configuration
        self.models_api_url = os.getenv("TESTLLM_MODELS_API_URL", self.models_api_url)
        self.models_api_key = os.getenv("TESTLLM_MODELS_API_KEY", self.models_api_key)
        
        # Feature flags from environment
        self.telemetry_enabled = self._get_bool_env("TESTLLM_TELEMETRY_ENABLED", self.telemetry_enabled)
        self.cloud_reporting_enabled = self._get_bool_env("TESTLLM_CLOUD_REPORTING_ENABLED", self.cloud_reporting_enabled)
        self.custom_models_enabled = self._get_bool_env("TESTLLM_CUSTOM_MODELS_ENABLED", self.custom_models_enabled)
        
        # Data collection preferences
        self.collect_inputs = self._get_bool_env("TESTLLM_COLLECT_INPUTS", self.collect_inputs)
        self.collect_outputs = self._get_bool_env("TESTLLM_COLLECT_OUTPUTS", self.collect_outputs)
        self.collect_metadata = self._get_bool_env("TESTLLM_COLLECT_METADATA", self.collect_metadata)
        
        # Request configuration
        self.timeout = int(os.getenv("TESTLLM_TIMEOUT", self.timeout))
        self.max_retries = int(os.getenv("TESTLLM_MAX_RETRIES", self.max_retries))
        self.retry_delay = float(os.getenv("TESTLLM_RETRY_DELAY", self.retry_delay))
        
        # Local settings
        self.local_storage_enabled = self._get_bool_env("TESTLLM_LOCAL_STORAGE_ENABLED", self.local_storage_enabled)
        self.local_storage_path = os.getenv("TESTLLM_LOCAL_STORAGE_PATH", self.local_storage_path)
        
        # Advanced settings
        self.batch_size = int(os.getenv("TESTLLM_BATCH_SIZE", self.batch_size))
        self.flush_interval = int(os.getenv("TESTLLM_FLUSH_INTERVAL", self.flush_interval))
        self.debug_mode = self._get_bool_env("TESTLLM_DEBUG_MODE", self.debug_mode)
        
        # Expand home directory in paths
        self.local_storage_path = os.path.expanduser(self.local_storage_path)
    
    def _get_bool_env(self, key: str, default: bool) -> bool:
        """Get boolean value from environment variable"""
        value = os.getenv(key, str(default)).lower()
        return value in ("true", "1", "yes", "on")
    
    @property
    def is_cloud_enabled(self) -> bool:
        """Check if cloud services are enabled and configured"""
        return self.api_key is not None and self.telemetry_enabled
    
    @property
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        return self.api_key is not None
    
    def get_api_headers(self) -> Dict[str, str]:
        """Get API headers for requests"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"testllm-python/{self._get_version()}"
        }
        
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        
        return headers
    
    def get_models_api_headers(self) -> Dict[str, str]:
        """Get API headers for custom models requests"""
        headers = {
            "Content-Type": "application/json",
            "User-Agent": f"testllm-python/{self._get_version()}"
        }
        
        if self.models_api_key:
            headers["Authorization"] = f"Bearer {self.models_api_key}"
        
        return headers
    
    def _get_version(self) -> str:
        """Get testLLM version"""
        try:
            from .__version__ import __version__
            return __version__
        except ImportError:
            return "unknown"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "dashboard_url": self.dashboard_url,
            "api_base_url": self.api_base_url,
            "api_key": "***" if self.api_key else None,
            "user_id": self.user_id,
            "telemetry_enabled": self.telemetry_enabled,
            "cloud_reporting_enabled": self.cloud_reporting_enabled,
            "custom_models_enabled": self.custom_models_enabled,
            "models_api_url": self.models_api_url,
            "models_api_key": "***" if self.models_api_key else None,
            "timeout": self.timeout,
            "max_retries": self.max_retries,
            "retry_delay": self.retry_delay,
            "collect_inputs": self.collect_inputs,
            "collect_outputs": self.collect_outputs,
            "collect_metadata": self.collect_metadata,
            "local_storage_enabled": self.local_storage_enabled,
            "local_storage_path": self.local_storage_path,
            "batch_size": self.batch_size,
            "flush_interval": self.flush_interval,
            "debug_mode": self.debug_mode,
            "is_cloud_enabled": self.is_cloud_enabled,
            "is_authenticated": self.is_authenticated
        }


# Global configuration instance
_config: Optional[CloudConfig] = None


def get_config() -> CloudConfig:
    """Get the global configuration instance"""
    global _config
    if _config is None:
        _config = CloudConfig()
    return _config


def set_config(config: CloudConfig) -> None:
    """Set the global configuration instance"""
    global _config
    _config = config


def reset_config() -> None:
    """Reset the global configuration instance"""
    global _config
    _config = None


# Convenience functions
def is_cloud_enabled() -> bool:
    """Check if cloud services are enabled"""
    return get_config().is_cloud_enabled


def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return get_config().is_authenticated


def get_api_key() -> Optional[str]:
    """Get the API key"""
    return get_config().api_key


def set_api_key(api_key: str) -> None:
    """Set the API key"""
    config = get_config()
    config.api_key = api_key
    # Also set as environment variable for persistence
    os.environ["TESTLLM_API_KEY"] = api_key


def get_dashboard_url() -> str:
    """Get the dashboard URL"""
    return get_config().dashboard_url


def get_api_base_url() -> str:
    """Get the API base URL"""
    return get_config().api_base_url


def enable_telemetry() -> None:
    """Enable telemetry"""
    config = get_config()
    config.telemetry_enabled = True


def disable_telemetry() -> None:
    """Disable telemetry"""
    config = get_config()
    config.telemetry_enabled = False


def enable_debug_mode() -> None:
    """Enable debug mode"""
    config = get_config()
    config.debug_mode = True


def disable_debug_mode() -> None:
    """Disable debug mode"""
    config = get_config()
    config.debug_mode = False