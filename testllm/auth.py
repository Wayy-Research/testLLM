"""
Authentication module for testLLM cloud services
"""

import os
import json
import time
import requests
from typing import Optional, Dict, Any, Tuple
from dataclasses import dataclass, asdict
from pathlib import Path

from .config import get_config, CloudConfig


@dataclass
class UserInfo:
    """User information from authentication"""
    user_id: str
    email: str
    subscription_tier: str
    subscription_status: str
    api_key: str
    created_at: str
    team_id: Optional[str] = None
    team_name: Optional[str] = None
    team_role: Optional[str] = None


class AuthenticationError(Exception):
    """Authentication related errors"""
    pass


class AuthenticationManager:
    """Manages authentication with testLLM cloud services"""
    
    def __init__(self, config: Optional[CloudConfig] = None):
        self.config = config or get_config()
        self.user_info: Optional[UserInfo] = None
        self._auth_cache_file = Path(self.config.local_storage_path) / "auth_cache.json"
        self._ensure_cache_dir()
    
    def _ensure_cache_dir(self):
        """Ensure the cache directory exists"""
        self._auth_cache_file.parent.mkdir(parents=True, exist_ok=True)
    
    def authenticate(self, api_key: str) -> UserInfo:
        """
        Authenticate with the testLLM cloud service
        
        Args:
            api_key: The API key from the dashboard
            
        Returns:
            UserInfo object with user details
            
        Raises:
            AuthenticationError: If authentication fails
        """
        if not api_key:
            raise AuthenticationError("API key is required")
        
        try:
            # Make authentication request to Vercel API
            response = requests.post(
                f"{self.config.api_base_url}/auth/verify",
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json"
                },
                json={"api_key": api_key},
                timeout=self.config.timeout
            )
            
            if response.status_code == 200:
                user_data = response.json()
                self.user_info = UserInfo(**user_data)
                
                # Cache the authentication
                self._cache_auth(self.user_info)
                
                # Update config with authenticated user info
                self.config.api_key = api_key
                self.config.user_id = self.user_info.user_id
                
                return self.user_info
                
            elif response.status_code == 401:
                raise AuthenticationError("Invalid API key")
            elif response.status_code == 403:
                raise AuthenticationError("API key is disabled or expired")
            else:
                raise AuthenticationError(f"Authentication failed: {response.status_code}")
                
        except requests.RequestException as e:
            raise AuthenticationError(f"Failed to connect to authentication service: {e}")
    
    def is_authenticated(self) -> bool:
        """Check if user is currently authenticated"""
        return self.user_info is not None and self.config.api_key is not None
    
    def get_user_info(self) -> Optional[UserInfo]:
        """Get current user information"""
        if self.user_info is None and self.config.api_key:
            # Try to load from cache
            cached_info = self._load_cached_auth()
            if cached_info:
                self.user_info = cached_info
        
        return self.user_info
    
    def refresh_auth(self) -> Optional[UserInfo]:
        """Refresh authentication information"""
        if not self.config.api_key:
            return None
        
        try:
            return self.authenticate(self.config.api_key)
        except AuthenticationError:
            self.logout()
            return None
    
    def logout(self):
        """Logout and clear authentication"""
        self.user_info = None
        self.config.api_key = None
        self.config.user_id = None
        self._clear_auth_cache()
    
    def check_subscription_status(self) -> Tuple[str, bool]:
        """
        Check current subscription status
        
        Returns:
            Tuple of (subscription_tier, is_active)
        """
        if not self.is_authenticated():
            return "free", False
        
        user_info = self.get_user_info()
        if not user_info:
            return "free", False
        
        is_active = user_info.subscription_status == "active"
        return user_info.subscription_tier, is_active
    
    def can_use_feature(self, feature: str) -> bool:
        """
        Check if user can use a specific feature
        
        Args:
            feature: Feature name ('cloud_storage', 'advanced_analytics', 'team_features', etc.)
            
        Returns:
            True if user can use the feature
        """
        tier, is_active = self.check_subscription_status()
        
        if not is_active and tier != "free":
            return False
        
        feature_map = {
            "cloud_storage": ["free", "pro", "team"],
            "advanced_analytics": ["pro", "team"],
            "data_export": ["pro", "team"],
            "team_features": ["team"],
            "priority_support": ["pro", "team"],
            "custom_models": ["team"],
            "unlimited_history": ["pro", "team"]
        }
        
        return tier in feature_map.get(feature, [])
    
    def get_usage_limits(self) -> Dict[str, Any]:
        """Get usage limits for current subscription"""
        tier, is_active = self.check_subscription_status()
        
        if tier == "free":
            return {
                "test_history_days": 7,
                "max_sessions_per_day": 50,
                "max_team_members": 1,
                "export_formats": ["json"],
                "support_level": "community"
            }
        elif tier == "pro" and is_active:
            return {
                "test_history_days": -1,  # unlimited
                "max_sessions_per_day": 500,
                "max_team_members": 1,
                "export_formats": ["json", "csv", "xlsx"],
                "support_level": "priority"
            }
        elif tier == "team" and is_active:
            return {
                "test_history_days": -1,  # unlimited
                "max_sessions_per_day": 2000,
                "max_team_members": 10,
                "export_formats": ["json", "csv", "xlsx", "pdf"],
                "support_level": "dedicated"
            }
        else:
            # Fallback to free tier if subscription is not active
            return self.get_usage_limits()
    
    def _cache_auth(self, user_info: UserInfo):
        """Cache authentication information locally"""
        if not self.config.local_storage_enabled:
            return
        
        try:
            cache_data = {
                "user_info": asdict(user_info),
                "cached_at": time.time(),
                "api_key": self.config.api_key
            }
            
            with open(self._auth_cache_file, 'w') as f:
                json.dump(cache_data, f, indent=2)
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"Warning: Failed to cache auth info: {e}")
    
    def _load_cached_auth(self) -> Optional[UserInfo]:
        """Load cached authentication information"""
        if not self.config.local_storage_enabled:
            return None
        
        try:
            if not self._auth_cache_file.exists():
                return None
            
            with open(self._auth_cache_file, 'r') as f:
                cache_data = json.load(f)
            
            # Check if cache is still valid (24 hours)
            cached_at = cache_data.get("cached_at", 0)
            if time.time() - cached_at > 86400:  # 24 hours
                return None
            
            # Verify API key matches
            if cache_data.get("api_key") != self.config.api_key:
                return None
            
            user_data = cache_data.get("user_info")
            if user_data:
                return UserInfo(**user_data)
                
        except Exception as e:
            if self.config.debug_mode:
                print(f"Warning: Failed to load cached auth info: {e}")
            
        return None
    
    def _clear_auth_cache(self):
        """Clear cached authentication information"""
        try:
            if self._auth_cache_file.exists():
                self._auth_cache_file.unlink()
        except Exception as e:
            if self.config.debug_mode:
                print(f"Warning: Failed to clear auth cache: {e}")


# Global authentication manager instance
_auth_manager: Optional[AuthenticationManager] = None


def get_auth_manager() -> AuthenticationManager:
    """Get the global authentication manager instance"""
    global _auth_manager
    if _auth_manager is None:
        _auth_manager = AuthenticationManager()
    return _auth_manager


def set_auth_manager(auth_manager: AuthenticationManager) -> None:
    """Set the global authentication manager instance"""
    global _auth_manager
    _auth_manager = auth_manager


def reset_auth_manager() -> None:
    """Reset the global authentication manager instance"""
    global _auth_manager
    _auth_manager = None


# Convenience functions
def authenticate(api_key: str) -> UserInfo:
    """Authenticate with API key"""
    return get_auth_manager().authenticate(api_key)


def is_authenticated() -> bool:
    """Check if user is authenticated"""
    return get_auth_manager().is_authenticated()


def get_user_info() -> Optional[UserInfo]:
    """Get current user information"""
    return get_auth_manager().get_user_info()


def logout() -> None:
    """Logout current user"""
    get_auth_manager().logout()


def check_subscription_status() -> Tuple[str, bool]:
    """Check current subscription status"""
    return get_auth_manager().check_subscription_status()


def can_use_feature(feature: str) -> bool:
    """Check if user can use a specific feature"""
    return get_auth_manager().can_use_feature(feature)


def get_usage_limits() -> Dict[str, Any]:
    """Get usage limits for current subscription"""
    return get_auth_manager().get_usage_limits()


def refresh_auth() -> Optional[UserInfo]:
    """Refresh authentication"""
    return get_auth_manager().refresh_auth()