# backend/config/settings.py
import yaml
from pathlib import Path
from functools import lru_cache
from typing import Optional, Dict, Any
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings

class AssistantConfig(BaseSettings):
    """Assistant configuration from YAML."""
    provider: str = Field(..., description="AI model provider")
    model: str = Field(..., description="AI model name")
    verbose: bool = Field(default=False, description="Enable verbose logging")
    logprobs: bool = Field(default=False, description="Enable log probabilities")
    reasoning_effort: Optional[str] = Field(default=None, description="Reasoning effort level")
    max_tokens: int = Field(default=4096, description="Maximum tokens for model response")
    temperature: float = Field(default=0.1, description="Model temperature")

class ProcessingConfig(BaseSettings):
    """Processing configuration from YAML."""
    detail_level: str = Field(default="medium", description="Processing detail level")

class Settings(BaseSettings):
    """Application settings combining environment variables and YAML config."""
    
    # ===== ENVIRONMENT VARIABLES =====
    # Secrets - never commit these
    google_api_key: Optional[str] = Field(default=None, description="Google API key")
    openai_api_key: Optional[str] = Field(default=None, description="OpenAI API key")
    anthropic_api_key: Optional[str] = Field(default=None, description="Anthropic API key")
    alpha_vantage_api_key: Optional[str] = Field(default=None, description="Alpha Vantage API key")
    finnhub_api_key: Optional[str] = Field(default=None, description="Finnhub API key")
    
    # Deployment/Infrastructure settings
    app_name: str = Field(default="GAgent Financial Advisor", description="Application name")
    app_version: str = Field(default="1.0.0", description="Application version")
    debug: bool = Field(default=False, description="Debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Server settings
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    reload: bool = Field(default=False, description="Auto-reload on changes")
    
    # CORS settings
    cors_origins: str = Field(
        default="http://localhost:8501,http://frontend:8501",
        description="Comma-separated list of allowed CORS origins"
    )
    
    # Database settings (for future use)
    database_url: Optional[str] = Field(default=None, description="Database connection URL")
    
    # Rate limiting
    rate_limit_enabled: bool = Field(default=False, description="Enable rate limiting")
    rate_limit_requests_per_minute: int = Field(default=60, description="Rate limit per minute")
    
    # Session management
    session_timeout_minutes: int = Field(default=60, description="Session timeout in minutes")
    max_sessions_per_user: int = Field(default=10, description="Maximum sessions per user")
    
    # ===== YAML LOADED CONFIGS =====
    # These are loaded from config.yaml and should not be in environment
    assistant_config: AssistantConfig = Field(default_factory=AssistantConfig)
    processing_config: ProcessingConfig = Field(default_factory=ProcessingConfig)
    
    # System prompt (loaded from file)
    system_prompt: str = Field(default="", description="System prompt for the assistant")
    
    # Config file paths
    config_file_path: str = Field(default="config/config.yaml", description="Path to YAML config file")
    system_prompt_path: str = Field(default="config/prompts/system_prompt.md", description="Path to system prompt file")
    
    def __init__(self, **kwargs):
        # Load YAML config before initializing
        config_data = self._load_yaml_config()
        
        # Load system prompt
        system_prompt = self._load_system_prompt()
        
        # Merge YAML data with kwargs
        merged_data = {**config_data, "system_prompt": system_prompt, **kwargs}
        
        super().__init__(**merged_data)
    
    def _load_yaml_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        config_path = Path(self.config_file_path if hasattr(self, 'config_file_path') else "config/config.yaml")
        
        if not config_path.exists():
            # Return default config if file doesn't exist
            return {
                "assistant_config": AssistantConfig(),
                "processing_config": ProcessingConfig()
            }
        
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                yaml_data = yaml.safe_load(f) or {}
            
            # Parse nested configs
            assistant_data = yaml_data.get('assistant_config', {})
            processing_data = yaml_data.get('processing_config', {})
            
            return {
                "assistant_config": AssistantConfig(**assistant_data),
                "processing_config": ProcessingConfig(**processing_data)
            }
            
        except Exception as e:
            raise ValueError(f"Failed to load YAML config from {config_path}: {e}")
    
    def _load_system_prompt(self) -> str:
        """Load system prompt from markdown file."""
        prompt_path = Path(self.system_prompt_path if hasattr(self, 'system_prompt_path') else "config/prompts/system_prompt.md")
        
        if not prompt_path.exists():
            return "You are GAgent, a helpful financial advisor AI assistant."
        
        try:
            with open(prompt_path, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except Exception as e:
            raise ValueError(f"Failed to load system prompt from {prompt_path}: {e}")
    
    @field_validator('cors_origins', mode='before')
    @classmethod
    def parse_cors_origins(cls, v):
        """Parse CORS origins from comma-separated string to list."""
        print(f"DEBUG: cors_origins received: {v} (type: {type(v)})")
        
        if isinstance(v, str):
            result = [origin.strip() for origin in v.split(',') if origin.strip()]
            print(f"DEBUG: Parsed string to list: {result}")
            return result
        elif isinstance(v, list):
            result = [str(origin).strip() for origin in v if str(origin).strip()]
            print(f"DEBUG: Processed existing list: {result}")
            return result
        
        # Fallback
        return [str(v)] if v else []
    
    def validate_api_keys(self) -> None:
        """Validate that required API keys are present based on configuration."""
        provider = self.assistant_config.provider.lower()
        
        if provider == "google" and not self.google_api_key:
            raise ValueError("GOOGLE_API_KEY environment variable is required for Google models")
        elif provider == "openai" and not self.openai_api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required for OpenAI models")
        elif provider == "anthropic" and not self.anthropic_api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required for Anthropic models")
    
    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration for client initialization."""
        return {
            "provider": self.assistant_config.provider,
            "model": self.assistant_config.model,
            "temperature": self.assistant_config.temperature,
            "max_tokens": self.assistant_config.max_tokens,
            "verbose": self.assistant_config.verbose,
            "logprobs": self.assistant_config.logprobs,
            "reasoning_effort": self.assistant_config.reasoning_effort,
        }
    
    def get_api_key(self) -> Optional[str]:
        """Get the appropriate API key based on provider."""
        provider = self.assistant_config.provider.lower()
        
        if provider == "google":
            return self.google_api_key
        elif provider == "openai":
            return self.openai_api_key
        elif provider == "anthropic":
            return self.anthropic_api_key
        else:
            return None
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        # Don't validate assignment to allow dynamic loading
        validate_assignment = False

@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    settings = Settings()
    settings.validate_api_keys()  # Validate on first load
    return settings

# Convenience function to reload settings (useful for testing)
def reload_settings() -> Settings:
    """Reload settings (clears cache)."""
    get_settings.cache_clear()
    return get_settings()