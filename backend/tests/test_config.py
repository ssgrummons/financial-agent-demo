# backend/tests/test_config.py
import pytest
import tempfile
from pathlib import Path
from unittest.mock import patch
import yaml

from src.config.settings import Settings, get_settings, reload_settings

class TestSettings:
    """Test configuration loading and validation."""
    
    def test_default_settings(self):
        """Test that default settings load without files."""
        with patch('src.config.settings.Path.exists', return_value=False):
            settings = Settings()
            
            assert settings.app_name == "GAgent Financial Advisor"
            assert settings.assistant_config.provider == "google"  # default
            assert settings.assistant_config.model == "gemini-1.5-flash"  # default
            assert "GAgent" in settings.system_prompt
    
    def test_yaml_config_loading(self):
        """Test loading configuration from YAML file."""
        yaml_content = {
            'assistant_config': {
                'provider': 'google',
                'model': 'gemini-1.5-pro',
                'verbose': True,
                'logprobs': True,
                'max_tokens': 30000
            },
            'processing_config': {
                'detail_level': 'high'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = f.name
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.md', delete=False) as f:
            f.write("Test system prompt")
            prompt_path = f.name
        
        try:
            settings = Settings(
                config_file_path=yaml_path,
                system_prompt_path=prompt_path
            )
            
            assert settings.assistant_config.provider == "google"
            assert settings.assistant_config.model == "gemini-1.5-pro"
            assert settings.assistant_config.verbose is True
            assert settings.assistant_config.max_tokens == 30000
            assert settings.processing_config.detail_level == "high"
            assert settings.system_prompt == "Test system prompt"
            
        finally:
            Path(yaml_path).unlink()
            Path(prompt_path).unlink()
    
    def test_environment_override(self):
        """Test that environment variables work."""
        with patch.dict('os.environ', {
            'GOOGLE_API_KEY': 'test-key-123',
            'DEBUG': 'true',
            'LOG_LEVEL': 'DEBUG'
        }):
            settings = Settings()
            
            assert settings.google_api_key == 'test-key-123'
            assert settings.debug is True
            assert settings.log_level == 'DEBUG'
    
    def test_api_key_validation(self):
        """Test API key validation."""
        yaml_content = {
            'assistant_config': {
                'provider': 'google',
                'model': 'gemini-1.5-pro'
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = f.name
        
        try:
            settings = Settings(config_file_path=yaml_path)
            
            # Should raise error for missing Google API key
            with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
                settings.validate_api_keys()
            
            # Should pass with API key
            settings.google_api_key = "test-key"
            settings.validate_api_keys()  # Should not raise
            
        finally:
            Path(yaml_path).unlink()
    
    def test_model_config_extraction(self):
        """Test extracting model configuration."""
        yaml_content = {
            'assistant_config': {
                'provider': 'google',
                'model': 'gemini-1.5-pro',
                'temperature': 0.2,
                'max_tokens': 8192
            }
        }
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False) as f:
            yaml.dump(yaml_content, f)
            yaml_path = f.name
        
        try:
            settings = Settings(config_file_path=yaml_path, google_api_key="test")
            config = settings.get_model_config()
            
            assert config['provider'] == 'google'
            assert config['model'] == 'gemini-1.5-pro'
            assert config['temperature'] == 0.2
            assert config['max_tokens'] == 8192
            
            api_key = settings.get_api_key()
            assert api_key == "test"
            
        finally:
            Path(yaml_path).unlink()
    
    def test_cors_origins_parsing(self):
        """Test CORS origins parsing from string."""
        settings = Settings(cors_origins="http://localhost:3000,http://localhost:8501, http://example.com")
        
        expected = ["http://localhost:3000", "http://localhost:8501", "http://example.com"]
        assert settings.cors_origins == expected

if __name__ == "__main__":
    # Quick manual test
    try:
        settings = get_settings()
        print(f"✅ Settings loaded successfully!")
        print(f"   Provider: {settings.assistant_config.provider}")
        print(f"   Model: {settings.assistant_config.model}")
        print(f"   System prompt length: {len(settings.system_prompt)} chars")
        print(f"   Debug: {settings.debug}")
    except Exception as e:
        print(f"❌ Settings loading failed: {e}")