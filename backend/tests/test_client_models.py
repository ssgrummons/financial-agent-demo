# backend/tests/clients/test_models.py
import pytest
from unittest.mock import patch, MagicMock
from src.clients.models import AWSBedrockModelFactory, AWSBedrockSettings


class TestAWSBedrockSettings:
    """Test suite for AWSBedrockSettings"""

    def test_default_settings(self):
        """Test default settings values"""
        settings = AWSBedrockSettings()
        
        assert settings.AWS_REGION == "us-east-1"
        assert settings.MAX_TOKENS == 4000
        assert settings.TEMPERATURE == 0.7
        assert settings.TOP_P is None
        assert settings.TOP_K is None
        assert settings.BEDROCK_MODEL_ID is None

    def test_settings_from_env(self, monkeypatch):
        """Test that settings load from environment variables"""
        monkeypatch.setenv("AWS_REGION", "us-west-2")
        monkeypatch.setenv("BEDROCK_MODEL_ID", "openai.gpt-oss-20b-v1:0")
        monkeypatch.setenv("MAX_TOKENS", "2000")
        monkeypatch.setenv("TEMPERATURE", "0.5")
        monkeypatch.setenv("TOP_P", "0.9")
        monkeypatch.setenv("TOP_K", "50")
        
        settings = AWSBedrockSettings()
        
        assert settings.AWS_REGION == "us-west-2"
        assert settings.BEDROCK_MODEL_ID == "openai.gpt-oss-20b-v1:0"
        assert settings.MAX_TOKENS == 2000
        assert settings.TEMPERATURE == 0.5
        assert settings.TOP_P == 0.9
        assert settings.TOP_K == 50


class TestAWSBedrockModelFactory:
    """Test suite for AWSBedrockModelFactory"""

    @pytest.fixture
    def factory(self):
        """Create a factory instance with test settings"""
        settings = AWSBedrockSettings(
            BEDROCK_MODEL_ID="openai.gpt-oss-20b-v1:0",
            AWS_REGION="us-east-1",
            MAX_TOKENS=4000,
            TEMPERATURE=0.7
        )
        return AWSBedrockModelFactory(settings=settings)

    def test_translate_parameters_mistral(self, factory):
        """Test parameter translation for Mistral models"""
        model_kwargs, bedrock_kwargs = factory._translate_parameters(
            model_id="mistral.mistral-large-2407-v1:0",
            max_tokens=2000,
            temperature=0.5,
            top_p=0.9,
            top_k=50,
            streaming=True,
            verbose=True,
            logprobs=False,
            reasoning_effort="medium"
        )
        
        # Check model_kwargs (goes to Bedrock API)
        assert model_kwargs["max_tokens"] == 2000
        assert model_kwargs["temperature"] == 0.5
        assert model_kwargs["top_p"] == 0.9
        assert model_kwargs["top_k"] == 50
        assert "logprobs" not in model_kwargs  # Not supported
        assert "reasoning_effort" not in model_kwargs  # Not supported
        
        # Check bedrock_kwargs (goes to ChatBedrock constructor)
        assert bedrock_kwargs["streaming"] is True
        assert bedrock_kwargs["verbose"] is True

    def test_translate_parameters_llama(self, factory):
        """Test parameter translation for Llama models"""
        model_kwargs, bedrock_kwargs = factory._translate_parameters(
            model_id="meta.llama3-1-405b-instruct-v1:0",
            max_tokens=2000,
            temperature=0.1,
            top_p=0.9,
            top_k=50,  # Should be ignored
            streaming=False,
            verbose=True
        )
        
        # Check model_kwargs
        assert model_kwargs["max_gen_len"] == 2000  # Translated
        assert model_kwargs["temperature"] == 0.1
        assert model_kwargs["top_p"] == 0.9
        assert "top_k" not in model_kwargs  # Not supported by Llama
        
        # Check bedrock_kwargs
        assert bedrock_kwargs["streaming"] is False
        assert bedrock_kwargs["verbose"] is True

    def test_translate_parameters_openai(self, factory):
        """Test parameter translation for OpenAI models"""
        model_kwargs, bedrock_kwargs = factory._translate_parameters(
            model_id="openai.gpt-oss-20b-v1:0",
            max_tokens=2000,
            temperature=0.3,
            top_p=0.9,
            top_k=50,  # Should be ignored
            streaming=True,
            verbose=True,
            logprobs=True,  # Should be included
            reasoning_effort="high"  # Should be included
        )
        
        # Check model_kwargs
        assert model_kwargs["max_completion_tokens"] == 2000  # Translated
        assert model_kwargs["temperature"] == 0.3
        assert model_kwargs["top_p"] == 0.9
        assert "top_k" not in model_kwargs  # Not supported
        assert model_kwargs["logprobs"] is True  # Supported!
        assert model_kwargs["reasoning_effort"] == "high"  # Supported!
        
        # Check bedrock_kwargs
        assert bedrock_kwargs["streaming"] is True
        assert bedrock_kwargs["verbose"] is True

    def test_translate_parameters_amazon_nova(self, factory):
        """Test parameter translation for Amazon Nova models"""
        model_kwargs, bedrock_kwargs = factory._translate_parameters(
            model_id="amazon.nova-pro-v1:0",
            max_tokens=3000,
            temperature=0.5,
            top_p=0.95,
            top_k=40,  # Should be ignored
            streaming=True,
            verbose=False
        )
        
        # Check model_kwargs (camelCase for Nova)
        assert model_kwargs["maxTokens"] == 3000  # camelCase
        assert model_kwargs["temperature"] == 0.5
        assert model_kwargs["topP"] == 0.95  # camelCase
        assert "top_k" not in model_kwargs  # Not supported
        
        # Check bedrock_kwargs
        assert bedrock_kwargs["streaming"] is True
        assert bedrock_kwargs["verbose"] is False

    def test_translate_parameters_none_values_filtered(self, factory):
        """Test that None values are filtered out"""
        model_kwargs, bedrock_kwargs = factory._translate_parameters(
            model_id="mistral.mistral-large-2407-v1:0",
            max_tokens=2000,
            temperature=0.5,
            top_p=None,  # Should be excluded
            top_k=None,  # Should be excluded
            streaming=True,
            verbose=True,
            logprobs=None,  # Should be excluded
            reasoning_effort=None  # Should be excluded
        )
        
        assert model_kwargs["max_tokens"] == 2000
        assert model_kwargs["temperature"] == 0.5
        assert "top_p" not in model_kwargs
        assert "top_k" not in model_kwargs
        assert "logprobs" not in model_kwargs
        assert "reasoning_effort" not in model_kwargs

    def test_translate_parameters_uses_settings_defaults(self, factory):
        """Test that missing parameters fall back to settings defaults"""
        model_kwargs, bedrock_kwargs = factory._translate_parameters(
            model_id="openai.gpt-oss-20b-v1:0",
            # Not passing max_tokens or temperature
            streaming=True,
            verbose=True
        )
        
        # Should use values from factory.settings
        assert model_kwargs["max_completion_tokens"] == 4000  # From settings.MAX_TOKENS
        assert model_kwargs["temperature"] == 0.7  # From settings.TEMPERATURE

    @patch('src.clients.models.ChatBedrock')
    def test_create_model_success(self, mock_chatbedrock, factory):
        """Test successful model creation"""
        mock_model = MagicMock()
        mock_chatbedrock.return_value = mock_model
        
        model = factory.create_model(
            model_name="openai.gpt-oss-20b-v1:0",
            max_tokens=2000,
            temperature=0.3,
            streaming=True,
            verbose=True
        )
        
        # Verify ChatBedrock was called
        assert mock_chatbedrock.called
        call_kwargs = mock_chatbedrock.call_args.kwargs
        
        assert call_kwargs["model_id"] == "openai.gpt-oss-20b-v1:0"
        assert call_kwargs["region_name"] == "us-east-1"
        assert call_kwargs["streaming"] is True
        assert call_kwargs["verbose"] is True
        assert "model_kwargs" in call_kwargs
        
        # Check model_kwargs
        model_kwargs = call_kwargs["model_kwargs"]
        assert model_kwargs["max_completion_tokens"] == 2000
        assert model_kwargs["temperature"] == 0.3

    @patch('src.clients.models.ChatBedrock')
    def test_create_model_with_credentials(self, mock_chatbedrock):
        """Test model creation with AWS credentials"""
        settings = AWSBedrockSettings(
            BEDROCK_MODEL_ID="openai.gpt-oss-20b-v1:0",
            AWS_ACCESS_KEY_ID="test-key-id",
            AWS_SECRET_ACCESS_KEY="test-secret-key"
        )
        factory = AWSBedrockModelFactory(settings=settings)
        
        mock_model = MagicMock()
        mock_chatbedrock.return_value = mock_model
        
        model = factory.create_model()
        
        # Verify credentials were passed
        call_kwargs = mock_chatbedrock.call_args.kwargs
        assert call_kwargs["aws_access_key_id"] == "test-key-id"
        assert call_kwargs["aws_secret_access_key"] == "test-secret-key"

    def test_create_model_no_model_id_raises(self):
        """Test that missing model ID raises ValueError"""
        factory = AWSBedrockModelFactory(settings=AWSBedrockSettings())
        
        with pytest.raises(ValueError, match="BEDROCK_MODEL_ID must be set"):
            factory.create_model()

    @patch('src.clients.models.ChatBedrock')
    def test_create_model_uses_default_from_settings(self, mock_chatbedrock, factory):
        """Test that create_model uses model from settings when not specified"""
        mock_model = MagicMock()
        mock_chatbedrock.return_value = mock_model
        
        model = factory.create_model()  # No model_name specified
        
        call_kwargs = mock_chatbedrock.call_args.kwargs
        assert call_kwargs["model_id"] == "openai.gpt-oss-20b-v1:0"  # From settings

    @pytest.mark.parametrize("model_id,expected_max_tokens_param", [
        ("mistral.mistral-large-2407-v1:0", "max_tokens"),
        ("meta.llama3-1-405b-instruct-v1:0", "max_gen_len"),
        ("openai.gpt-oss-20b-v1:0", "max_completion_tokens"),
        ("amazon.nova-pro-v1:0", "maxTokens"),
        ("amazon.titan-text-express-v1", "maxTokenCount"),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", "max_tokens"),
        ("cohere.command-r-plus-v1:0", "max_tokens"),
    ])
    @patch('src.clients.models.ChatBedrock')
    def test_max_tokens_translation_for_all_models(self, mock_chatbedrock, model_id, expected_max_tokens_param):
        """Test that max_tokens is correctly translated for all model families"""
        factory = AWSBedrockModelFactory(settings=AWSBedrockSettings(
            BEDROCK_MODEL_ID=model_id
        ))
        
        mock_model = MagicMock()
        mock_chatbedrock.return_value = mock_model
        
        factory.create_model(max_tokens=2000)
        
        call_kwargs = mock_chatbedrock.call_args.kwargs
        model_kwargs = call_kwargs["model_kwargs"]
        
        assert expected_max_tokens_param in model_kwargs
        assert model_kwargs[expected_max_tokens_param] == 2000


