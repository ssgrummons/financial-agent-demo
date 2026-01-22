# backend/tests/clients/test_capability_map.py
import pytest
from src.clients.capability_map import BedrockModelParameters


class TestBedrockModelParameters:
    """Test suite for BedrockModelParameters parameter translation"""

    # Test model prefix extraction
    @pytest.mark.parametrize("model_id,expected_prefix", [
        ("mistral.mistral-large-2407-v1:0", "mistral"),
        ("mistral.pixtral-12b-2409-v1:0", "mistral"),
        ("meta.llama3-1-405b-instruct-v1:0", "meta.llama"),
        ("meta.llama3-2-90b-instruct-v1:0", "meta.llama"),
        ("anthropic.claude-3-5-sonnet-20241022-v2:0", "anthropic.claude"),
        ("openai.gpt-oss-20b-v1:0", "openai"),
        ("openai.o1-preview-v1:0", "openai"),
        ("amazon.nova-pro-v1:0", "amazon.nova"),
        ("amazon.nova-lite-v1:0", "amazon.nova"),
        ("amazon.titan-text-express-v1", "amazon.titan"),
        ("ai21.jamba-instruct-v1:0", "ai21.jamba"),
        ("google.gemma-7b-it-v1:0", "google.gemma"),
        ("cohere.command-r-plus-v1:0", "cohere.command"),
        ("unknown.model-v1:0", "generic"),
    ])
    def test_get_model_prefix(self, model_id, expected_prefix):
        """Test that model prefix extraction works correctly"""
        prefix = BedrockModelParameters.get_model_prefix(model_id)
        assert prefix == expected_prefix

    # Test parameter config retrieval
    def test_get_parameter_config_mistral(self):
        """Test parameter config for Mistral models"""
        config = BedrockModelParameters.get_parameter_config("mistral.mistral-large-2407-v1:0")
        
        assert config["max_tokens"] == "max_tokens"
        assert config["temperature"] == "temperature"
        assert config["top_p"] == "top_p"
        assert config["top_k"] == "top_k"
        assert config["streaming"] == "streaming"
        assert config["verbose"] == "verbose"
        assert config["logprobs"] is None
        assert config["reasoning_effort"] is None

    def test_get_parameter_config_llama(self):
        """Test parameter config for Llama models"""
        config = BedrockModelParameters.get_parameter_config("meta.llama3-1-405b-instruct-v1:0")
        
        assert config["max_tokens"] == "max_gen_len"
        assert config["temperature"] == "temperature"
        assert config["top_p"] == "top_p"
        assert config["top_k"] is None  # Llama doesn't support top_k
        assert config["streaming"] == "streaming"
        assert config["verbose"] == "verbose"
        assert config["logprobs"] is None
        assert config["reasoning_effort"] is None

    def test_get_parameter_config_openai(self):
        """Test parameter config for OpenAI models"""
        config = BedrockModelParameters.get_parameter_config("openai.gpt-oss-20b-v1:0")
        
        assert config["max_tokens"] == "max_completion_tokens"
        assert config["temperature"] == "temperature"
        assert config["top_p"] == "top_p"
        assert config["top_k"] is None  # OpenAI doesn't support top_k
        assert config["streaming"] == "streaming"
        assert config["verbose"] == "verbose"
        assert config["logprobs"] == "logprobs"  # OpenAI supports logprobs
        assert config["reasoning_effort"] == "reasoning_effort"  # For o1 models

    def test_get_parameter_config_anthropic(self):
        """Test parameter config for Anthropic Claude models"""
        config = BedrockModelParameters.get_parameter_config("anthropic.claude-3-5-sonnet-20241022-v2:0")
        
        assert config["max_tokens"] == "max_tokens"
        assert config["temperature"] == "temperature"
        assert config["top_p"] == "top_p"
        assert config["top_k"] == "top_k"
        assert config["streaming"] == "streaming"
        assert config["verbose"] == "verbose"
        assert config["logprobs"] is None
        assert config["reasoning_effort"] is None

    def test_get_parameter_config_amazon_nova(self):
        """Test parameter config for Amazon Nova models"""
        config = BedrockModelParameters.get_parameter_config("amazon.nova-pro-v1:0")
        
        assert config["max_tokens"] == "maxTokens"  # camelCase
        assert config["temperature"] == "temperature"
        assert config["top_p"] == "topP"  # camelCase
        assert config["top_k"] is None
        assert config["streaming"] == "streaming"
        assert config["verbose"] == "verbose"
        assert config["logprobs"] is None
        assert config["reasoning_effort"] is None

    def test_get_parameter_config_amazon_titan(self):
        """Test parameter config for Amazon Titan models"""
        config = BedrockModelParameters.get_parameter_config("amazon.titan-text-express-v1")
        
        assert config["max_tokens"] == "maxTokenCount"
        assert config["temperature"] == "temperature"
        assert config["top_p"] == "topP"
        assert config["top_k"] is None
        assert config["streaming"] == "streaming"
        assert config["verbose"] == "verbose"
        assert config["logprobs"] is None
        assert config["reasoning_effort"] is None

    def test_get_parameter_config_cohere(self):
        """Test parameter config for Cohere models"""
        config = BedrockModelParameters.get_parameter_config("cohere.command-r-plus-v1:0")
        
        assert config["max_tokens"] == "max_tokens"
        assert config["temperature"] == "temperature"
        assert config["top_p"] == "p"  # Cohere uses 'p'
        assert config["top_k"] == "k"  # Cohere uses 'k'
        assert config["streaming"] == "streaming"
        assert config["verbose"] == "verbose"
        assert config["logprobs"] is None
        assert config["reasoning_effort"] is None

    def test_get_parameter_config_unknown_model(self):
        """Test that unknown models get safe defaults"""
        config = BedrockModelParameters.get_parameter_config("unknown.provider-model-v1:0")
        
        # Should return generic defaults
        assert config["max_tokens"] == "max_tokens"
        assert config["temperature"] == "temperature"
        assert config["top_p"] == "top_p"
        assert config["top_k"] == "top_k"
        assert config["streaming"] == "streaming"
        assert config["verbose"] == "verbose"
        assert config["logprobs"] is None
        assert config["reasoning_effort"] is None

    def test_all_configs_have_required_keys(self):
        """Test that all model configs have the required parameter keys"""
        required_keys = {
            "max_tokens", "temperature", "top_p", "top_k",
            "streaming", "verbose", "logprobs", "reasoning_effort"
        }
        
        for model_family, config in BedrockModelParameters.MODEL_CONFIGS.items():
            assert set(config.keys()) == required_keys, \
                f"Model family '{model_family}' is missing required keys"