from typing import Optional, Dict
import logging

logger = logging.getLogger(__name__)

class BedrockModelParameters:
    """Model-specific parameter mappings and filters for Bedrock models"""
    
    # Define parameter translations and support for each model family
    MODEL_CONFIGS = {
        # Mistral models
        "mistral": {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "streaming": "streaming",
            "verbose": "verbose",
            "logprobs": None,
            "reasoning_effort": None,
        },
        # Meta Llama models
        "meta.llama": {
            "max_tokens": "max_gen_len",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": None,
            "streaming": "streaming",
            "verbose": "verbose",
            "logprobs": None,
            "reasoning_effort": None,
        },
        # Anthropic Claude
        "anthropic.claude": {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "streaming": "streaming",
            "verbose": "verbose",
            "logprobs": None,
            "reasoning_effort": None,
        },
        # OpenAI models (gpt-oss, o1, etc.)
        "openai": {
            "max_tokens": "max_completion_tokens",  # OpenAI uses this parameter name
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": None,  # OpenAI doesn't support top_k
            "streaming": "streaming",
            "verbose": "verbose",
            "logprobs": "logprobs",  # OpenAI supports logprobs!
            "reasoning_effort": "reasoning_effort",  # For o1 models
        },
        # Amazon Nova models
        "amazon.nova": {
            "max_tokens": "maxTokens",
            "temperature": "temperature",
            "top_p": "topP",
            "top_k": None,
            "streaming": "streaming",
            "verbose": "verbose",
            "logprobs": None,
            "reasoning_effort": None,
        },
        # Amazon Titan (legacy)
        "amazon.titan": {
            "max_tokens": "maxTokenCount",
            "temperature": "temperature",
            "top_p": "topP",
            "top_k": None,
            "streaming": "streaming",
            "verbose": "verbose",
            "logprobs": None,
            "reasoning_effort": None,
        },
        # AI21 Jamba
        "ai21.jamba": {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": None,
            "streaming": "streaming",
            "verbose": "verbose",
            "logprobs": None,
            "reasoning_effort": None,
        },
        # Google Gemma
        "google.gemma": {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "streaming": "streaming",
            "verbose": "verbose",
            "logprobs": None,
            "reasoning_effort": None,
        },
        # Cohere Command
        "cohere.command": {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "p",
            "top_k": "k",
            "streaming": "streaming",
            "verbose": "verbose",
            "logprobs": None,
            "reasoning_effort": None,
        },
    }
    
    @classmethod
    def get_model_prefix(cls, model_id: str) -> str:
        """Extract model family prefix from Bedrock model ID"""
        # Handle cases like "mistral.pixtral-12b-2409-v1:0"
        # or "meta.llama3-1-405b-instruct-v1:0"
        
        # First try exact match with dots (e.g., "mistral.pixtral")
        for prefix in cls.MODEL_CONFIGS.keys():
            if model_id.startswith(prefix):
                # For cases like "mistral.pixtral", check if there's a more specific match
                if "." in prefix:
                    return prefix
                # For cases like "mistral", check if there's a sub-family
                sub_parts = model_id.split("-")[0]  # Get "mistral.pixtral" from "mistral.pixtral-12b..."
                if sub_parts in cls.MODEL_CONFIGS:
                    return sub_parts
                return prefix
        
        # Fallback: extract just the provider prefix (before first dot)
        provider = model_id.split(".")[0]
        for prefix in cls.MODEL_CONFIGS.keys():
            if prefix.startswith(provider):
                return prefix
        
        logger.warning(f"Unknown model family for {model_id}, using generic defaults")
        return "generic"
    
    @classmethod
    def get_parameter_config(cls, model_id: str) -> Dict[str, Optional[str]]:
        """Get parameter mapping config for a model"""
        prefix = cls.get_model_prefix(model_id)
        
        # Return config if found, otherwise safe defaults
        return cls.MODEL_CONFIGS.get(prefix, {
            "max_tokens": "max_tokens",
            "temperature": "temperature",
            "top_p": "top_p",
            "top_k": "top_k",
            "streaming": "streaming",
            "verbose": "verbose",
            "logprobs": None,
            "reasoning_effort": None,
        })