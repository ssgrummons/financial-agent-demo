from pydantic import BaseModel, Field, model_validator
from pydantic_settings import BaseSettings
from typing import Any, Dict, List, Optional, Literal, TypedDict, Annotated
from datetime import datetime
from enum import Enum
from uuid import UUID
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class Settings(BaseSettings):
    """Application settings."""
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    
    # Security Configuration
    CORS_ORIGINS: str = Field(
        default="http://localhost:8501,http://localhost:3000",
        description="Comma-separated list of allowed CORS origins"
    )

    MCP_SERVER_URL: str = Field(
        default="http://localhost:8888/mcp",
        description="MCP server URL"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True
        extra = "ignore"

    @property
    def allowed_origins(self) -> List[str]:
        """Get list of allowed CORS origins."""
        return [origin.strip() for origin in self.CORS_ORIGINS.split(",")]
    
