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
    
class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    user_prompt: str
    session_id: str
    document_ids: Optional[List[UUID]] = None  # UUIDs from uploaded_documents
    mode: Optional[str] = "chat"

class ChatResponse(BaseModel):
    """Response model for chat endpoint."""
    response: str

class SessionRequest(BaseModel):
    """Request model for session operations."""
    session_id: str
    # Removed user_id from request since we get it from auth

class SessionHistoryResponse(BaseModel):
    """Response model for session history."""
    session_id: str
    user_id: str
    messages: List[Dict[str, Any]]
    title: Optional[str]  
    
class NewSessionResponse(BaseModel):
    """Response model for new session creation."""
    session_id: str
    message: str
    
class SessionListRequest(BaseModel):
    limit: int = Field(default=20, ge=1, le=100, description="Number of sessions to return (1-100)")
    offset: int = Field(default=0, ge=0, description="Number of sessions to skip for pagination")
    order_by: str = Field(default="updated_at", description="Sort field (updated_at or created_at)")
    order_direction: str = Field(default="DESC", description="Sort direction (ASC or DESC)")

class SessionSummary(BaseModel):
    session_id: str
    title: Optional[str]
    created_at: datetime
    updated_at: datetime
    message_count: int

class SessionListResponse(BaseModel):
    sessions: List[SessionSummary]
    total_count: int
    has_more: bool
    limit: int
    offset: int