# backend/src/models/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
from datetime import datetime

class ChatRequest(BaseModel):
    """Request model for chat endpoint."""
    user_prompt: str = Field(..., min_length=1, max_length=2000, description="User's message to the financial advisor")
    session_id: str = Field(..., description="Session identifier for conversation continuity")
    
    class Config:
        json_schema_extra = {
            "example": {
                "user_prompt": "What's the current price of AAPL stock?",
                "session_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }

class ChatResponse(BaseModel):
    """Response model for chat endpoint (for non-streaming responses)."""
    response: str = Field(..., description="Assistant's response")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: Optional[datetime] = Field(None, description="Response timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "response": "The current price of AAPL is $185.25 per share.",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }

class StreamChunk(BaseModel):
    """Model for streaming response chunks."""
    type: str = Field(..., description="Type of chunk: assistant_response, tool_execution, thinking, error, done")
    content: Optional[str] = Field(None, description="Chunk content")
    session_id: Optional[str] = Field(None, description="Session identifier")
    timestamp: Optional[datetime] = Field(None, description="Chunk timestamp")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional chunk details")
    
    class Config:
        json_schema_extra = {
            "example": {
                "type": "assistant_response",
                "content": "Based on current market data, AAPL is trading at...",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }

class SessionRequest(BaseModel):
    """Request model for session operations."""
    session_id: str = Field(..., description="Session identifier")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000"
            }
        }

class NewSessionResponse(BaseModel):
    """Response model for new session creation."""
    session_id: str = Field(..., description="Newly created session identifier")
    message: str = Field(..., description="Success message")
    created_at: Optional[datetime] = Field(None, description="Session creation timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "message": "Session created successfully",
                "created_at": "2025-01-15T10:00:00Z"
            }
        }

class SessionHistoryResponse(BaseModel):
    """Response model for session history."""
    session_id: str = Field(..., description="Session identifier")
    history: List[Dict[str, Any]] = Field(..., description="Conversation history")
    message_count: Optional[int] = Field(None, description="Total number of messages")
    
    class Config:
        json_schema_extra = {
            "example": {
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "history": [
                    {"role": "user", "content": "What's AAPL price?", "timestamp": "2025-01-15T10:00:00Z"},
                    {"role": "assistant", "content": "AAPL is trading at $185.25", "timestamp": "2025-01-15T10:00:05Z"}
                ],
                "message_count": 2
            }
        }

class UserProfile(BaseModel):
    """User profile model for personalization."""
    risk_tolerance: str = Field(default="moderate", description="User's risk tolerance: conservative, moderate, aggressive")
    investment_goals: List[str] = Field(default_factory=list, description="User's investment goals")
    portfolio: Dict[str, Any] = Field(default_factory=dict, description="User's current portfolio")
    annual_income: Optional[float] = Field(None, description="User's annual income")
    age: Optional[int] = Field(None, description="User's age")
    investment_experience: Optional[str] = Field(None, description="User's investment experience level")
    
    class Config:
        json_schema_extra = {
            "example": {
                "risk_tolerance": "moderate",
                "investment_goals": ["retirement", "wealth_building"],
                "portfolio": {"AAPL": 100, "TSLA": 50, "bonds": 200},
                "annual_income": 75000.0,
                "age": 35,
                "investment_experience": "intermediate"
            }
        }

class ErrorResponse(BaseModel):
    """Standard error response model."""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    session_id: Optional[str] = Field(None, description="Session identifier if applicable")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Error timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "error": "SessionNotFound",
                "message": "Session 123e4567-e89b-12d3-a456-426614174000 not found",
                "session_id": "123e4567-e89b-12d3-a456-426614174000",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    timestamp: datetime = Field(default_factory=datetime.utcnow, description="Health check timestamp")
    
    class Config:
        json_schema_extra = {
            "example": {
                "status": "healthy",
                "service": "GAgent Financial Advisor API",
                "version": "1.0.0",
                "timestamp": "2025-01-15T10:30:00Z"
            }
        }