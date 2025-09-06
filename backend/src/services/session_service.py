# backend/src/services/session_service.py
import uuid
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime, timezone
import logging

logger = logging.getLogger(__name__)

class SessionService:
    """
    Service to manage chat sessions and conversation history.
    For demo purposes, using in-memory storage. In production,
    this would use a database like Redis or PostgreSQL.
    """
    
    def __init__(self):
        # In-memory storage for demo
        self._sessions: Dict[str, Dict[str, Any]] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(self) -> str:
        """Create a new chat session and return session ID."""
        session_id = str(uuid.uuid4())
        
        async with self._lock:
            self._sessions[session_id] = {
                "id": session_id,
                "created_at": datetime.now(timezone.utc),
                "last_activity": datetime.now(timezone.utc),
                "messages": [],
                "user_profile": {
                    "risk_tolerance": "moderate",
                    "investment_goals": [],
                    "portfolio": {}
                }
            }
        
        logger.info(f"Created session: {session_id}")
        return session_id
    
    async def session_exists(self, session_id: str) -> bool:
        """Check if a session exists."""
        async with self._lock:
            return session_id in self._sessions
    
    async def delete_session(self, session_id: str) -> None:
        """Delete a session."""
        async with self._lock:
            if session_id not in self._sessions:
                raise ValueError(f"Session {session_id} not found")
            
            del self._sessions[session_id]
        
        logger.info(f"Deleted session: {session_id}")
    
    async def get_session_config(self, session_id: str) -> Dict[str, Any]:
        """Get configuration for LangGraph (thread_id, etc.)."""
        if not await self.session_exists(session_id):
            raise ValueError(f"Session {session_id} not found")
        
        return {
            "configurable": {
                "thread_id": session_id,
                "user_id": "demo_user"
            }
        }
    
    async def add_message(self, session_id: str, role: str, content: str) -> None:
        """Add a message to session history."""
        if not await self.session_exists(session_id):
            raise ValueError(f"Session {session_id} not found")
        
        message = {
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        async with self._lock:
            self._sessions[session_id]["messages"].append(message)
            self._sessions[session_id]["last_activity"] = datetime.now(timezone.utc)
        
        logger.debug(f"Added {role} message to session {session_id}")
    
    async def get_session_history(self, session_id: str) -> List[Dict[str, Any]]:
        """Get conversation history for a session."""
        if not await self.session_exists(session_id):
            raise ValueError(f"Session {session_id} not found")
        
        async with self._lock:
            return self._sessions[session_id]["messages"].copy()
    
    async def update_user_profile(self, session_id: str, profile_updates: Dict[str, Any]) -> None:
        """Update user profile information for personalization."""
        if not await self.session_exists(session_id):
            raise ValueError(f"Session {session_id} not found")
        
        async with self._lock:
            current_profile = self._sessions[session_id]["user_profile"]
            current_profile.update(profile_updates)
            self._sessions[session_id]["last_activity"] = datetime.now(timezone.utc)
        
        logger.info(f"Updated user profile for session {session_id}")
    
    async def get_user_profile(self, session_id: str) -> Dict[str, Any]:
        """Get user profile for personalization."""
        if not await self.session_exists(session_id):
            raise ValueError(f"Session {session_id} not found")
        
        async with self._lock:
            return self._sessions[session_id]["user_profile"].copy()
    
    async def cleanup(self) -> None:
        """Cleanup resources (for production, close DB connections, etc.)."""
        async with self._lock:
            session_count = len(self._sessions)
            self._sessions.clear()
        
        logger.info(f"Cleaned up {session_count} sessions")
    
    async def get_session_stats(self) -> Dict[str, Any]:
        """Get statistics about current sessions (for monitoring)."""
        async with self._lock:
            return {
                "total_sessions": len(self._sessions),
                "sessions": [
                    {
                        "id": session_id,
                        "created_at": session_data["created_at"].isoformat(),
                        "last_activity": session_data["last_activity"].isoformat(),
                        "message_count": len(session_data["messages"])
                    }
                    for session_id, session_data in self._sessions.items()
                ]
            }