from typing import TypedDict, List, Dict, Optional
from langchain_core.messages import AnyMessage
from langgraph.graph.message import add_messages

class ChatState(TypedDict):
    """Core chat state for financial advisor"""
    ## Required for Assistant Node
    messages: Optional[List[AnyMessage]] = None
    system_prompt: Optional[str] = None
    user_prompt: Optional[str] = None
    image_data: Optional[List[Dict]] = None
    
    ## Session Management
    session_id: str
    user_id: str
    user_profile: Optional[dict]  # Risk tolerance, portfolio, etc.
    conversation_context: Optional[dict]  # Recent topics, preferences
