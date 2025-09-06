from abc import ABC, abstractmethod
from typing import Optional, AsyncGenerator, Callable
import os
import json
import asyncio
import aiohttp
import logging
import uuid
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class APIClientInterface(ABC):
    @abstractmethod
    def get_response(self, user_input: str, session_id: Optional[str] = None) -> str:
        pass

    @abstractmethod
    async def get_streaming_response(self, user_input: str, session_id: Optional[str] = None, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        pass

class FinancialAPIClient(APIClientInterface):
    def __init__(self, base_url: Optional[str] = None, max_retries: int = 3, retry_delay: float = 1.0):
        load_dotenv()
        self.base_url = base_url or os.getenv("API_BASE_URL", "http://localhost:8000")
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.current_session_id = None

    def _get_headers(self) -> dict:
        """Get headers for API requests."""
        return {
            "Content-Type": "application/json"
        }

    def _get_or_create_session_id(self, session_id: Optional[str] = None) -> str:
        """Get existing session ID or create a new one."""
        if session_id:
            self.current_session_id = session_id
            return session_id
        elif self.current_session_id:
            return self.current_session_id
        else:
            # Create new session ID for this client instance
            self.current_session_id = str(uuid.uuid4())
            logger.info(f"Created new session: {self.current_session_id}")
            return self.current_session_id

    def get_response(self, user_input: str, session_id: Optional[str] = None) -> str:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        try:
            return loop.run_until_complete(self._get_response_async(user_input, session_id))
        finally:
            loop.close()

    async def get_streaming_response(self, user_input: str, session_id: Optional[str] = None, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        async for chunk in self._get_streaming_response_async(user_input, session_id, callback):
            yield chunk

    async def _get_response_async(self, user_input: str, session_id: Optional[str] = None) -> str:
        # For non-streaming, we'll collect all streaming chunks
        full_response = ""
        async for chunk in self._get_streaming_response_async(user_input, session_id):
            full_response += chunk
        return full_response

    async def _get_streaming_response_async(self, user_input: str, session_id: Optional[str] = None, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
        url = f"{self.base_url}/chat/stream"
        payload = {
            "user_prompt": user_input,
            "session_id": self._get_or_create_session_id(session_id)
        }

        for attempt in range(self.max_retries):
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(url, json=payload, headers=self._get_headers()) as response:
                        if response.status == 200:
                            async for chunk in self._handle_response_stream(response, callback):
                                yield chunk
                            return
                        else:
                            await self._log_and_retry(response, attempt)
            except Exception as e:
                if attempt < self.max_retries - 1:
                    logger.error(f"Streaming request error (attempt {attempt+1}): {str(e)}")
                    await asyncio.sleep(self.retry_delay)
                else:
                    raise Exception(f"Failed to get streaming response after {self.max_retries} attempts: {str(e)}")

    async def _handle_response_stream(self, response: aiohttp.ClientResponse, callback: Optional[Callable[[str], None]]) -> AsyncGenerator[str, None]:
        async for line in response.content:
            line = line.decode('utf-8').strip()
            if not line.startswith("data: "):
                continue
            data = line[6:]  # Remove "data: " prefix
            
            content = self._parse_stream_line(data)
            if content is not None:
                if callback:
                    callback(content)
                yield content

    def _parse_stream_line(self, data: str) -> Optional[str]:
        """Parse a single stream line and return content if it's an assistant response."""
        try:
            json_data = json.loads(data)
            
            # Check the type field to determine how to handle this chunk
            message_type = json_data.get('type', '')
            
            if message_type == 'assistant_response':
                # Return the content from assistant responses
                return json_data.get('content', '')
            elif message_type == 'done':
                # Signal that streaming is complete by returning None
                # The generator will naturally stop yielding
                return None
            elif message_type == 'error':
                # Handle error messages
                error_msg = json_data.get('content', 'Unknown error occurred')
                logger.error(f"Streaming error: {error_msg}")
                raise Exception(f"Stream error: {error_msg}")
            else:
                # Unknown message type, log it but don't yield anything
                logger.debug(f"Unknown message type '{message_type}': {data}")
                return None
                
        except json.JSONDecodeError:
            logger.warning(f"Failed to parse JSON: {data}")
            return None

    async def _log_and_retry(self, response: aiohttp.ClientResponse, attempt: int):
        error_text = await response.text()
        logger.error(f"API error (attempt {attempt+1}/{self.max_retries}): {response.status} - {error_text}")
        if attempt < self.max_retries - 1:
            await asyncio.sleep(self.retry_delay)
        else:
            raise Exception(f"API error: {response.status} - {error_text}")

    def get_session_id(self) -> Optional[str]:
        """Get the current session ID."""
        return self.current_session_id

    def reset_session(self) -> str:
        """Reset the session by creating a new session ID."""
        self.current_session_id = str(uuid.uuid4())
        logger.info(f"Reset to new session: {self.current_session_id}")
        return self.current_session_id

    async def create_new_session(self) -> str:
        """Create a new session using the API endpoint."""
        url = f"{self.base_url}/sessions"  # Updated to match OpenAPI spec
        
        async with aiohttp.ClientSession() as session:
            async with session.post(url, headers=self._get_headers()) as response:
                if response.status == 200:
                    data = await response.json()
                    new_session_id = data.get("session_id")
                    self.current_session_id = new_session_id
                    logger.info(f"Created new API session: {new_session_id}")
                    return new_session_id
                else:
                    error_text = await response.text()
                    raise Exception(f"Failed to create new session: {response.status} - {error_text}")

class APIClientFactory:
    @staticmethod
    def create_client(client_type: str = "financial", **kwargs) -> APIClientInterface:
        if client_type.lower() in ["financial", "rag"]:
            return FinancialAPIClient(**kwargs)
        raise ValueError(f"Unsupported client type: {client_type}")

# Global client instance to maintain session across calls
_global_client = None

def _get_global_client() -> FinancialAPIClient:
    """Get or create a global client instance to maintain session."""
    global _global_client
    if _global_client is None:
        _global_client = APIClientFactory.create_client()
    return _global_client

# Backward compatibility functions
def get_response(user_input: str, session_id: Optional[str] = None) -> str:
    client = _get_global_client()
    return client.get_response(user_input, session_id)

async def get_streaming_response(user_input: str, session_id: Optional[str] = None, callback: Optional[Callable[[str], None]] = None) -> AsyncGenerator[str, None]:
    client = _get_global_client()
    async for chunk in client.get_streaming_response(user_input, session_id, callback):
        yield chunk

def reset_global_session() -> str:
    """Reset the global session for testing."""
    client = _get_global_client()
    return client.reset_session()

def get_global_session_id() -> Optional[str]:
    """Get the current global session ID."""
    client = _get_global_client()
    return client.get_session_id()

async def create_new_global_session() -> str:
    """Create a new session using the API endpoint."""
    client = _get_global_client()
    return await client.create_new_session()