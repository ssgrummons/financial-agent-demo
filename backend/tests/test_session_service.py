"""
Tests for SessionService - manages conversation state and persistence.
Focuses on interface contracts and backward compatibility to resist database changes.
"""

import pytest
import asyncio
from unittest.mock import Mock, patch, AsyncMock
from typing import List, Dict, Any

from services.session_service import SessionService


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

async def temporary_session(session_service: SessionService):
    """Context manager for temporary test sessions"""
    session_id = await session_service.create_session()
    try:
        yield session_id
    finally:
        try:
            await session_service.delete_session(session_id)
        except:
            pass  # Ignore cleanup errors in tests


# ============================================================================
# INTERFACE CONTRACT TESTS - These should NEVER break
# ============================================================================

class TestSessionServiceInterface:
    """Test the public interface contract - critical for backward compatibility"""
    
    @pytest.mark.asyncio
    async def test_has_required_methods(self):
        """Test that SessionService has all required public methods"""
        service = SessionService()
        
        # Core session lifecycle methods
        assert hasattr(service, 'create_session')
        assert callable(service.create_session)
        
        assert hasattr(service, 'delete_session')
        assert callable(service.delete_session)
        
        assert hasattr(service, 'session_exists')
        assert callable(service.session_exists)
        
        # Message management methods
        assert hasattr(service, 'add_message')
        assert callable(service.add_message)
        
        assert hasattr(service, 'get_session_history')
        assert callable(service.get_session_history)
        
        # Configuration and cleanup
        assert hasattr(service, 'get_session_config')
        assert callable(service.get_session_config)
        
        assert hasattr(service, 'cleanup')
        assert callable(service.cleanup)

    @pytest.mark.asyncio
    async def test_create_session_returns_string_id(self):
        """Test create_session contract: returns a string session ID"""
        service = SessionService()
        
        session_id = await service.create_session()
        
        # Contract: must return a string ID
        assert isinstance(session_id, str)
        assert len(session_id) > 0
        
        await service.cleanup()

    @pytest.mark.asyncio  
    async def test_session_exists_returns_boolean(self):
        """Test session_exists contract: returns boolean"""
        service = SessionService()
        
        # Test with non-existent session
        result = await service.session_exists("nonexistent_session")
        assert isinstance(result, bool)
        assert result is False
        
        # Test with existing session
        session_id = await service.create_session()
        result = await service.session_exists(session_id)
        assert isinstance(result, bool)
        assert result is True
        
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_get_session_history_returns_list(self):
        """Test get_session_history contract: returns list of messages"""
        service = SessionService()
        
        session_id = await service.create_session()
        
        history = await service.get_session_history(session_id)
        
        # Contract: must return a list (even if empty)
        assert isinstance(history, list)
        
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_get_session_config_returns_dict(self):
        """Test get_session_config contract: returns dict"""
        service = SessionService()
        
        session_id = await service.create_session()
        
        config = await service.get_session_config(session_id)
        
        # Contract: must return a dict (even if empty)
        assert isinstance(config, dict)
        
        await service.cleanup()


# ============================================================================
# CORE FUNCTIONALITY TESTS - Key behaviors that must work
# ============================================================================

class TestSessionServiceCore:
    """Test core session management functionality"""
    
    @pytest.mark.asyncio
    async def test_session_lifecycle(self):
        """Test complete session lifecycle: create -> exists -> delete -> gone"""
        service = SessionService()
        
        # Create session
        session_id = await service.create_session()
        assert session_id is not None
        
        # Session should exist
        exists = await service.session_exists(session_id)
        assert exists is True
        
        # Delete session
        await service.delete_session(session_id)
        
        # Session should no longer exist
        exists = await service.session_exists(session_id)
        assert exists is False
        
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_message_storage_and_retrieval(self):
        """Test storing and retrieving conversation messages"""
        service = SessionService()
        
        session_id = await service.create_session()
        
        # Initially empty
        history = await service.get_session_history(session_id)
        assert len(history) == 0
        
        # Add messages
        await service.add_message(session_id, "user", "Hello")
        await service.add_message(session_id, "assistant", "Hi there!")
        await service.add_message(session_id, "user", "How are you?")
        
        # Retrieve history
        history = await service.get_session_history(session_id)
        assert len(history) == 3
        
        # Check message structure (basic contract)
        for message in history:
            assert isinstance(message, dict)
            # Basic fields that should exist regardless of storage implementation
            assert "role" in message or "type" in message  # Either is acceptable
            assert "content" in message or "message" in message  # Either is acceptable
        
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_conversation_ordering(self):
        """Test that messages maintain chronological order"""
        service = SessionService()
        
        session_id = await service.create_session()
        
        # Add messages in sequence
        messages = [
            ("user", "First message"),
            ("assistant", "Second message"),
            ("user", "Third message"),
            ("assistant", "Fourth message")
        ]
        
        for role, content in messages:
            await service.add_message(session_id, role, content)
        
        # Retrieve and verify order
        history = await service.get_session_history(session_id)
        assert len(history) == 4
        
        # Messages should be in chronological order
        # (The exact format may vary, but order should be preserved)
        retrieved_contents = []
        for msg in history:
            # Handle different possible field names
            content = msg.get("content") or msg.get("message") or str(msg)
            retrieved_contents.append(content)
        
        # Check that our test messages appear in order
        for i, (_, original_content) in enumerate(messages):
            assert original_content in str(retrieved_contents[i])
        
        await service.cleanup()


# ============================================================================
# ERROR HANDLING TESTS - Ensure graceful failure
# ============================================================================

class TestSessionServiceErrorHandling:
    """Test error handling and edge cases"""
    
    @pytest.mark.asyncio
    async def test_delete_nonexistent_session(self):
        """Test deleting session that doesn't exist"""
        service = SessionService()
        
        fake_session_id = "nonexistent_session_12345"
        
        # Should raise appropriate error (likely ValueError)
        with pytest.raises((ValueError, KeyError, Exception)) as exc_info:
            await service.delete_session(fake_session_id)
        
        # Error message should be meaningful
        error_msg = str(exc_info.value).lower()
        assert any(word in error_msg for word in ["not found", "exist", "invalid"])
        
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_operations_on_nonexistent_session(self):
        """Test operations on sessions that don't exist"""
        service = SessionService()
        
        fake_session_id = "nonexistent_session_12345"
        
        # session_exists should return False, not raise
        exists = await service.session_exists(fake_session_id)
        assert exists is False
        
        # Other operations should fail gracefully
        with pytest.raises((ValueError, KeyError, Exception)):
            await service.add_message(fake_session_id, "user", "test")
        
        with pytest.raises((ValueError, KeyError, Exception)):
            await service.get_session_history(fake_session_id)
        
        await service.cleanup()


# ============================================================================
# CONCURRENCY TESTS - Important for production use
# ============================================================================

class TestSessionServiceConcurrency:
    """Test concurrent access patterns"""
    
    @pytest.mark.asyncio
    async def test_concurrent_session_creation(self):
        """Test creating sessions concurrently"""
        service = SessionService()
        
        # Create multiple sessions concurrently
        tasks = [service.create_session() for _ in range(5)]
        session_ids = await asyncio.gather(*tasks)
        
        # All should be unique
        assert len(set(session_ids)) == 5
        
        # All should exist
        for session_id in session_ids:
            exists = await service.session_exists(session_id)
            assert exists is True
        
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_concurrent_message_addition(self):
        """Test adding messages to same session concurrently"""
        service = SessionService()
        
        session_id = await service.create_session()
        
        # Add messages concurrently
        tasks = [
            service.add_message(session_id, "user", f"Message {i}")
            for i in range(5)
        ]
        await asyncio.gather(*tasks)
        
        # All messages should be stored
        history = await service.get_session_history(session_id)
        assert len(history) >= 5  # At least 5, order might vary
        
        await service.cleanup()


# ============================================================================
# BACKWARD COMPATIBILITY TESTS - Ensure upgrades don't break
# ============================================================================

class TestSessionServiceCompatibility:
    """Test patterns that ensure backward compatibility"""
    
    @pytest.mark.asyncio
    async def test_method_signatures_unchanged(self):
        """Test that core method signatures are stable"""
        service = SessionService()
        
        # Test that methods can be called with expected parameters
        # (This test will break if signatures change incompatibly)
        
        session_id = await service.create_session()
        
        # Core methods should accept basic parameters
        await service.add_message(session_id, "user", "test message")
        history = await service.get_session_history(session_id) 
        config = await service.get_session_config(session_id)
        exists = await service.session_exists(session_id)
        
        # Should be able to pass additional kwargs (for forward compatibility)
        try:
            await service.add_message(session_id, "user", "test", extra_param="ignored")
        except TypeError:
            # If extra params not accepted, that's okay too
            pass
        
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_return_value_structure(self):
        """Test that return values maintain expected structure"""
        service = SessionService()
        
        session_id = await service.create_session()
        await service.add_message(session_id, "user", "test")
        
        # History should be iterable with dict-like items
        history = await service.get_session_history(session_id)
        assert hasattr(history, '__iter__')
        
        for item in history:
            # Each item should be dict-like (have keys/values)
            assert hasattr(item, 'keys') or hasattr(item, '__getitem__')
        
        # Config should be dict-like
        config = await service.get_session_config(session_id)
        assert hasattr(config, 'keys') or hasattr(config, '__getitem__')
        
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_session_id_format_stability(self):
        """Test that session ID format is consistent"""
        service = SessionService()
        
        # Create multiple sessions and check ID format consistency
        session_ids = []
        for _ in range(3):
            session_id = await service.create_session()
            session_ids.append(session_id)
        
        # All IDs should be strings
        for session_id in session_ids:
            assert isinstance(session_id, str)
            assert len(session_id) > 0
        
        # IDs should be reasonably unique (no exact duplicates)
        assert len(set(session_ids)) == len(session_ids)
        
        await service.cleanup()


# ============================================================================
# CLEANUP AND RESOURCE MANAGEMENT TESTS
# ============================================================================

class TestSessionServiceCleanup:
    """Test cleanup and resource management"""
    
    @pytest.mark.asyncio
    async def test_cleanup_method_exists_and_works(self):
        """Test that cleanup method works without errors"""
        service = SessionService()
        
        # Create some sessions
        session_ids = []
        for _ in range(3):
            session_id = await service.create_session()
            session_ids.append(session_id)
        
        # Cleanup should not raise errors
        await service.cleanup()
        
        # After cleanup, service might or might not still work
        # (Implementation dependent - some may reset, others may be reusable)

    @pytest.mark.asyncio
    async def test_multiple_cleanup_calls(self):
        """Test that multiple cleanup calls don't break anything"""
        service = SessionService()
        
        # Multiple cleanups should be safe
        await service.cleanup()
        await service.cleanup()
        await service.cleanup()
        
        # Should not raise exceptions


# ============================================================================
# INTEGRATION PATTERN TESTS
# ============================================================================

class TestSessionServiceIntegrationPatterns:
    """Test common usage patterns that will be used by the application"""
    
    @pytest.mark.asyncio
    async def test_typical_conversation_flow(self):
        """Test a typical conversation flow pattern"""
        service = SessionService()
        
        # 1. Create session (like API endpoint does)
        session_id = await service.create_session()
        
        # 2. Verify session exists (like middleware might do)
        assert await service.session_exists(session_id) is True
        
        # 3. Add user message (like chat endpoint does)
        await service.add_message(session_id, "user", "Hello, I need financial advice")
        
        # 4. Add assistant response (like chat endpoint does)
        await service.add_message(session_id, "assistant", "I'd be happy to help with your financial questions.")
        
        # 5. Continue conversation
        await service.add_message(session_id, "user", "What's the price of AAPL?")
        await service.add_message(session_id, "assistant", "AAPL is currently trading at $150.00")
        
        # 6. Retrieve history (like session endpoint does)
        history = await service.get_session_history(session_id)
        assert len(history) == 4
        
        # 7. Get config if needed
        config = await service.get_session_config(session_id)
        assert isinstance(config, dict)
        
        # 8. Clean up
        await service.delete_session(session_id)
        assert await service.session_exists(session_id) is False
        
        await service.cleanup()

    @pytest.mark.asyncio
    async def test_assignment_example_conversation(self):
        """Test conversation flow using assignment example queries"""
        service = SessionService()
        
        session_id = await service.create_session()
        
        # Simulate the assignment example conversation
        conversation = [
            ("user", "What's the current price of AAPL stock?"),
            ("assistant", "AAPL is currently trading at $150.00. This information is for educational purposes only and not financial advice."),
            ("user", "Analyze my portfolio: 50% AAPL, 30% TSLA, 20% bonds. What's my expected return?"),
            ("assistant", "Based on your allocation, your portfolio shows moderate diversification..."),
            ("user", "Suggest investments for a conservative investor with $10,000."),
            ("assistant", "For a conservative approach, consider a diversified portfolio...")
        ]
        
        # Add all messages
        for role, content in conversation:
            await service.add_message(session_id, role, content)
        
        # Verify conversation is stored correctly
        history = await service.get_session_history(session_id)
        assert len(history) == len(conversation)
        
        await service.cleanup()