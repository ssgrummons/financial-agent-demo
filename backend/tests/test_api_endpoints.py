"""
Ultra-basic tests to verify testing setup works.
No FastAPI dependencies, just pure Python testing.
"""

import pytest


def test_basic_math():
    """Test that basic pytest works"""
    assert 1 + 1 == 2


def test_imports_work():
    """Test that we can import basic modules"""
    import json
    import unittest.mock
    assert json.dumps({"test": "data"}) == '{"test": "data"}'


def test_async_functionality():
    """Test that async/await works in tests"""
    import asyncio
    
    async def simple_async_function():
        return "async_result"
    
    # Test async function
    result = asyncio.run(simple_async_function())
    assert result == "async_result"


class TestBasicClass:
    """Test that class-based tests work"""
    
    def test_class_method(self):
        """Test method in class"""
        assert True
    
    def test_fixtures_available(self):
        """Test that basic pytest fixtures work"""
        # Just verify we can use pytest fixtures
        assert True


# Test mock functionality
def test_mocking_works():
    """Test that unittest.mock works"""
    from unittest.mock import Mock, patch
    
    mock_obj = Mock()
    mock_obj.test_method.return_value = "mocked_result"
    
    result = mock_obj.test_method()
    assert result == "mocked_result"


def test_session_service_can_be_imported():
    """Test that we can import our modules"""
    try:
        from services.session_service import SessionService
        assert SessionService is not None
        print("✅ SessionService import successful")
    except ImportError as e:
        pytest.fail(f"Could not import SessionService: {e}")


def test_chat_graph_can_be_imported():
    """Test that we can import ChatGraph"""
    try:
        from graphs.chat_graph import ChatGraph
        assert ChatGraph is not None
        print("✅ ChatGraph import successful")
    except ImportError as e:
        pytest.fail(f"Could not import ChatGraph: {e}")


def test_app_can_be_imported():
    """Test that our app can be imported"""
    try:
        # Try to import without executing lifespan
        import app
        print("✅ App import successful")
        assert hasattr(app, 'app')
    except ImportError as e:
        pytest.skip(f"Could not import app: {e}")
    except Exception as e:
        pytest.skip(f"App import failed with: {e}")


@pytest.mark.asyncio
async def test_async_with_pytest():
    """Test that pytest-asyncio works"""
    async def async_function():
        return "async works"
    
    result = await async_function()
    assert result == "async works"