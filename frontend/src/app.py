import streamlit as st
import os
import asyncio
from api_client import (
    get_response, 
    get_streaming_response, 
    get_global_session_id, 
    reset_global_session,
    create_new_global_session
)
from dotenv import load_dotenv
import threading
import queue

class ChatApp:
    def __init__(self, use_streaming: bool = True):
        load_dotenv()
        self.port = int(os.getenv("STREAMLIT_PORT", 8501))
        self.use_streaming = use_streaming
        self._initialize_session_state()
        self._setup_page_config()

    def _initialize_session_state(self) -> None:
        """Initialize the chat history and other state in session state."""
        if "messages" not in st.session_state:
            st.session_state["messages"] = []
        if "api_session_initialized" not in st.session_state:
            st.session_state["api_session_initialized"] = False
        if "demo_authenticated" not in st.session_state:
            st.session_state["demo_authenticated"] = False

    def _setup_page_config(self) -> None:
        """Configure the Streamlit page settings."""
        st.set_page_config(
            page_title="GAgent: Financial Services Agent", 
            layout="wide",
            initial_sidebar_state="expanded"
        )

    def _render_demo_auth(self) -> bool:
        """Simple demo authentication - just a click to enter."""
        if st.session_state.get("demo_authenticated", False):
            return True
            
        st.title("🏦 GAgent: Financial Services Agent")
        st.markdown("### Welcome to the Financial AI Demo")
        st.markdown("*This is a demonstration of our AI-powered financial advisor*")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            if st.button("🚀 Enter Demo", use_container_width=True, type="primary"):
                st.session_state["demo_authenticated"] = True
                st.rerun()
        
        st.markdown("---")
        st.markdown("**Demo Features:**")
        st.markdown("- Real-time AI financial advice")
        st.markdown("- Streaming responses")
        st.markdown("- Session management")
        st.markdown("- No registration required")
        
        return False

    def _initialize_api_session(self):
        """Initialize API session automatically."""
        if not st.session_state.get("api_session_initialized", False):
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                try:
                    new_session_id = loop.run_until_complete(create_new_global_session())
                    st.session_state["api_session_initialized"] = True
                    st.success(f"Connected to API! Session: {new_session_id[:8]}...")
                finally:
                    loop.close()
            except Exception as e:
                st.error(f"Failed to initialize API session: {str(e)}")

    def _display_chat_history(self) -> None:
        """Display all messages from the chat history."""
        for message in st.session_state["messages"]:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def _handle_user_input(self, user_input: str) -> None:
        """Handle user input and generate response."""
        # Add user message to session state
        st.session_state["messages"].append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)
        
        # Create a placeholder for the assistant's response
        with st.chat_message("assistant"):
            response_placeholder = st.empty()
            
            try:
                if self.use_streaming:
                    # Use streaming response with proper real-time updates
                    self._handle_streaming_response(user_input, response_placeholder)
                else:
                    # Use non-streaming response (backward compatibility)
                    response = get_response(user_input)
                    response_placeholder.markdown(response)
                    st.session_state["messages"].append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = str(e)
                st.error(f"Error: {error_msg}")
                response_placeholder.markdown(f"❌ Error: {error_msg}")
    
    def _handle_streaming_response(self, user_input: str, placeholder) -> None:
        """Handle streaming response with real-time updates using a queue-based approach."""
        full_response = ""
        response_queue = queue.Queue()
        error_occurred = False
        
        def run_async_streaming():
            """Run the async streaming in a separate thread."""
            nonlocal error_occurred
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            
            try:
                async def stream_chunks():
                    async for chunk in get_streaming_response(user_input):
                        response_queue.put(('chunk', chunk))
                    response_queue.put(('done', None))
                
                loop.run_until_complete(stream_chunks())
            except Exception as e:
                error_occurred = True
                response_queue.put(('error', str(e)))
            finally:
                loop.close()
        
        # Start streaming in a separate thread
        thread = threading.Thread(target=run_async_streaming)
        thread.start()
        
        # Process chunks as they arrive
        while True:
            try:
                # Check for new chunks with a timeout
                message_type, content = response_queue.get(timeout=0.1)
                
                if message_type == 'chunk':
                    full_response += content
                    placeholder.markdown(full_response)
                elif message_type == 'done':
                    break
                elif message_type == 'error':
                    full_response = f"Error: {content}"
                    placeholder.markdown(full_response)
                    break
                    
            except queue.Empty:
                # No new chunks yet, continue waiting
                continue
        
        # Wait for thread to complete
        thread.join()
        
        # Add the complete response to session state
        if full_response and not error_occurred:
            st.session_state["messages"].append({"role": "assistant", "content": full_response})

    def _render_sidebar_controls(self):
        """Render sidebar controls."""
        with st.sidebar:
            st.markdown("---")
            st.header("⚙️ Chat Settings")
            
            # Streaming toggle
            self.use_streaming = st.checkbox("Use streaming responses", value=True)
            
            # Session management
            st.subheader("📱 Session Info")
            session_id = get_global_session_id()
            if session_id:
                st.info(f"Session ID: {session_id[:8]}...")
            else:
                st.warning("No active session")
            
            # Session controls
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 New Session", use_container_width=True):
                    try:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        try:
                            new_session = loop.run_until_complete(create_new_global_session())
                            st.session_state["messages"] = []  # Clear local chat history
                            st.success(f"New session: {new_session[:8]}...")
                            st.rerun()
                        finally:
                            loop.close()
                    except Exception as e:
                        st.error(f"Failed to create new session: {str(e)}")
            
            with col2:
                if st.button("🗑️ Clear Chat", use_container_width=True):
                    st.session_state["messages"] = []
                    st.success("Chat history cleared!")
                    st.rerun()
            
            # Demo controls
            st.markdown("---")
            if st.button("🚪 Exit Demo", use_container_width=True):
                # Reset everything for demo exit
                st.session_state["demo_authenticated"] = False
                st.session_state["api_session_initialized"] = False
                st.session_state["messages"] = []
                reset_global_session()
                st.rerun()
            
            # Debug info
            with st.expander("🔧 Debug Info"):
                st.write("**Demo Status:**", "✅ Authenticated" if st.session_state.get("demo_authenticated") else "❌ Not authenticated")
                st.write("**API Session:**", "✅ Initialized" if st.session_state.get("api_session_initialized") else "❌ Not initialized")
                st.write("**Messages Count:**", len(st.session_state.get("messages", [])))

    def run(self) -> None:
        """Run the chat application."""
        # Check demo authentication first
        if not self._render_demo_auth():
            return  # User not authenticated, show demo entrance
        
        # User is authenticated, show main app
        st.title("🏦 GAgent: Financial Services Agent")
        st.markdown("*Your AI-powered financial advisor*")
        
        # Initialize API session if needed
        if not st.session_state.get("api_session_initialized", False):
            self._initialize_api_session()
        
        # Render sidebar controls
        self._render_sidebar_controls()
        
        # Show connection status
        if not st.session_state.get("api_session_initialized", False):
            st.warning("⚠️ API connection not initialized. Please wait...")
            return
        
        # Main chat interface
        st.markdown("---")
        
        # Display chat history
        self._display_chat_history()
        
        # Handle user input
        if user_input := st.chat_input("Ask me about investments, market trends, financial planning..."):
            self._handle_user_input(user_input)

def main() -> None:
    """Main function to run the chat application."""
    try:
        chat_app = ChatApp()
        chat_app.run()
    except Exception as e:
        st.error(f"Application error: {str(e)}")
        st.info("Please check your environment variables and try refreshing the page.")

if __name__ == "__main__":
    main()